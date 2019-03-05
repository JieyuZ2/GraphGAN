import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils
import collections
import tqdm
import copy
import numpy as np
import time
import os
import multiprocessing
import config
import evaluation.eval_link_prediction as elp


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.enabled = True


def l2_loss(tensor):
    return 0.5*((tensor ** 2).sum())


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numberation stablity
    return e_x / e_x.sum()


class Generator(nn.Module):
    def __init__(self, lambda_gen, node_emd_init):
        super(Generator, self).__init__()
        self.lambda_gen = lambda_gen
        self.n_node, self.emd_size = node_emd_init.shape
        self.node_emd = nn.Embedding.from_pretrained(torch.from_numpy(node_emd_init), freeze=False)
        self.bias_vector = nn.Parameter(torch.zeros(self.n_node))
        self.double()

    def forward(self, node_ids, neighbor_ids, reward):
        node_ids = torch.LongTensor(node_ids).cuda()
        neighbor_ids = torch.LongTensor(neighbor_ids).cuda()
        # reward = torch.DoubleTensor(reward).cuda()

        node_embedding = self.node_emd(node_ids)
        neighbor_node_embedding = self.node_emd(neighbor_ids)
        bias = self.bias_vector.gather(0, neighbor_ids)
        score = (node_embedding*neighbor_node_embedding).sum(dim=1)+bias
        prob = score.sigmoid().clamp(1e-5, 1)
        loss = -(prob.log()*reward).mean() + self.lambda_gen * (l2_loss(node_embedding)+l2_loss(neighbor_node_embedding)+l2_loss(bias))
        return loss

    def get_all_score(self):
        with torch.no_grad():
            node_emd = self.node_emd.weight.data
            score = node_emd.mm(node_emd.t()) + self.bias_vector.data
            return score.data.cpu().numpy()

    def save(self, index2node, path):
        embeddings = self.node_emb.weight.data.numpy()
        fout = open(path, 'w', encoding="UTF-8")
        fout.write('{} {}\n'.format(self.N, self.dim))
        for i in range(self.N):
            fout.write('{} {}\n'.format(index2node[i], ' '.join(str(n) for n in embeddings[i])))
        fout.close()


class Discriminator(nn.Module):
    def __init__(self, lambda_dis, node_emd_init):
        super(Discriminator, self).__init__()
        self.lambda_dis = lambda_dis
        self.n_node, self.emd_size = node_emd_init.shape
        self.node_emd = nn.Embedding.from_pretrained(torch.from_numpy(node_emd_init), freeze=False)
        self.bias_vector = nn.Parameter(torch.zeros(self.n_node))
        self.double()

    def forward(self, node_ids, neighbor_ids, label):
        node_ids = torch.LongTensor(node_ids).cuda()
        neighbor_ids = torch.LongTensor(neighbor_ids).cuda()
        label = torch.DoubleTensor(label).cuda()
        node_embedding = self.node_emd(node_ids)
        neighbor_node_embedding = self.node_emd(neighbor_ids)
        bias = self.bias_vector.gather(0, neighbor_ids)
        score = (node_embedding*neighbor_node_embedding).sum(dim=1)+bias
        loss = (F.binary_cross_entropy_with_logits(score, label)).mean() + \
               self.lambda_dis * (l2_loss(node_embedding)+l2_loss(neighbor_node_embedding)+l2_loss(bias))
        return loss

    def get_reward(self, node_ids, neighbor_ids):
        with torch.no_grad():
            node_ids = torch.LongTensor(node_ids).cuda()
            neighbor_ids = torch.LongTensor(neighbor_ids).cuda()
            node_embedding = self.node_emd(node_ids)
            neighbor_node_embedding = self.node_emd(neighbor_ids)
            bias = self.bias_vector.gather(0, neighbor_ids)
            score = (node_embedding * neighbor_node_embedding).sum(dim=1) + bias
            reward = (score.data.clamp(-10, 10).exp() + 1).log()
            return reward.data

    def save(self, index2node, path):
        embeddings = self.node_emb.weight.data.numpy()
        fout = open(path, 'w', encoding="UTF-8")
        fout.write('{} {}\n'.format(self.N, self.dim))
        for i in range(self.N):
            fout.write('{} {}\n'.format(index2node[i], ' '.join(str(n) for n in embeddings[i])))
        fout.close()


class GraphGan(object):
    def __init__(self):
        """initialize the parameters, prepare the data and build the network"""

        self.n_node, self.linked_nodes = utils.read_edges(config.train_filename, config.test_filename)
        self.root_nodes = [i for i in range(self.n_node)]
        self.discriminator = None
        self.generator = None
        assert self.n_node == config.n_node
        print("start reading initial embeddings")
        # read the initial embeddings
        self.node_embed_init_d = utils.read_emd(filename=config.pretrain_emd_filename_d, n_node=config.n_node, n_embed=config.n_embed)
        self.node_embed_init_g = utils.read_emd(filename=config.pretrain_emd_filename_g, n_node=config.n_node, n_embed=config.n_embed)
        print("finish reading initial embeddings")
        # use the BFS to construct the trees
        print("Constructing Trees")
        if config.app == "recommendation":
            self.mul_construct_trees_for_recommend(self.user_nodes)
        else:  # classification
            self.mul_construct_trees(self.root_nodes)
        config.max_degree = utils.get_max_degree(self.linked_nodes)

        self.generator = Generator(lambda_gen=config.lambda_gen, node_emd_init=self.node_embed_init_g)
        self.discriminator = Discriminator(lambda_dis=config.lambda_dis, node_emd_init=self.node_embed_init_d)
        self.generator.cuda()
        self.discriminator.cuda()
        self.all_score = None

    def update_all_score(self):
        self.all_score = self.generator.get_all_score()

    def mul_construct_trees(self, nodes):
        """use the multiprocessing to speed the process of constructing trees

        Args:
            nodes: list, the root of the trees
        """

        if config.use_mul:
            t1 = time.time()
            cores = multiprocessing.cpu_count() // 2
            pool = multiprocessing.Pool(cores)
            new_nodes = []
            node_per_core = self.n_node // cores
            for i in range(cores):
                if i != cores - 1:
                    new_nodes.append(nodes[i*node_per_core:(i+1)*node_per_core])
                else:
                    new_nodes.append(nodes[i*node_per_core:])

            self.trees = {}
            trees_result = pool.map(self.construct_tree, new_nodes)
            for tree in trees_result:
                self.trees.update(tree)
            t2 = time.time()
            print(t2-t1)
        else:
            self.trees = self.construct_tree(nodes)
        # serialized the trees to the disk
        print("Dump the trees to the disk")

    def construct_tree(self, nodes):
        """use the BFS algorithm to construct the trees

        Works OK.
        test case: [[0,1],[0,2],[1,3],[1,4],[2,4],[3,5]]
        "Node": [father, children], if node is the root, then the father is itself.
        Args:
            nodes:
        Returns:
            trees: dict, <key, value>:<node_id, {dict(store the neighbor nodes)}>
        """
        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            tmp = copy.copy(self.linked_nodes[root])
            trees[root][root] = [root] + tmp
            if len(tmp) == 0:  # isolated user
                continue
            queue = collections.deque(tmp)  # the nodes in this queue all are items
            for x in tmp:
                trees[root][x] = [root]
            used_nodes = set(tmp)
            used_nodes.add(root)

            while len(queue) > 0:
                cur_node = queue.pop()
                used_nodes.add(cur_node)
                for sub_node in self.linked_nodes[cur_node]:
                    if sub_node not in used_nodes:
                        queue.appendleft(sub_node)
                        used_nodes.add(sub_node)
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
        return trees

    def sample_for_gan(self, root, sample_num, sample_for_dis: bool):
        """ sample the nodes from the tree

        Args:
            root: int, root, the query
            tree: dict, tree information
            sample_num: the number of the desired sampling nodes
            all_score: pre-computed score matrix, speed the sample process
            sample_for_dis: bool, indicates it is sampling for generator or discriminator
        Returns:
            sample: list, include the index of the sampling nodes
        """

        tree = self.trees[root]
        sample = []
        trace = []
        all_score = self.all_score

        while len(sample) < sample_num:
            node_select = root
            node_father = -1
            flag = 1    # to exclude root
            if not sample_for_dis:
                trace.append([node_select])
            while True:
                node_neighbor = tree[node_select][1:] if flag == 1 else tree[node_select]
                flag = 0
                if sample_for_dis:  # only sample the negative examples for discriminator, thus should exclude the root node tobe sampled
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                if not node_neighbor:  # the tree only has the root
                    return sample, trace
                prob = softmax(all_score[node_select, node_neighbor])
                node_check = np.random.choice(node_neighbor, size=1, p=prob)[0]
                if not sample_for_dis:
                    trace[-1].append(node_check)
                if node_check == node_father:
                    sample.append(node_select)
                    break
                node_father = node_select
                node_select = node_check
        return sample, trace

    def sample_for_d(self):
        center_nodes = []
        neighbor_nodes = []
        labels = []
        for u in self.root_nodes:
            if np.random.rand() < config.update_ratio:
                pos = self.linked_nodes[u]  # pos samples
                if len(pos) < 1:
                    continue
                neg, _ = self.sample_for_gan(u, len(pos), sample_for_dis=True)
                if len(neg)!=len(pos):
                    continue
                neighbors = pos + neg
                center_nodes.extend(len(neighbors) * [u])
                neighbor_nodes.extend(neighbors)
                labels.extend(len(pos) * [1] + len(neg) * [0])
        return center_nodes, neighbor_nodes, labels

    def get_batch_data(self, left, right, label, batch_size):
        for start in range(0, len(label), batch_size):
            end = start + batch_size
            yield left[start:end], right[start:end], label[start:end]

    def train_gan(self):
        """train the whole graph gan network"""

        g_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=config.lr_gen)
        d_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=config.lr_dis)

        print("Evaluation")
        self.write_emb_to_txt()
        result = self.eval_test()  # evaluation
        print(result)
        for epoch in range(config.max_epochs):

            self.update_all_score()

            for d_epoch in range(config.max_epochs_dis):
                if d_epoch % config.gen_for_d_iters == 0:  # every gen_for_d_iters round, we generate new data
                    center_nodes, neighbor_nodes, labels = self.sample_for_d()
                #  traverse the whole training dataset sequentially, train the discriminator
                for batch_data in self.get_batch_data(center_nodes, neighbor_nodes, labels, config.batch_size_dis):
                    d_optimizer.zero_grad()
                    loss = self.discriminator(*batch_data)
                    loss.backward()
                    d_optimizer.step()

            for g_epoch in range(config.max_epochs_gen):
                cnt = 0
                root_nodes = []  # just for record how many trees that have been traversed
                macro_trace = []  # the trace when sampling the nodes, from the root to leaf  bach to leaf's father. e.g.: 0 - 1 - 2 -1
                for root_node in self.root_nodes:  # random update trees
                    if np.random.rand() < config.update_ratio:
                        # sample the nodes according to our method.
                        # feed the reward from the discriminator and the sampled nodes to the generator.
                        if cnt % config.gen_update_iter == 0 and cnt > 0:
                            # generate update pairs along the path, [q_node, rel_node]
                            center_nodes, neighbor_nodes = self.generate_window_pairs(macro_trace)
                            rewards = self.discriminator.get_reward(center_nodes, neighbor_nodes)

                            for batch_data in self.get_batch_data(center_nodes, neighbor_nodes, rewards, config.batch_size_gen):
                                g_optimizer.zero_grad()
                                loss = self.generator(*batch_data)
                                loss.backward()
                                g_optimizer.step()

                            self.update_all_score()
                            root_nodes = []
                            macro_trace = []
                            cnt = 0
                        _, trace = self.sample_for_gan(root_node, config.n_sample_gen, sample_for_dis=False)
                        # if len(sample) < config.n_sample_gen:
                        #     cnt = len(root_nodes)
                        #     continue
                        macro_trace.extend(trace)
                        root_nodes.append(root_node)
                        cnt = cnt + 1

            print("Evaluation")
            self.write_emb_to_txt()
            result = self.eval_test()  # evaluation
            print(result)

    def generate_window_pairs(self, paths):
        """
        given a sample path list from root to a sampled node, generate all the pairs corresponding to the windows size
        e.g.: [1, 0, 2, 4, 2], window_size = 2 -> [1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]
        :param sample_path:
        :return:
        """
        left, right = [], []
        for path in paths:
            path = path[:-1]
            for i, center_node in enumerate(path):
                for j in range(max(i - config.window_size, 0), min(i + config.window_size + 1, len(path))):
                    if i != j:
                        left.append(center_node)
                        right.append(path[j])
        return left, right

    def padding_neighbor(self, neighbor):
        return neighbor + (config.max_degree - len(neighbor)) * [0]

    def save_emb(self, node_embed, filename):
        np.savetxt(filename, node_embed, fmt="%10.5f", delimiter='\t')

    def write_emb_to_txt(self):
        """write the emd to the txt file"""

        modes = [self.generator, self.discriminator]
        for i in range(2):
            node_embed = modes[i].node_emd.weight.data.cpu().numpy()
            a = np.array(range(self.n_node)).reshape(-1, 1)
            node_embed = np.hstack([a, node_embed])
            node_embed_list = node_embed.tolist()
            node_embed_str = ["\t".join([str(x) for x in line]) + "\n" for line in node_embed_list]
            with open(config.emb_filenames[i], "w+") as f:
                lines = [str(config.n_node) + "\t" + str(config.n_embed) + "\n"] + node_embed_str
                f.writelines(lines)

    def eval_test(self):
        """do the evaluation when training

        :return:
        """
        results = []
        if config.app == "link_prediction":
            for i in range(2):
                LPE = elp.LinkPredictEval(config.emb_filenames[i], config.test_filename, config.test_neg_filename, config.n_node, config.n_embed)
                result = LPE.eval_link_prediction()
                results.append(config.modes[i] + ":" + str(result))
        to_print = '\t'.join(results)+'\n'
        with open(config.result_filename, mode="a+") as f:
            f.write(to_print)
        return to_print


if __name__ == "__main__": 
    g_g = GraphGan()
    g_g.train_gan()
