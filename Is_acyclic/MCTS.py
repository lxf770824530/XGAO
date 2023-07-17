import sys
import math
import random
import numpy as np
import os.path as osp
import torch
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Data
import time

from GNNs import GCN
from utils import regular_graph_generate, Rectify, Fix_nodes_index, Fix_mask_single_false, Is_exixt_in_list, Draw_graph

from Utils.mask_learner import Mask_learner
from Utils.SA import Temperature_State, Metropolos
from Utils.lagrangian_optimization import LagrangianOptimization

from main import arg_parse
args = arg_parse()

TAR_CLASS = args.explain_class

GNN_MODEL = GCN(10, 2)
model_name = args.dataset + '_gcn_model.pth'
model_save_path = osp.join('checkpoint', args.dataset, model_name)
GNN_MODEL.load_state_dict(torch.load(model_save_path))

SA_STATE = Temperature_State(args.temperature)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class Node(object):
    def __init__(self, Graph):
        self.parent = None
        self.children = []

        self.visit_times = 0
        self.quality_value = 0.0
        self.is_root_node = False
        self.is_leaf_node = False
        self.is_expanded_i = True

        #graph
        self.graph = Graph
        self.mask_learner = Mask_learner()
        self.subgraphs = []
        self.subgraphs_num = 0

        GNN_MODEL.eval()
        batch = torch.zeros((1, len(self.graph.x)), dtype=torch.int64)[0]
        prediction = GNN_MODEL(self.graph.x, self.graph.edge_index, self.graph.edge_attr, batch)

        self.target_probability = prediction[0][TAR_CLASS].item()

        target_label = torch.full(size=(1, 1), fill_value=TAR_CLASS)
        target_label = target_label.squeeze(0).long()
        last_loss = 100.0
        mask_learner_optimizer = torch.optim.Adam(self.mask_learner.parameters(), lr=0.001)
        CE_loss = CrossEntropyLoss()
        lagrangian_optimization = LagrangianOptimization(mask_learner_optimizer, device)
        for epoch in range(10):
            batch = torch.zeros((1, len(self.graph.x)), dtype=torch.int64)[0]
            self.mask_learner.train()
            mask, penalty = self.mask_learner(self.graph)
            mask = mask.bool()
            edge_index = self.graph.edge_index[torch.stack((mask,mask))].view(2,-1)
            edge_index = Fix_nodes_index(edge_index)
            edge_attr = self.graph.edge_attr[mask]
            masked_graph = Data(x=self.graph.x, edge_index=edge_index, edge_attr=edge_attr)
            GNN_MODEL.eval()
            prediction = GNN_MODEL(masked_graph.x, masked_graph.edge_index, masked_graph.edge_attr, batch)
            Loss_ce = CE_loss(prediction, target_label)
            g = torch.relu(Loss_ce - 0.03).mean()
            f = penalty
            Loss = lagrangian_optimization.update(f, g)

            # log = 'Epoch: {:03d}, Train loss: {:.5f}, prediction: {:.5f}'
            # print(log.format(epoch + 1, Loss, prediction[0].max()))
            if epoch == 0:
                log = 'Epoch: {:03d}, Train loss: {:.5f}, prediction: {:.5f}'
                print(log.format(epoch + 1, Loss, prediction[0].max()))
            elif epoch % 10 == 9:
                log = 'Epoch: {:03d}, Train loss: {:.5f}, prediction: {:.5f}'
                print(log.format(epoch + 1, Loss, prediction[0].max()))

            if Loss <= last_loss:
                last_loss = Loss
                last_mask = mask
            elif Loss > last_loss and prediction[0][TAR_CLASS] >= 0.99:
                break
        self.mask = last_mask.float()

        self.mask_list = []
        if len(self.mask_list) == 0:
            for i,m in enumerate(self.mask):
                if m.item() == 0.0:
                    all_true_mask = torch.ones_like(self.mask)
                    all_true_mask[i] = 0.0
                    fixed_mask_i = Fix_mask_single_false(all_true_mask, self.graph.edge_index)
                    if Is_exixt_in_list(fixed_mask_i, self.mask_list):
                        self.mask_list.append(fixed_mask_i)


        for m in self.mask_list:
            m_bool = m.bool()
            edge_index = self.graph.edge_index[torch.stack((m_bool, m_bool))].view(2, -1)
            edge_index = Fix_nodes_index(edge_index)
            edge_attr = self.graph.edge_attr[m_bool]
            subgraph = Data(x=self.graph.x, edge_index=edge_index, edge_attr=edge_attr)
            rectified_graph = Rectify(subgraph, self.graph)
            GNN_MODEL.eval()
            batch = torch.zeros((1, len(rectified_graph.x)), dtype=torch.int64)[0]
            rectified_graph_prediction = GNN_MODEL(rectified_graph.x, rectified_graph.edge_index, rectified_graph.edge_attr, batch)
            rectified_graph_prediction_t = rectified_graph_prediction[0][TAR_CLASS].item()

            discriminate_result = Metropolos(self.get_probability(), rectified_graph_prediction_t, SA_STATE)
            if discriminate_result:
                self.subgraphs.append(rectified_graph)
            else:
                try:
                    self.mask_list.remove(m)
                except RuntimeError:
                    print()


        self.subgraphs_num = len(self.subgraphs)
        if self.subgraphs_num == 0:
            self.set_leaf_node()

    def is_expanded(self):

        if len(self.mask_list) == len(self.graph.edge_index)/2:
            self.is_expanded_i = False

        return self.is_expanded_i


    def remove_subgraph(self, subgraph):
        self.subgraphs.remove(subgraph)

    def set_root_node(self):
        self.is_root_node = True

    def set_leaf_node(self):
        self.is_leaf_node = True

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def is_all_expand(self):
        return (len(self.children) == len(self.subgraphs)) and (len(self.children) >= 1)

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_add_n(self, n):
        self.quality_value += n

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)

    # graph operation
    def get_graph(self):
        return self.graph

    def get_mask(self):
        return self.mask

    def get_subgraphs(self):
        return self.subgraphs

    def get_subgraphs_num(self):
        return self.subgraphs_num

    def get_probability(self):
        return self.target_probability

    def __repr__(self):
        return "Node: {}, Q/N: {}/{}".format(
            hash(self), self.quality_value, self.visit_times)

def tree_policy(node):
    """
    蒙特卡罗树搜索的Selection和Expansion阶段，传入当前需要开始搜索的节点（例如根节点），根据exploration/exploitation算法返回最好的需要expend的节点，注意如果节点是叶子结点直接返回。
    基本策略是先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找权衡过exploration/exploitation的UCB值最大的，如果UCB值相等则随机选。
    """
    input_node = node
    # Check if the current node is the leaf node
    while SA_STATE.is_continue():

        if node.is_all_expand():
            node = best_child(node, True)
            if node.is_leaf_node == True:
                break
        else:
            # Return the new sub node
            sub_node = expand(node)
            return sub_node

    # Return the leaf node
    return node

def expand(node):
    """
    输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
    """

    subgraphs = node.get_subgraphs()

    tried_sub_node_graph = [sub_node.get_graph() for sub_node in node.get_children()]
    i = 0

    pre_expand_graph = subgraphs[i]
    while pre_expand_graph in tried_sub_node_graph:
        i += 1
        if len(subgraphs) > i:
            pre_expand_graph = subgraphs[i]

    subnode = Node(pre_expand_graph)


    if subnode.is_expanded():
        node.add_child(subnode)
        return subnode
    else:
        node.remove_subgraph(subnode.get_graph())
        node.subgraphs_num-=1
        if node.subgraphs_num == 0:
            node.set_leaf_node()
        return False


def best_child(node, is_exploration):
    """
    使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
    """

    # TODO: Use the min float value
    best_score = -sys.maxsize
    best_sub_node = None

    root_node = node
    while root_node.is_root_node is not True:
        root_node = root_node.get_parent()
    all_visit_num = root_node.get_visit_times()

    # Travel all sub nodes to find the best one
    for sub_node in node.get_children():

        # Ignore exploration for inference
        if is_exploration:
            C = 1 / math.sqrt(2.0)
        else:
            C = 0.0

        # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
        try:
            left = sub_node.get_quality_value() / sub_node.get_visit_times()
        except ZeroDivisionError:
            print()
        right = 2.0 * math.log(all_visit_num) / sub_node.get_visit_times()
        score = left + C * math.sqrt(right)

        if score > best_score:
            best_sub_node = sub_node
            best_score = score

    return best_sub_node

def default_policy(node):

    current_node = node
    # 向下搜寻3轮 返回reward
    simulation_round = 3
    for i in range(simulation_round):
        current_node_subgraph = current_node.get_subgraphs()
        if len(current_node_subgraph) == 0:
            break
        current_subgraph = random.choice([choice for choice in current_node_subgraph])
        current_node = Node(current_subgraph)
    reward_value = current_node.get_probability()

    return reward_value

def backup(node, reward):
    """
    蒙特卡洛树搜索的Backpropagation阶段，输入前面获取需要expend的节点和新执行Action的reward，反馈给expend节点和上游所有节点并更新对应数据。
    """

    # Update util the root node
    while node != None:
        # Update the visit times
        node.visit_times_add_one()

        # Update the quality value
        node.quality_value_add_n(reward)

        # Change the node to the parent node
        node = node.parent


def monte_carlo_tree_search(node):

    computation_budget = 100  # 对每一层的节点  总共搜索1000次 最后算哪个节点最合适


    # Run as much as possible under the computation budget
    for i in range(computation_budget):
        # 1. Find the best node to expand
        if node.is_leaf_node == False:
            expand_node = tree_policy(node)
        else:
            return node
        if expand_node == False:
            continue

        # 2. Random run to add node and get reward
        reward = default_policy(expand_node)

        # 3. Update all passing nodes with reward
        backup(expand_node, reward)

    # N. Get the best next node

    best_next_node = best_child(node, False)

    return best_next_node


def Run_MCTS():

    # Create the initialized state and initialized node

    init_node = Node(regular_graph_generate(args))
    init_node.set_root_node()

    current_node = init_node
    i=0
    # Set the rounds to play
    while SA_STATE.is_continue():
        print("Play round: {}".format(i + 1))
        SA_STATE.temperature_dropping()
        i += 1

        current_node = monte_carlo_tree_search(current_node)
        if current_node.is_leaf_node == True:
            break


    print(current_node.graph)
    batch = torch.zeros((1, len(current_node.graph.x)), dtype=torch.int64)[0]
    prediction = GNN_MODEL(current_node.graph.x, current_node.graph.edge_index, current_node.graph.edge_attr, batch)
    print('Prediction:',prediction[0].max())
    Draw_graph(current_node.graph)


if __name__ == "__main__":
    star_time = time.time()
    Run_MCTS()
    end_time = time.time()
    print('run-time',end_time-star_time)