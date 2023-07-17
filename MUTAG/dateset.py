import torch
from torch_geometric.data import Data,InMemoryDataset
import numpy as np
import os.path as osp
import networkx as  nx

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def net2Data(graph,label):
    x = torch.ones((graph.number_of_nodes(),10), dtype=torch.float)

    # adj是图G的邻接矩阵的稀疏表示，左边节点对代表一条边，右边是边的值，adj是对称矩阵。
    adj = nx.to_scipy_sparse_matrix(graph).tocoo()

    # row是adj中非零元素所在的行索引
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    # col是adj中非零元素所在的列索引。
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)

    # 将行和列进行拼接，shape变为[2, num_edges], 包含两个列表，第一个是row, 第二个是col
    edge_index = torch.stack([row, col], dim=0)
    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    if data.edge_attr == None:
        edge_att = torch.ones(data.num_edges, dtype=torch.float)
        data.edge_attr = edge_att
    return data




class IsAcyclicDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(IsAcyclicDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        datasets = []
        labels = []
        for i in range(2, 9):
            for j in range(2, 9):
                datasets.append(nx.grid_2d_graph(i, j))
                labels.append(0)
        for i in range(3, 65):
            datasets.append(nx.cycle_graph(i))
            labels.append(0)

        for i in range(20):
            datasets.append(nx.cycle_graph(3))
            labels.append(0)

        for i in range(2, 65):
            datasets.append(nx.wheel_graph(i))
            labels.append(0)

        for i in range(2, 35):
            datasets.append(nx.circular_ladder_graph(i))
            labels.append(0)

        for i in range(2, 65):
            datasets.append(nx.star_graph(i))
            labels.append(1)

        g = nx.balanced_tree(2, 5)
        datasets.append(g)
        labels.append(1)
        for i in range(62, 2, -1):
            g.remove_node(i)
            datasets.append(g)
            labels.append(1)

        for i in range(3, 65):
            datasets.append(nx.path_graph(i))
            labels.append(1)

        for i in range(3, 5):
            for j in range(5, 65):
                datasets.append(nx.full_rary_tree(i, j))
                labels.append(1)

        data_list = []
        for i, graph in enumerate(datasets):
            data_list.append(net2Data(graph, labels[i]))
        data = data_list if self.pre_transform is None else self.pre_transform(data_list)
        torch.save(self.collate(data), self.processed_paths[0])





# data_is = IsAcyclicDataset('data', name='Is_Acyclic')
#
# print(data_is[0].edge_attr)