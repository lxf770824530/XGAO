import torch
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.nn.inits import glorot, uniform

from torch.nn import ReLU, Sequential, LayerNorm
from torch.nn import Parameter
import numpy as np


class SingleGCNLayer(MessagePassing):


    def __init__(self, in_dim, out_dim, num_relations = 4):
        super(SingleGCNLayer, self).__init__('add')

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations    #边的关系的类别数

        self.basis = Parameter(torch.Tensor(in_dim, out_dim * num_relations))
        self.bias = Parameter(torch.Tensor(num_relations, out_dim))

        self.process_message = Sequential(LayerNorm(out_dim),
                                          ReLU())

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_relations * self.in_dim

        glorot(self.basis)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type):
        """"""
        size = [x.shape[0], x.shape[0]]       #图神经网络消息传递时的size参数

        res = self.propagate(edge_index, x=x, edge_type=edge_type, size=size)

        return res

    def message(self, x_j, x_i, edge_type):    #在这里就是做了对边的mask操作，若有mask（gate），则根据mask后的边的信息执行GNN

        edge_type = (edge_type-1).long()
        edge_type_list = []
        for t in edge_type:
            edge_type_list.append(t.argmax().item())
        edge_type_int = torch.tensor(edge_type_list)
        b = torch.index_select(self.bias, 0, edge_type_int)  #self.bias shape:[6,100] 是代表每个边的type的100维向量(边有6个类别，每种类别是100维)，b就是根据每条边的type，从self.bias中挑选出对应的向量

        basis_messages = torch.matmul(x_j, self.basis).view(-1, self.bias.shape[0], self.out_dim)  #self.basis是指每个节点对应不同的边的type的向量[100,600], basis_messages将源节点的信息融合至边，初始化每个节点连接6种不同的type的边时，边的信息
        count = torch.arange(edge_type.shape[0])
        basis_messages = basis_messages[count, edge_type_int, :] + b   #将对应type的边信息截取出来并与b聚合

        basis_messages = self.process_message(basis_messages)  #正则化，relu激活

        self.latest_messages = basis_messages    #处理后的边信息，有mask就处理，没有mask就返回原来的边信息
        self.latest_source_embeddings = x_j      #源节点嵌入
        self.latest_target_embeddings = x_i      #目标节点嵌入

        return basis_messages

    def get_latest_source_embeddings(self):
        return self.latest_source_embeddings

    def get_latest_target_embeddings(self):
        return self.latest_target_embeddings

    def get_latest_messages(self):
        return self.latest_messages



