import torch
from Utils.SingleGCNLayer import SingleGCNLayer
from torch.nn import ReLU, Linear, LayerNorm
from Utils.hard_concrete import HardConcrete
from Utils.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from Utils.squeezer import Squeezer
from utils import Get_nodepairs_embeddings


class Mask_learner(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.singleGCN_output_dim = 64
        self.hidden_dim = 64
        self.input_dim = 7

        self.singGCN = SingleGCNLayer(self.input_dim, self.singleGCN_output_dim)

        vertex_embedding_dims = self.singleGCN_output_dim
        message_dims = self.singleGCN_output_dim
        gate_input_shape = [vertex_embedding_dims, message_dims, vertex_embedding_dims]  # [100, 100, 100]

        self.mask_learing_nn = torch.nn.Sequential(
            MultipleInputsLayernormLinear(gate_input_shape, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, 1),
            Squeezer(),
            HardConcrete()
        )

    def forward(self, Graph):
        x = self.singGCN(Graph.x, Graph.edge_index, Graph.edge_attr)
        latest_source_embeddings, latest_target_embeddings = Get_nodepairs_embeddings(x, Graph.edge_index)
        latest_messages = self.singGCN.get_latest_messages()

        mask_input = [latest_source_embeddings, latest_messages, latest_target_embeddings]  # 初始化mask learner的输入，每条边的源节点、边信息。目标节点
        mask, penalty = self.mask_learing_nn(mask_input)

        return mask, penalty




