import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import remove_isolated_nodes
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.distributions.categorical import Categorical
from dateset import IsAcyclicDataset

from networkx.algorithms.components import number_connected_components


def Get_tensor_classes_num(y_tensor):
    #获取类别数目

    return len(set(y_tensor.numpy().tolist()))



def load_checkpoint(model, checkpoint_PATH):
    #加载模型

    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    # print('loading checkpoint......')
    # optimizer.load_state_dict(model_CKPT['optimizer'])
    return model



def root_graph_generate(args):
    x = torch.ones((args.initNodeNum, 10), dtype=torch.float)

    # Obtaining probability distribution
    data = IsAcyclicDataset('data', name='Is_Acyclic')
    absence_edge_ratio_sum = 0.0
    for i in data:
        single_absence_edge_ratio = 1 - len(i.edge_index[0]) / (len(i.x) * degree(i.edge_index[0]).max())
        absence_edge_ratio_sum += single_absence_edge_ratio
    absence_edge_ratio = absence_edge_ratio_sum / len(data)
    probs = torch.FloatTensor([absence_edge_ratio, 1 - absence_edge_ratio])
    edge_category_distribution = Categorical(probs)
    edge_categories = []
    for i in range(int(args.initNodeNum * (args.initNodeNum - 1) / 2)):
        indicate_edge = edge_category_distribution.sample().item()
        edge_categories.append(indicate_edge)
        edge_categories.append(indicate_edge)

    edge_index_complete = []
    for i in range(args.initNodeNum):
        for j in range(args.initNodeNum):
            if j > i:
                edge_index_complete.append([i, j])
                edge_index_complete.append([j, i])
    edge_index = torch.tensor(edge_index_complete, dtype=torch.long).T

    edge_categories_tensor = torch.tensor(edge_categories, dtype=torch.float)
    edge_presence_indicator = edge_categories_tensor.bool()

    edge_index = torch.stack((torch.masked_select(edge_index[0], edge_presence_indicator),
                              torch.masked_select(edge_index[1], edge_presence_indicator)))

    edge_attr = torch.masked_select(edge_categories_tensor, edge_presence_indicator)
    root_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if root_graph.has_isolated_nodes():
        _, _, node_mask = remove_isolated_nodes(root_graph.edge_index, num_nodes=len(root_graph.x))
        x = x[node_mask]
        edge_index = Fix_nodes_index(edge_index)
        root_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    print(root_graph)
    return root_graph


def Fix_nodes_index(edge_index):
    b = edge_index.view(-1)
    c = []
    d = {}
    e = []
    for i in b:
        if i not in c:
            c.append(i.item())
    c.sort()

    for v,k in enumerate(c):
        d[k] = v

    for i in edge_index:
        for j in i :
            e.append(d[j.item()])
    t = torch.tensor(e).view(2,-1)
    return t


def Get_dataset_class_num(dataset_name):
    if dataset_name == 'BA_shapes':
        return 2
    elif dataset_name == 'Tree_Cycle':
        return 2



def Draw_graph(Data,j=1):
    edge_attr = []
    edge_max = 0.0
    for i in Data.edge_attr:
        edge_attr.append(i.item())
        if i.item() > edge_max:
            edge_max = i.item()
        G = to_networkx(Data, to_undirected=True, remove_self_loops=True)
    # print(edge_attr)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    i=0
    for (u, v, d) in G.edges(data=True):

        # print(u, v, edge_attr[i])
        # G.add_edge(u, v, weight=edge_attr[i])
        nx.draw_networkx_edges(G, pos, edgelist=[(u,v)])
        i += 1
    # nx.draw(G)
    image_save_path = 'Img/graph'+str(j+1)+'.png'
    plt.savefig(image_save_path)
    plt.close('all')
    # plt.show()


def make_one_hot(data1,args):
    l = Get_dataset_class_num(args.dataset)

    return (np.arange(l)==data1[:,None]).astype(np.integer)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def Random_select_0(gate):
    index_0 = []
    for i in range(len(gate)):
        if gate[i]<1:
            index_0.append(i)
    ran_max = random.randint(0,len(index_0)-1)
    ga=(gate >= 0).float()
    ga[index_0[ran_max]]=0
    return ga


def Fix_mask_single_false(mask, graph_index):
    if len(graph_index) == 2:
        graph_index = torch.t(graph_index)


    min_index = mask.argmin()
    u = graph_index[min_index.item()][0].item()
    v = graph_index[min_index.item()][1].item()
    # inverse_uv = torch.Tensor([v,u]).to(device)
    inverse_uv = torch.Tensor([v, u])

    for i, edge in enumerate(graph_index.float()):
        if torch.equal(edge, inverse_uv):
            mask[i]=0
    return mask

def Get_nodepairs_embeddings(x, edge_index):
    #取节点对中两个节点的嵌入
    count = 0
    for i,j in zip(edge_index[0],edge_index[1]):
        if count ==0:
            x_i = torch.index_select(x, 0, i, out=None)
            x_j = torch.index_select(x, 0, j, out=None)
        else:
            x_i=torch.cat((x_i,torch.index_select(x, 0, i, out=None)),1)
            x_j = torch.cat((x_j, torch.index_select(x, 0, j, out=None)), 1)
        count+=1
    x_i = x_i.view(count,len(x[0]))
    x_j = x_j.view(count, len(x[0]))

    return x_i, x_j


def Rectify(graph, parent_graph):
    G_nx = to_networkx(graph, to_undirected=True, remove_self_loops=True)
    try:
        _, _, node_mask = remove_isolated_nodes(graph.edge_index, num_nodes=len(graph.x))
    except IndexError:
        node_mask = None
        print()

    if nx.is_connected(G_nx):
        Rectified_graph = graph

    elif (node_mask.long().sum().item() < len(graph.x) and number_connected_components(G_nx) == 2 and node_mask.long().argmin() != 0):


    # elif ( or node_mask == None ) and number_connected_components(G_nx) == 2 and node_mask.long().argmin() != 0:
        graph.x = graph.x[node_mask]
        Rectified_graph = graph

    else:
        Rectified_graph = Data(x=parent_graph.x, edge_index=parent_graph.edge_index, edge_attr=parent_graph.edge_attr)

    return Rectified_graph

def Is_exixt_in_list(current_mask, mask_list):
    for m in mask_list:
        if torch.equal(current_mask, m):
            return False
        else:
            continue
    return True