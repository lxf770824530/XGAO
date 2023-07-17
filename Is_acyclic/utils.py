import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import remove_isolated_nodes
import matplotlib.pyplot as plt
import numpy as np
import random

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



def Initial_graph_generate(args):
    #初始化一个无向完全图
    edge_d1 = []
    edge_d2 = []
    for i in range(args.initNodeNum):
        for j in range(args.initNodeNum):
            if i!=j:
                edge_d1.append(i)
                edge_d2.append(j)

    x = torch.Tensor(args.initNodeNum, 10).uniform_(-1,1)
    edge_index = torch.tensor([edge_d1, edge_d2]).long()

    index_mask1 = torch.randint(low=0, high=2, size=(1, args.initNodeNum * (args.initNodeNum - 1))).bool()
    index_mask2 = torch.randint(low=0, high=2, size=(1, args.initNodeNum * (args.initNodeNum - 1))).bool()
    index_mask = index_mask1 | index_mask2
    index_mask = torch.stack((index_mask[0],index_mask[0]))

    masked_edge_index = edge_index[index_mask].view(2,-1)
    edge_attr = torch.ones(len(edge_d1),dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(data)
    return data


def K_N_for_each_datdaset(args):
    n=0
    k=0
    if args.explain_class == 0:
        if args.dataset == 'Is_Acyclic':
            n = 10
            k = 3
        elif args.dataset == 'BA-2Motif':
            n = 30
            k = 3
        elif args.dataset == 'Twitch_Egos':
            n = 31
            k = 4
        elif args.dataset == 'MUTAG':
            n = 14
            k = 3
    elif args.explain_class == 1:
        if args.dataset == 'Is_Acyclic':
            n = 10
            k = 3
        elif args.dataset == 'BA-2Motif':
            n = 30
            k = 3
        elif args.dataset == 'Twitch_Egos':
            n = 28
            k = 5
        elif args.dataset == 'MUTAG':
            n = 20
            k = 3

    return k,n


def regular_graph_generate(args):


    degree, initial_node_num = K_N_for_each_datdaset(args)
    G = nx.generators.random_graphs.random_regular_graph(degree, initial_node_num)
    while 1:
        # initial_node_num+=1
        if (degree * initial_node_num) % 2 != 0:
            initial_node_num += 1
            continue
        # Construct graph
        G = nx.generators.random_graphs.random_regular_graph(degree, initial_node_num)
        # Compute Laplacian matrix
        L = nx.laplacian_matrix(G)

        # Compute eigenvalues
        eigvals = np.linalg.eigvals(L.A)

        # Check if graph is Ramanujan
        d = G.degree(0)
        lambda2 = eigvals[1]
        if lambda2 <= 2 * np.sqrt(d - 1):
            print("True!")
            break
        else:
            print("False!")
    # x是节点特征矩阵，这里设为单位矩阵。
    x = torch.Tensor(G.number_of_nodes(), 10).uniform_(-1, 1)

    # adj是图G的邻接矩阵的稀疏表示，左边节点对代表一条边，右边是边的值，adj是对称矩阵。
    adj = nx.to_scipy_sparse_matrix(G).tocoo()

    # row是adj中非零元素所在的行索引
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    # col是adj中非零元素所在的列索引。
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)

    # 将行和列进行拼接，shape变为[2, num_edges], 包含两个列表，第一个是row, 第二个是col
    edge_index = torch.stack([row, col], dim=0)

    edge_attr = torch.ones(len(row), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data






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