import torch
import torch.nn.functional as F
from Utils.pytorchtools import EarlyStopping
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from Utils.RGCNLayer import RGCN


import os.path as osp
from utils import Get_tensor_classes_num, load_checkpoint, Get_dataset_class_num
from torch.nn import ReLU, Linear, LayerNorm
from torch_geometric.datasets import TUDataset

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = 32
        self.output_dim = output_dim

        self.injected_message_scale = None
        self.injected_message_replacement = None

        self.transform = torch.nn.Sequential(
            Linear(self.input_dim, self.hid_dim),
            LayerNorm(self.hid_dim),
            torch.nn.ReLU(),
            Linear(self.hid_dim, self.hid_dim),
            LayerNorm(self.hid_dim),
            torch.nn.ReLU(),
        )

        self.rgcn = RGCN(self.input_dim, 32, n_relations=4, n_layers=1, inverse_edges=False)
        self.conv1 = GCNConv(32, 64)
        # self.conv2 = GCNConv(48, 64)
        self.conv3 = GCNConv(64, 64)
        self.linear1 = Linear(64, 32)
        self.linear2 = Linear(32, self.output_dim)


    def forward(self, data_x, data_edge_index, data_edge_attr, batch):
        # x, edge_index = data.x, data.edge_index
        # x = self.transform(data_x)
        x = self.rgcn(data_x, data_edge_index, data_edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv1(x, data_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        # x = self.conv2(x, data_edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, data_edge_index)
        x = global_mean_pool(x, batch)
        x = self.linear1(x)
        x = self.linear2(x)

        x = F.dropout(x, p=0.3, training=self.training)
        x = F.softmax(x)

        return x

    def inject_message_scale(self, message_scale):
        self.rgcn.inject_message_scale(message_scale)

    def inject_message_replacement(self, message_replacement):
        self.rgcn.inject_message_scale(message_replacement)  # Have to store it in a list to prevent the pytorch module from thinking it is a parameter

    def get_vertex_embedding_dims(self):
        return self.hid_dim

    def get_message_dims(self):
        return self.hid_dim

    def get_latest_source_embeddings(self):
        return self.rgcn.get_latest_source_embeddings()
        # return [layer.get_latest_source_embeddings() for layer in self.gnn_layers]

    def get_latest_target_embeddings(self):
        return self.rgcn.get_latest_target_embeddings()
        # return [layer.get_latest_target_embeddings() for layer in self.gnn_layers]

    def get_latest_messages(self):
        return self.rgcn.get_latest_messages()
        # return [layer.get_latest_messages() for layer in self.gnn_layers]

    def count_latest_messages(self):
        return sum([layer_messages.numel() / layer_messages.shape[-1] for layer_messages in self.get_latest_messages()])




def Test_model(model, loader, criterion):
    model.eval()

    correct = 0
    for data in loader:  # 批遍历测试集数据集。
        out = model(data.x, data.edge_index,  data.edge_attr, data.batch)  # 一次前向传播
        loss = criterion(out, data.y)
        pred = out.argmax(dim=1)  # 使用概率最高的类别
        correct += int((pred == data.y).sum())  # 检查真实标签
    return correct / len(loader.dataset), loss


# if __name__ == '__main__':
def Train_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_save_dir = osp.join('checkpoint', 'MUTAG')
    model_name = 'MUTAG' + '_gcn_model.pth'
    model_save_path = osp.join(model_save_dir, model_name)
    data = TUDataset('data/TUDataset', name = 'MUTAG')

    torch.manual_seed(42)
    dataset = data.shuffle()

    train_dataset = dataset[:188]
    test_dataset = dataset[150:]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = GCN(7, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(2000, verbose=True, path=model_save_path)

    best_train_acc = 0
    best_test_acc = 0
    best_loss = 100
    for epoch in range(5000):
        model.train()

        for data in train_loader:
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)

            loss.backward()
            optimizer.step()
        train_acc, train_loss = Test_model(model, train_loader, criterion)
        test_acc, val_loss = Test_model(model, test_loader, criterion)
        if train_acc>best_train_acc:
            best_train_acc = train_acc
            best_test_acc = test_acc
        if epoch%10 == 9:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        early_stopping(train_loss, model)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")

            # 结束模型训练
            break
    print(f'Result acc: (Train Acc: {best_train_acc:.4f}, Test Acc: {best_test_acc:.4f})')
