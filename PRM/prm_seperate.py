import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset,random_split
import numpy as np
from tqdm import tqdm
import os
import json
import numpy
import pandas as pd
from sklearn.metrics import roc_auc_score,accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim[1], output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x) 
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

class EmbLabelDataset(Dataset):
    def __init__(self, emb_path, label_path, threshold, mode):
        """
        初始化数据集
        :param emb_path: 嵌入数据的路径
        :param label_path: 标签数据的路径
        :param threshold: 阈值，用于将 acc 转换为二分类标签
        """
        self.emb_data = self.load_data(emb_path)  # 加载嵌入数据
        self.label_data = self.load_data(label_path)  # 加载标签数据
        self.mode = mode
        self.threshold = threshold
        self.embs1,self.embs2, self.labels = self._prepare_data()  # 准备数据

    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    def _prepare_data(self):
        """
        准备嵌入和标签数据
        """
        # 使得正负样本1:1
        embs1,embs2,embs3, labels = [], [],[],[]
        for user, blocks in self.label_data.items():
            for block_id, emb in blocks.items():
                embs1.append(self.emb_data[user][block_id][0])
                embs2.append(self.emb_data[user][block_id][1])
                label = self.label_data[user][block_id]['label']
                # labels.append(1 if acc >= self.threshold else 0)
                labels.append(label)
        return embs1,embs2, labels

        # embs, labels = [], []

        # 首先收集所有样本
        # for user, blocks in self.emb_data.items():
        #     for block_id, emb in blocks.items():
        #         acc = self.label_data[user][block_id]['acc']
        #         label = 1 if acc >= self.threshold else 0
        #         embs.append(emb)
        #         labels.append(label)

        # # 将列表转换为 NumPy 数组以便操作
        # embs = np.array(embs)
        # labels = np.array(labels)

        # # 统计正负样本的索引
        # positive_indices = np.where(labels == 1)[0]
        # negative_indices = np.where(labels == 0)[0]

        # # 确保正负样本数量一致
        # min_samples = min(len(positive_indices), len(negative_indices))
        # positive_indices = np.random.choice(positive_indices, min_samples, replace=False)
        # negative_indices = np.random.choice(negative_indices, min_samples, replace=False)

        # # 合并正负样本的索引
        # balanced_indices = np.concatenate([positive_indices, negative_indices])

        # # 打乱顺序
        # np.random.shuffle(balanced_indices)

        # # 根据筛选后的索引获取平衡后的嵌入和标签
        # balanced_embs = embs[balanced_indices]
        # balanced_labels = labels[balanced_indices]

        # return balanced_embs.tolist(), balanced_labels.tolist()


    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.embs1)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本
        """
        emb1 = torch.tensor(self.embs1[idx], dtype=torch.float32)  # 将嵌入转换为张量
        emb2 = torch.tensor(self.embs2[idx], dtype=torch.float32)  # 将嵌入转换为张量
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # 将标签转换为张量
        return emb1,emb2, label


def train():
    input_dim = 1024*2  # 输入维度，与嵌入维度一致
    hidden_dim = [512,128]  # 隐藏层维度
    output_dim = 2   # 输出维度，二分类问题
    emb_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/bge_avg_seperate_emb.sum'
    label_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/prm_train.json' 
    threshold = 0.5
    data = EmbLabelDataset(emb_path, label_path, threshold,'train')
    # test_data = EmbLabelDataset(emb_path, label_path, threshold,'test')

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    train_dataset, test_dataset = random_split(data, [train_size, test_size])
    # 创建 DataLoader
    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 定义余弦学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)  # T_max 是周期长度，eta_min 是最小学习率
    num_epochs = 10
    softmax = nn.Softmax(dim=-1)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        for i, (batch_embeddings1,batch_embeddings2, batch_labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # 将数据移动到设备
            batch_embeddings1 = batch_embeddings1.to(device)
            batch_embeddings2 = batch_embeddings2.to(device)
            batch_embeddings = torch.cat([batch_embeddings1,batch_embeddings2],dim=-1)
            batch_labels = batch_labels.to(device)

            # 前向传播
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 保存模型权重
        torch.save(model.state_dict(), f'/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/checkpoint/cls_epoch_{epoch}_threshold_{threshold}.pth')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        pp = []
        ll = []
        # 初始化一个空的 DataFrame
        df_list = []

        with torch.no_grad():
            for batch_embeddings1,batch_embeddings2, batch_labels in test_dataloader:
                # 将数据移动到设备
                batch_embeddings1 = batch_embeddings1.to(device)
                batch_embeddings2 = batch_embeddings2.to(device)
                batch_embeddings = torch.cat([batch_embeddings1,batch_embeddings2],dim=-1)

                batch_labels = batch_labels.to(device)

                # 前向传播
                outputs = model(batch_embeddings)
                outputs = softmax(outputs)
                loss = criterion(outputs, batch_labels)

                # 计算测试损失和准确率
                test_loss += loss.item()
                # _, predicted = torch.max(outputs.data, 1)
                # predicted > threshold
                preds = outputs[:,1]
                predicted = (preds > threshold).long()  
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                pp.extend(outputs[:,1].cpu().numpy().tolist())
                ll.extend(batch_labels.cpu().numpy().tolist())
                data_frame = pd.DataFrame({
                    'Predictions': outputs[:,1].cpu().numpy().tolist(),
                    'Labels': batch_labels.cpu().numpy().tolist()
                })
                df_list.append(data_frame)

        # 打印测试结果
        test_loss /= len(test_dataloader)
        accuracy = 100 * correct / total
        auc = roc_auc_score(ll, pp)

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Test AUC: {auc:.2f}')
        # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
        results_df = pd.concat(df_list, ignore_index=True)
        results_df.to_csv(f'/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/test_prm_epoch_{epoch}.csv')

if __name__ == '__main__':
    
    train()