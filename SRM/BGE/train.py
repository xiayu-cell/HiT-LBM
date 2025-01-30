from torch.optim import Adam
from tqdm import tqdm
import torch
import json
import numpy as np
from transformers import BertTokenizer
from BertClassifier import BertClassifier
from torch.utils.data import DataLoader, TensorDataset, Dataset,random_split
import os
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
from transformers import AutoTokenizer, AutoModel

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
# torch.cuda.set_device(4)
def load_data(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self._data = load_data(data_path)  # 加载标签数据
        self.texts, self.labels = self._prepare_data()  # 准备数据
        self.tokenizer = AutoTokenizer.from_pretrained('/mmu_nlp_hdd/xiayu12/LIBER/llm/longformer-base-4096',  trust_remote_code=True)
    
    
    def __len__(self):
        return len(self.labels)

    def _prepare_data(self):
        """
        准备嵌入和标签数据
        """
        # 使得正负样本1:1
        texts,labels = [], []
        for user, blocks in self._data.items():
            for block_id, emb in blocks.items():
                cur_prompt = emb['cur_prompt']
                cur_sum = emb['cur_sum']
                label = emb['label']
                # labels.append(1 if acc >= self.threshold else 0)
                texts.append('</s>'.join([cur_prompt,cur_sum]))
                labels.append(label)
        return texts, labels

    def __getitem__(self, idx):
        text = self.texts[idx]  # 将嵌入转换为张量
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # 将标签转换为张量
        x = self.tokenizer(text, padding='max_length', truncation=True, max_length=4096, return_tensors="pt",
                              return_attention_mask=True)
        # print(x.keys())
        
        # print(input_id.shape)
        return x, label

def train():
    data_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/prm_train.json' 
    threshold = 0.5
    data = Dataset(data_path)

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    train_dataset, test_dataset = random_split(data, [train_size, test_size])
    # 创建 DataLoader
    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertClassifier(0.2)
    model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    # 定义余弦学习率调度器
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)  # T_max 是周期长度，eta_min 是最小学习率
    num_epochs = 10
    softmax = nn.Softmax(dim=-1)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        for i, (x, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # 将数据移动到设备
            input_id = x['input_ids'].squeeze(1)
            mask = x['attention_mask'].squeeze(1)
            # print(input_id.shape)
            label = label.to(device)
            input_id = input_id.to(device)
            mask = mask.to(device)
            # 前向传播
            outputs = model(input_id,mask)
            loss = criterion(outputs, label)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            if i%10 == 0:
                print(f'{i}/{len(train_dataloader)}, {loss}')

        # 保存模型权重
        torch.save(model.state_dict(), f'/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/BGE/checkpoint/epoch_{epoch}_threshold_{threshold}.pth')
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
            for x, label in test_dataloader:
                input_id = x['input_ids'].squeeze(1)
                mask = x['attention_mask'].squeeze(1)
                label = label.to(device)
                input_id = input_id.to(device)
                mask = mask.to(device)
                # 前向传播
                outputs = model(input_id,mask)
                outputs = softmax(outputs)
                loss = criterion(outputs, label)

                # 计算测试损失和准确率
                test_loss += loss.item()
                # _, predicted = torch.max(outputs.data, 1)
                # predicted > threshold
                preds = outputs[:,1]
                predicted = (preds > threshold).long()  
                total += label.size(0)
                correct += (predicted == label).sum().item()
                pp.extend(outputs[:,1].cpu().numpy().tolist())
                ll.extend(label.cpu().numpy().tolist())
                data_frame = pd.DataFrame({
                    'Predictions': outputs[:,1].cpu().numpy().tolist(),
                    'Labels': label.cpu().numpy().tolist()
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