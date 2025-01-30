import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from tqdm import tqdm
import json
from transformers import AutoTokenizer
from BertClassifier import BertClassifier
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModel
import pandas as pd

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def load_data(path):
    """加载数据"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self._data = load_data(data_path)
        self.texts, self.labels = self._prepare_data()
        self.tokenizer = AutoTokenizer.from_pretrained('/mmu_nlp_hdd/xiayu12/LIBER/llm/longformer-base-4096', trust_remote_code=True)

    def __len__(self):
        return len(self.labels)

    def _prepare_data(self):
        """准备数据"""
        texts, labels = [], []
        for user, blocks in self._data.items():
            for block_id, emb in blocks.items():
                cur_prompt = emb['cur_prompt']
                cur_sum = emb['cur_sum']
                label = emb['label']
                texts.append('</s>'.join([cur_prompt, cur_sum]))
                labels.append(label)
        return texts, labels

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        x = self.tokenizer(text, padding='max_length', truncation=True, max_length=4096, return_tensors="pt", return_attention_mask=True)
        return x, label

def trian(model, train_loader, optimizer, scheduler, criterion, local_rank):
    # 训练循环
    num_epochs = 10
    softmax = nn.Softmax(dim=-1)

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # 设置 epoch 以打乱数据
        for i, (x, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_id = x['input_ids'].squeeze(1).to(device)
            mask = x['attention_mask'].squeeze(1).to(device)
            label = label.to(device)

            # 前向传播
            outputs = model(input_id, mask)
            loss = criterion(outputs, label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 保存模型权重
        if rank == 0:  # 只在主进程保存模型
            torch.save(model.state_dict(), f'/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/BGE/checkpoint/epoch_{epoch}_threshold_{threshold}.pth')

        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        pp = []
        ll = []
        df_list = []

        with torch.no_grad():
            for x, label in test_dataloader:
                input_id = x['input_ids'].squeeze(1).to(device)
                mask = x['attention_mask'].squeeze(1).to(device)
                label = label.to(device)

                outputs = model(input_id, mask)
                outputs = softmax(outputs)
                loss = criterion(outputs, label)

                test_loss += loss.item()
                preds = outputs[:, 1]
                predicted = (preds > threshold).long()
                total += label.size(0)
                correct += (predicted == label).sum().item()
                pp.extend(outputs[:, 1].cpu().numpy().tolist())
                ll.extend(label.cpu().numpy().tolist())

                data_frame = pd.DataFrame({
                    'Predictions': outputs[:, 1].cpu().numpy().tolist(),
                    'Labels': label.cpu().numpy().tolist()
                })
                df_list.append(data_frame)

        # 打印测试结果
        test_loss /= len(test_dataloader)
        accuracy = 100 * correct / total
        auc = roc_auc_score(ll, pp)

        if rank == 0:  # 只在主进程打印结果
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Test AUC: {auc:.2f}')
            results_df = pd.concat(df_list, ignore_index=True)
            results_df.to_csv(f'/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/test_prm_epoch_{epoch}.csv')

    cleanup()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
    # 为每个进程配置GPU
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = BertClassifier(0.2).to(device)
    model = DDP(model, device_ids=[rank],find_unused_parameters=True, device_ids=[local_rank],
                                    output_device=local_rank)
    # 加载数据
    data_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/prm_train.json'
    data = Dataset(data_path)
    # 划分训练集和测试集
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = random_split(data, [train_size, test_size])

    # 使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),
                      lr=1e-5,  # args.learning_rate - default is 5e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8
                      )
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=100,
                                                num_training_steps=total_steps)

    trian(model, train_loader, optimizer, scheduler, local_rank)