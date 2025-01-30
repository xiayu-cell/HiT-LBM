import torch
import torch.nn as nn
from prm import MLPClassifier

# a = torch.tensor([1])
# b = torch.tensor([2])
# c = torch.cat([a,b],dim=0).unsqueeze(1)
# print(c)
# print(c.shape)
prm_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/checkpoint/epoch_2.pth'
input_dim = 1024  # 输入维度，与嵌入维度一致
hidden_dim = 128  # 隐藏层维度
output_dim = 2 
device = torch.device("cuda:0")
prm = MLPClassifier(input_dim,hidden_dim,output_dim).to(device)
prm.load_state_dict(torch.load(prm_path))

out = prm(torch.randn(10,1024).to(device))
softmax = nn.Softmax(dim=-1)
print(softmax(out)[:,1])