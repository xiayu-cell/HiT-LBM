import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from dataset import AmzDataset
class Dice(nn.Module):
    def __init__(self,emb_size,eps=1e-8,dim=3):
        super(Dice,self).__init__()
        self.name = 'dice'
        self.dim = dim
        self.bn = nn.BatchNorm1d(emb_size,eps=eps)
        self.sig = nn.Sigmoid()
        if dim == 2:   #[B,C]
            self.alpha = torch.zeros((emb_size,))
            self.beta = torch.zeros((emb_size,))
        elif dim == 3: #[B,C,E]
            self.alpha = torch.zeros((emb_size,1))
            self.beta = torch.zeros((emb_size,1))
    
    def forward(self,x):
        if self.dim == 2:
            self.beta  = self.beta.to(x.device)
            self.alpha = self.alpha.to(x.device)
            x_n = self.sig(self.beta * self.bn(x))
            return self.alpha * (1-x_n) * x + x_n * x
        elif self.dim == 3:
            self.beta  = self.beta.to(x.device)
            self.alpha = self.alpha.to(x.device)
            x = torch.transpose(x,1,2)
            x_n = self.sig(self.beta * self.bn(x))
            output = self.alpha * (1-x_n) * x + x_n * x
            output = torch.transpose(output,1,2)
            return output

class ActivationUnit(nn.Module):
    def __init__(self, in_size, af='dice', hidden_size=36):
        super(ActivationUnit, self).__init__()
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        if af == 'dice':
            self.af = Dice(hidden_size, dim=3)
        elif af == 'prelu':
            self.af = nn.PReLU()
        else:
            raise ValueError('Only "dice" and "prelu" can be chosen for activation function')

    def forward(self, item1, item2):
        # item1: [batch_size, hidden_size*2]
        # item2: [batch_size, hidden_size*2]
        # item2 = item2.unsqueeze(1) # bs,1,dim
        cross = torch.matmul(item1, item2.transpose(-1,-2)).squeeze()  # [batch_size, len, 1]
        item2 = item2.expand(-1, item1.size(1), -1)  # 形状: (bs, len, dim)
        # print(item1.shape)
        # print(item2.shape)
        # print(cross.shape)

        x = torch.cat([item1, cross.unsqueeze(-1), item2], dim=-1)  # [batch_size, hidden_size*4 + 1]
        x = self.linear1(x)
        x = self.af(x).to(x.device)
        x = self.linear2(x)
        return x

class SIM(nn.Module):
    def __init__(self, AmzDataset, hidden_size=32, mode='soft'):
        super(SIM, self).__init__()
        self.item_num = AmzDataset.item_num
        self.cate_num = AmzDataset.attr_num
        self.rating_num = AmzDataset.rating_num
        self.attr_ft_num = AmzDataset.attr_ft_num

        self.hidden_size = hidden_size
        self.mode = mode

        print(self.cate_num)
        
        self.item_embedding = nn.Embedding(30000, hidden_size)
        self.cate_embedding = nn.Embedding(7519, hidden_size)
        self.rating_embedding = nn.Embedding(10, hidden_size)


        self.linear = nn.Sequential(
            nn.Linear(hidden_size * (1+self.attr_ft_num*2+2), 80),
            Dice(80, dim=2),
            nn.Linear(80, 40),
            Dice(40, dim=2),
            nn.Linear(40, 1)
        )

        self.au = ActivationUnit(hidden_size * (2+self.attr_ft_num*2)+1, af='dice', hidden_size=36)

    # def process_input(self, inp):
    #     device = next(self.parameters()).device
    #     hist_item_emb = self.item_embedding(inp['hist_iid_seq'].to(device)).view(-1, self.max_hist_len, self.embed_dim)
    #     hist_attr_emb = self.attr_embedding(inp['hist_aid_seq'].to(device)).view(-1, self.max_hist_len,
    #                                                                              self.embed_dim * self.attr_fnum)
    #     hist_rating_emb = self.rating_embedding(inp['hist_rate_seq'].to(device)).view(-1, self.max_hist_len,
    #                                                                                   self.embed_dim)
    #     hist_emb = torch.cat([hist_item_emb, hist_attr_emb, hist_rating_emb], dim=-1)
    #     hist_len = inp['hist_seq_len'].to(device)

    #     iid_emb = self.item_embedding(inp['iid'].to(device))
    #     attr_emb = self.attr_embedding(inp['aid'].to(device)).view(-1, self.embed_dim * self.attr_fnum)
    #         # item_emb = item_emb.view(-1, self.itm_emb_dim)
    #     labels = inp['lb'].to(device)
           
    #     return hist,item, attr, hist_len, labels


    def get_ctr_output(self, logits, labels=None):
        outputs = {
            'logits': torch.sigmoid(logits),
            'labels': labels,
        }


        if labels is not None:
         
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            outputs['loss'] = loss

        return outputs

    def forward(self, inp):
        # hist: [batch_size, max_seq_length, 3] where 3 is [item, cate, rating]
        # item: [batch_size, 1]
        # cate: [batch_size, 1]
        device = next(self.parameters()).device
        item = inp['iid'].unsqueeze(1).to(device)
        cate = inp['aid'].to(device)
        lb = inp['lb'].to(device)


        # Embedding lookup
        hist_items = inp['hist_iid_seq'].to(device)  # [batch_size, max_seq_length]
        hist_cates = inp['hist_aid_seq'].to(device) # [batch_size, max_seq_length]
        hist_ratings = inp['hist_rate_seq'].to(device) # [batch_size, max_seq_length]

        # Embeddings
        item_emb = self.item_embedding(item)  # [batch_size, hidden_size]
        cate_emb = self.cate_embedding(cate).view(-1, 1,32 * self.attr_ft_num)  # [batch_size, hidden_size]
        hist_item_emb = self.item_embedding(hist_items)  # [batch_size, max_seq_length, hidden_size]
        hist_cate_emb = self.cate_embedding(hist_cates).view(-1, 50, 32 * self.attr_ft_num)  # [batch_size, max_seq_length, hidden_size]
        hist_rating_emb = self.rating_embedding(hist_ratings)  # [batch_size, max_seq_length, hidden_size]

        # Concatenate item and cate embeddings
        # print(hist_item_emb.shape)
        # print(hist_cate_emb.shape)
        hist_cate_emb  =hist_cate_emb.squeeze(2)
        # exit(0)
        item_emb_cat = torch.cat([item_emb, cate_emb], dim=-1)  # [batch_size, hidden_size*2]
        hist_emb_cat = torch.cat([hist_item_emb, hist_cate_emb], dim=-1)  # [batch_size, max_seq_length, hidden_size*2]

        # Compute cosine similarity
        item_emb_cat_exp = item_emb_cat # [batch_size, 1, hidden_size*2]
        sim = torch.nn.functional.cosine_similarity(item_emb_cat_exp, hist_emb_cat, dim=-1)  # [batch_size, max_seq_length]

        if self.mode == 'hard':
            mask = torch.eq(hist_cates, cate.unsqueeze(1))  # [batch_size, max_seq_length]
            weights = mask.float() * sim
        elif self.mode == 'soft':
            weights = sim
            # weights = torch.where(sim >= self.thre, weights, torch.zeros_like(weights))
            topk_values, topk_indices = torch.topk(weights, k=5, dim=1)  # 形状: (batch_size, 5)
            # print(topk_indices)
            index = topk_indices.unsqueeze(-1)
            topk_embeddings = torch.gather(hist_emb_cat, dim=1, index=index.expand(-1, -1, hist_emb_cat.size(-1))) # bs,len,dim
            topk_rating_embs = torch.gather(hist_rating_emb, dim=1, index=index.expand(-1, -1, hist_rating_emb.size(-1))) # bs,len,dim
            topk_values = topk_values.unsqueeze(2)
            topk_values = topk_values / (torch.sum(topk_values, dim=1, keepdim=True) + 1e-8)
            hist_emb_weighted = (topk_values * topk_embeddings)  # [batch_size, len, hidden_size*2]

            # print(hist_emb_weighted.shape)
            au_output = self.au(hist_emb_weighted, item_emb_cat) # bs,5
            # print(au_output.shape)
            topk_embeddings = torch.cat([hist_emb_weighted,topk_rating_embs],dim=-1)
            final_hist = (topk_embeddings*au_output).sum(dim=1)
            # print(final_hist.shape)
            # print(item_emb)
        else:
            raise ValueError('Mode should be "hard" or "soft"')

        # Normalize weights
        # weights = weights.unsqueeze(2)  # [batch_size, max_seq_length, 1]
        # weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)

        # # Weighted sum of history embeddings
        # hist_emb_weighted = (weights * hist_emb_cat).sum(dim=1)  # [batch_size, hidden_size*2]

        # # Activation Unit
        # au_output = self.au(hist_emb_weighted, item_emb_cat).squeeze()  # [batch_size]

        # Concatenate item, cate, and weighted history
        res = torch.cat([item_emb.squeeze(1), cate_emb.squeeze(1), final_hist], dim=1)  # [batch_size, hidden_size*6]
        logits = self.linear(res)  # [batch_size, 1]
        out = self.get_ctr_output(logits, lb)
        return out

# # 测试用例
# batch_size = 10
# max_seq_length = 50
# item_num = 100
# cate_num = 20
# rating_num = 5
# hidden_size = 32

# # 随机生成输入数据
# hist = torch.randint(0, rating_num, (batch_size, max_seq_length, 3))  # [batch_size, max_seq_length, 3]
# item = torch.randint(0, item_num, (batch_size, 1))  # [batch_size, 1]
# cate = torch.randint(0, cate_num, (batch_size, 1))  # [batch_size, 1]

# # 初始化模型
# model = SIM(item_num, cate_num, rating_num, hidden_size, mode='soft')

# # 前向传播
# output = model(hist, item, cate)
# print("Output shape:", output.shape)  # 应该输出: torch.Size([4, 2])