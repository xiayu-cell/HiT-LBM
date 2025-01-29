import torch
import torch.utils.data as Data
import pickle
from utils import load_json, load_pickle
import pdb
import json

class AmzDataset(Data.Dataset):
    def __init__(self, data_path, set='train', task='ctr', max_hist_len=10, augment=False, aug_prefix=None,prm=False):
        self.task = task
        self.max_hist_len = max_hist_len
        self.augment = augment
        self.prm = prm
        self.set = set
        self.data = load_pickle(data_path + f'/{task}.{set}')
        self.stat = load_json(data_path + '/stat.json')
        self.item_num = self.stat['item_num']
        self.attr_num = self.stat['attribute_num']
        self.attr_ft_num = self.stat['attribute_ft_num']
        self.rating_num = self.stat['rating_num']
        self.dense_dim = self.stat['dense_dim']
        if task == 'rerank':
            self.max_list_len = self.stat['rerank_list_len']
        self.length = len(self.data)
        self.sequential_data = load_json(data_path + '/sequential_data.json')
        self.item2attribution = load_json(data_path + '/item2attributes.json')
        datamaps = load_json(data_path + '/datamaps.json')
        self.id2item = datamaps['id2item']
        self.id2user = datamaps['id2user']
        self.max_block = self.stat['max_block']
        if augment:
            self.hist_aug_data = load_json(data_path + f'/{aug_prefix}_augment.hist')
            self.item_aug_data = load_json(data_path + f'/{aug_prefix}_augment.item')
        # print(type(prm))
        if prm:
            # print('prm : ', prm)
            prefix = aug_prefix.split('_')[-1]
            encoder = aug_prefix.split('_')[0]
            if encoder == 'bert' or encoder == 'longformer':
                self.prm_data = load_json(data_path + f'/all_max_prm_interest.json')

            else:
                self.prm_data = load_json(data_path + f'/all_{prefix}_prm_interest.json')
            # print('item key', list(self.item_aug_data.keys())[:6], len(self.item_aug_data), self.item_num)

    def __len__(self):
        return self.length

    def __getitem__(self, _id):
        if self.task == 'ctr':
            uid, block_id, seq_idx, lb = self.data[_id]
            item_seq, rating_seq = self.sequential_data[str(uid)]
            iid = item_seq[seq_idx]
            hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
            attri_id = self.item2attribution[str(iid)]
            hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
            out_dict = {
                'uid': torch.tensor(int(uid)).long(),
                'iid': torch.tensor(iid).long(),
                'aid': torch.tensor(attri_id).long(),
                'lb': torch.tensor(lb).long(),
                'hist_iid_seq': torch.tensor(hist_item_seq).long(),
                'hist_aid_seq': torch.tensor(hist_attri_seq).long(),
                'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
                'hist_seq_len': torch.tensor(hist_seq_len).long()
            }
            if self.augment:
                item_aug_vec = self.item_aug_data[str(self.id2item[str(iid)])]
                hist_aug_vec_list = []
                if block_id > self.max_block:
                    print(block_id)
                for i in range(block_id+1):
                    v = torch.tensor(self.hist_aug_data[str(self.id2user[uid])][str(i)]).float()

                    hist_aug_vec_list.append(v)
                padding = torch.zeros(self.dense_dim).float()
                cur_block = len(hist_aug_vec_list)
                for i in range(cur_block,self.max_block):
                    hist_aug_vec_list.append(padding)
                # out_dict['item_aug_vec'] = torch.tensor(item_aug_vec).float()
                # pdb.set_trace()
                # print(len(hist_aug_vec_list))
                out_dict['hist_aug_vec_len'] = torch.tensor(cur_block).long()
                out_dict['hist_aug_vec'] = torch.cat(hist_aug_vec_list,dim=0)
                out_dict['item_aug_vec'] = torch.tensor(item_aug_vec).float()
                # print(out_dict['hist_aug_vec'])
            # print('augment:', self.augment)
            # print('prm:', self.prm)
            if self.prm:
                prm_list = []
                if block_id > self.max_block:
                    print(block_id)
                for i in range(block_id+1):
                    v = torch.tensor([self.prm_data[str(self.id2user[uid])][str(i)]['score']]).float()
                    prm_list.append(v)
                cur_block = len(prm_list)
                for i in range(cur_block,self.max_block):
                    prm_list.append(torch.tensor([0]))
                out_dict['prm_vector'] = torch.cat(prm_list,dim=0).unsqueeze(1) # n*1

        elif self.task == 'rerank':
            uid, seq_idx, candidates, candidate_lbs = self.data[_id]
            candidates_attr = [self.item2attribution[str(idx)] for idx in candidates]
            item_seq, rating_seq = self.sequential_data[str(uid)]
            hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
            hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
            out_dict = {
                'iid_list': torch.tensor(candidates).long(),
                'aid_list': torch.tensor(candidates_attr).long(),
                'lb_list': torch.tensor(candidate_lbs).long(),
                'hist_iid_seq': torch.tensor(hist_item_seq).long(),
                'hist_aid_seq': torch.tensor(hist_attri_seq).long(),
                'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
                'hist_seq_len': torch.tensor(hist_seq_len).long()
            }
            if self.augment:
                item_aug_vec = [torch.tensor(self.item_aug_data[str(self.id2item[str(idx)])]).float()
                                for idx in candidates]
                hist_aug_vec = self.hist_aug_data[str(self.id2user[str(uid)])]
                out_dict['item_aug_vec_list'] = item_aug_vec
                out_dict['hist_aug_vec'] = torch.tensor(hist_aug_vec).float()
        else:
            raise NotImplementedError

        return out_dict

if __name__ == '__main__':
    train_set = AmzDataset('/mmu_nlp_ssd/xiayu12/LIBER_ours_train/data/ml-1m/proc_data/block_len_50', 'train', 'ctr', 5, True, 'bge_avg',True)
    train_loader = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    # test_set = AmzDataset('../data/ml-1m/proc_data', 'test', 'ctr', 5, True, 'bert_avg')
    # test_loader = Data.DataLoader(dataset=test_set, batch_size=256, shuffle=True)

    for _, data in enumerate(train_loader):
        print(data)
        for k,v in data.items():
            data[k] = v.cpu()
        data['hist_aug_vec'] = data['hist_aug_vec'].cpu().tolist()
        with open('./test.json','w') as f:
            json.dump(data,f,ensure_ascii=False,indent=4)
            break
        # pdb.set_trace()
    # test_set = AmzDataset(args.data_dir, 'test', 'ctr', args.max_hist_len, args.augment, args.aug_prefix)

