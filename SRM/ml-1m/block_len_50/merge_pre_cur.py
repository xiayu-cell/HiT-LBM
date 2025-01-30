import json

pre_prm_data_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/pre_final_prm_data.json'
cur_prm_data_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/final_prm_data.json'
with open(cur_prm_data_path,'r',encoding='utf-8') as f:
    cur = json.load(f)

with open(pre_prm_data_path,'r',encoding='utf-8') as f:
    pre = json.load(f)

pos_len = 0
neg_len = 0
for user, blocks in cur.items():
    for block_id, data in blocks.items():
        cur_auc = data['auc']
        pre_auc = pre[user][block_id]['auc']
        if cur_auc > pre_auc:
            label = 1
            pos_len+=1
        else:
            label = 0
            neg_len+=1
        cur[user][block_id]['label'] = label

print(pos_len)
print(neg_len)
save_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/prm_train.json'
with open(save_path,'w',encoding='utf-8') as f:
    json.dump(cur,f,ensure_ascii=False,indent=4)


