import json
from sklearn.model_selection import train_test_split
pre_prm_data_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/pre_final_prm_data.json'
cur_prm_data_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/final_prm_data.json'
with open(cur_prm_data_path,'r',encoding='utf-8') as f:
    cur = json.load(f)

with open(pre_prm_data_path,'r',encoding='utf-8') as f:
    pre = json.load(f)

data_list = []

for user, blocks in cur.items():
    for block_id, d in blocks.items():
        instruction = 'You are a user interest summarization discriminator, and your task is to determine whether the new interest generated based on the user\'s previous interests and the currently viewed item is reasonable. Below are the user\'s previous interests and the newly generated interest.'
        input = '[USER PREVIOUS INTEREST]\n{}, [USER NER INTEREST]\n{} Is the newly generated user interest reasonable? Output only Yes or No.'.format(d['cur_prompt'],d['cur_sum'])
        cur_auc = d['auc']
        pre_auc = pre[user][block_id]['auc']
        if cur_auc > pre_auc:
            output = 'Yes'
        else:
            output = 'No'

        data_list.append({
            "instruction": instruction,
            "input": input,
            "output": output
        })

# 使用 train_test_split 函数将列表划分为训练集和测试集
train_data, test_data = train_test_split(data_list, train_size=0.8, shuffle=True)

save_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/sft_final_prm_data.json'
with open(save_path,'w',encoding='utf-8') as f:
    json.dump(train_data,f,ensure_ascii=False,indent=4)

with open('/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/sft_test_final_prm_data.json','w',encoding='utf-8') as f:
    json.dump(test_data,f,ensure_ascii=False,indent=4)