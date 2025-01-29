import json
from sklearn.model_selection import train_test_split

prm_interest_folder = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/ml_prm/all_final_prm_interest.json'
with open(prm_interest_folder,'r',encoding='utf-8') as f:
    cur = json.load(f)

data_list = []

for user, blocks in cur.items():
    for block_id, d in blocks.items():
        score = d['score']
        cur_prompt = d['cur_prompt']
        cur_sum = d['cur_sum']
        instruction = cur_prompt
        input = ''
       
        if score > 0.65:
            data_list.append({
            "instruction": instruction,
            "input": input,
            "output": cur_sum
        })

# 使用 train_test_split 函数将列表划分为训练集和测试集
# train_data, test_data = train_test_split(data_list, train_size=0.8, shuffle=True)
print(len(data_list))
save_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/ml_prm/sft_final_prm_data.json'
with open(save_path,'w',encoding='utf-8') as f:
    json.dump(data_list,f,ensure_ascii=False,indent=4)
