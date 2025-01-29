import json

with open('/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/ml-1m/prm_data.json','r',encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    cur_prompt = item['cur_prompt']
    user_hist = cur_prompt.split('[Current Movie Viewing History]')[-1].split('\n')[1]
    print(user_hist)