import json
import os
block_interest_folder = '/mmu_nlp_hdd/xiayu12/LIBER/preference_generation/ml-1m'
all = {}
for filename in os.listdir(block_interest_folder):
    if filename.startswith(f'interest') and filename.endswith(".json"):
        file_path = os.path.join(block_interest_folder, filename)
        # 读取 JSON 文件
        with open(file_path, 'r') as file:
        # pdb.set_trace()
            t = json.load(file)
            for user_id, cot in t.items():
                if user_id in all:
                    for k,v in cot.items():
                        if k not in all[user_id]:
                            all[user_id][k] = v
                else:
                    all[user_id] = cot

for k,v in all.items():
    sorted_dict = dict(sorted(v.items(), key=lambda item: int(item[0])))
    all[k] = sorted_dict
sorted_dict = dict(sorted(all.items(), key=lambda item: int(item[0])))

print(len(sorted_dict))
path = '/mmu_nlp_hdd/xiayu12/LIBER/preference_generation/ml-1m/all_block_interest.json'
with open(path,'w',encoding='utf-8') as f:
    json.dump(sorted_dict,f,ensure_ascii=False,indent=4)
