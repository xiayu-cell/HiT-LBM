import json
import os
item_knowledge_folder = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/preference_generation/ml-1m/block_len_50/item_knowledge/14B'
all = {}
for filename in os.listdir(item_knowledge_folder):
    if filename.startswith(f'item_knowledge') and filename.endswith(".json"):
        file_path = os.path.join(item_knowledge_folder, filename)
        # 读取 JSON 文件
        with open(file_path, 'r') as file:
        # pdb.set_trace()
            t = json.load(file)
            all.update(t)

print(len(all))
path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/preference_generation/ml-1m/block_len_50/item_knowledge/14B/all_item_knowledge.json'
with open(path,'w',encoding='utf-8') as f:
    json.dump(all,f,ensure_ascii=False,indent=4)