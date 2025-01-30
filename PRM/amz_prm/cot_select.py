import json
import os

prm_interest_folder = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/amz_prm'
all = {}
for filename in os.listdir(prm_interest_folder):
    if filename.startswith(f'prm_interest') and filename.endswith(".json"):
        file_path = os.path.join(prm_interest_folder, filename)
        # 读取 JSON 文件
        with open(file_path, 'r') as file:
        # pdb.set_trace()
            t = json.load(file)
            for user_id, cot in t.items():
                if user_id in all:
                    for k,v in cot.items():
                        if k not in all[user_id]:
                            preds = v['preds']
                            max_index = preds.index(max(preds))
                            all[user_id][k] = {
                                'score': max(preds),
                                'pre_sum': v['pre_sum'],
                                'cur_sum': v['cur_sum'][max_index],
                                'cur_prompt': v['cur_prompt']
                            }
                else:
                    all[user_id] = {}
                    for k,v in cot.items():
                        if k not in all[user_id]:
                            preds = v['preds']
                            max_index = preds.index(max(preds))
                            all[user_id][k] = {
                                'score': max(preds),
                                'pre_sum': v['pre_sum'],
                                'cur_sum': v['cur_sum'][max_index],
                                'cur_prompt': v['cur_prompt']
                            }

for k,v in all.items():
    sorted_dict = dict(sorted(v.items(), key=lambda item: int(item[0])))
    all[k] = sorted_dict
# sorted_dict = dict(sorted(all.items(), key=lambda item: int(item[0])))

print(len(all))
path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/amz_prm/all_max_prm_interest.json'
with open(path,'w',encoding='utf-8') as f:
    json.dump(all,f,ensure_ascii=False,indent=4)
