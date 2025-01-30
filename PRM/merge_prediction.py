import json
import os
from sklearn.metrics import roc_auc_score,accuracy_score

with open('/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/ml-1m/prm_data.json','r',encoding='utf-8') as f:
    prm_data = json.load(f)
prediction_folder = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/ml-1m'
all_prediction = {}
total = 0
pos = 0
for filename in os.listdir(prediction_folder):
    if filename.startswith(f'prediction') and filename.endswith(".json"):
        file_path = os.path.join(prediction_folder, filename)
        # 读取 JSON 文件
        with open(file_path, 'r') as file:
        # pdb.set_trace()
            t = json.load(file)
            for user_id, preds in t.items():
                if user_id in all_prediction:
                    for k,v in preds.items():
                        total +=1
                        y_true = []
                        y_pred = []
                        y_scores = []
                        for p in v:
                            y_true.append(p[1])
                            y_scores.append(p[0])
                            y_pred.append(p[0]>=0.5)
                        acc = accuracy_score(y_true, y_pred)
                        if acc >= 0.7:
                            pos+=1
                        # auc = roc_auc_score(y_true, y_scores)
                        if k not in all_prediction[user_id]:
                            all_prediction[user_id][k] = {
                                # 'auc':auc,
                                'acc':acc
                            }
                else:
                    all_prediction[user_id] = {}
                    for k,v in preds.items():
                        total +=1
                        y_true = []
                        y_pred = []
                        y_scores = []
                        for p in v:
                            y_true.append(p[1])
                            y_scores.append(p[0])
                            y_pred.append(p[0]>=0.5)
                        acc = accuracy_score(y_true, y_pred)
                        if acc >= 0.7:
                            pos+=1
                        # auc = roc_auc_score(y_true, y_scores)
                        if k not in all_prediction[user_id]:
                            all_prediction[user_id][k] = {
                                # 'auc':auc,
                                'acc':acc
                            }

for item in prm_data:
    all_prediction[item['user']][item['block']]['targets'] = item['targets']
    all_prediction[item['user']][item['block']]['cur_sum'] = item['cur_sum']
    all_prediction[item['user']][item['block']]['pre_sum'] = item['pre_sum']
    all_prediction[item['user']][item['block']]['cur_prompt'] = item['cur_prompt']

print(len(all_prediction))
path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/ml-1m/prompt_prm_data.json'
with open(path,'w',encoding='utf-8') as f:
    json.dump(all_prediction,f,ensure_ascii=False,indent=4)

print(total)
print(pos)