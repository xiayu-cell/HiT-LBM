from sklearn.metrics import roc_auc_score,accuracy_score
import json
import os
merged_data = []
folder_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m_prm/llm_prm'
for filename in os.listdir(folder_path):
    if filename.startswith('untrain_test') and filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            merged_data.extend(data)

preds = []
labels = []
scores = []
for item in merged_data:
    scores.append(item['prob'])
    label = item['label']
    if label == 'Yes':
        labels.append(1)
    else:
        labels.append(0)
    if item['prob'] >= 0.5:
        preds.append(1)
    else:
        preds.append(0)
auc = roc_auc_score(labels, scores)
acc = accuracy_score(labels,preds)

print(len(merged_data))
print(auc)
print(acc)