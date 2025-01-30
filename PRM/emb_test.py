import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# def cosine_similarity(A, B):
#     dot_product = np.dot(A, B)
#     norm_A = np.linalg.norm(A)
#     norm_B = np.linalg.norm(B)
#     return dot_product / (norm_A * norm_B)

with open('/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/amz/bge_avg_cur_emb.sum','r',encoding='utf-8') as f:
    embs = json.load(f)

arrays = []
n = 20
idx = 0
for user, blocks in embs.items():
    for block_id, emb in blocks.items():
        arrays.append(np.array(emb))
        idx += 1
        if idx > n:
            break
    if idx > n:
            break

# 计算两两之间的余弦相似度
n = len(arrays)
similarity_matrix = np.zeros((n, n))  # 初始化相似度矩阵

for i in range(n):
    for j in range(n):
        # 计算数组i和数组j的余弦相似度
        similarity_matrix[i, j] = cosine_similarity([arrays[i]], [arrays[j]])[0][0]

print("相似度矩阵：")
print(similarity_matrix)



