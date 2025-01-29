import json
import numpy as np
with open('/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/bge_avg_emb.sum','r',encoding='utf-8') as f:
    embs = json.load(f)


for user, blocks in embs.items():
    emb_list = []

    for block_id, emb in blocks.items():
        emb_list.append(emb)
    emb_list = np.array(emb_list)
    # 归一化向量（使每个向量的 L2 范数为 1）
    norm = np.linalg.norm(emb_list, axis=1, keepdims=True)
    normalized_vectors = emb_list / norm
    
    # 计算余弦相似度矩阵（点积）
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)

    print(similarity_matrix)
    # break


