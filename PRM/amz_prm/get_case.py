import json
import pdb

with open('/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/amz_prm/prm_interest_0.json','r',encoding='utf-8') as f:
    data = json.load(f)

for user, blocks in data.items():
    for block_id, d in blocks.items():
        cur_sums = d['cur_sum']
        preds = d['preds']
        cur_prompt = d['cur_prompt']
        mmax = max(preds)
        mmin = min(preds)
        max_cur = cur_sums[preds.index(mmax)]
        min_cur = cur_sums[preds.index(mmin)]
        # pdb.set_trace()
        p = '我这里有两个根据用户上一段时间的兴趣和当前的浏览记录生成的当前兴趣。你需要判断哪个好。'
        p1 = '【之前的兴趣和当前记录】'
        p2 = '【当前兴趣1】'
        p3 = '【当前兴趣2】'
        o = f'{p}{p1}{cur_prompt}{p2}{min_cur}{p3}{max_cur}'
        print(o)
