import json
import pdb

with open('/mmu_nlp_hdd/xiayu12/Open-World-Knowledge-Augmented-Recommendation/data/amz/raw_data/meta_Books.json','r') as f:
    data = json.load(f)

print(data.keys())
pdb.set_trace()
