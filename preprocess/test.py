from pre_utils import load_json, load_pickle, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING

data = load_pickle('/mmu_nlp_ssd/xiayu12/LIBER_ours_train/data/ml-1m/proc_data/ctr.train')
print(len(data))
max_n = 0
for i in data:
    if i[1] > max_n :
        max_n = i[1]
print(max_n)