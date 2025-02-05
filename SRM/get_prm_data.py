import json
import random

if __name__ == '__main__':
    data_path = '/preference_generation/amz/block_len_50/summary/all_interest.json'
    block2item_path = '/data/amz/proc_data/block_len_50/block2item.json'
    with open(block2item_path,'r',encoding='utf-8') as f:
        block2item = json.load(f)

    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)

    sft_ques = 'Based on the user\'s current book preferences and interest changes, please determine whether the user will like the target book. Output only "Yes" or "No."'
    data_list = []
    for user, blocks in data.items():
        for block_id, d in blocks.items(): 
            target_items = block2item[user][str(block_id)]
            # 随机sample target item
            # 将数据分为 rating > 4 和 rating <= 4 两组
            high_rating = [item for item in target_items if item[2] > 4]
            low_rating = [item for item in target_items if item[2] <= 4]
            # sampled_elements = random.sample(target_items, min(len(target_items),5))
            # 确保每组至少有一个样本
            sample_high = random.sample(high_rating, min(3, len(high_rating)))  # 从高评分组中抽取 2 个
            sample_low = random.sample(low_rating, min(3, len(low_rating)))    # 从低评分组中抽取 3 
            sampled_data = sample_high + sample_low
            # 如果总样本数不足 6，从剩余数据中随机补充
            if len(sampled_data) < 6:
                remaining_data = [item for item in target_items if item not in sampled_data]
                sampled_data += random.sample(remaining_data, min(len(remaining_data),6 - len(sampled_data)))
            for i in range(len(sampled_data)):
                title, category, rating = sampled_data[i]
                if rating > 4:
                    label = 1
                else:
                    label = 0
                sampled_data[i][2] = label
                
            data_list.append({
                "user": user,
                'block': block_id,
                'targets': sampled_data,
                'cur_sum': d['cur_sum'],
                'pre_sum': d['pre_sum'],
                'cur_prompt': d['cur_prompt']
            })
    print(len(data_list))
    # prm_data = random.sample(data_list,2000)
    with open('/PRM/amz/block_len_50/prm_data.json','w',encoding='utf-8') as f:
        json.dump(data_list,f,ensure_ascii=False,indent=4)