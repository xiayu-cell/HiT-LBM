import json
import os
import pickle
from datetime import date
import random
from collections import defaultdict
import csv
from pre_utils import load_json, save_json, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING

rerank_item_from_hist = 4
rerank_hist_len = 10
rerank_list_len = 10
ctr_hist_len = 10
start_idx = 200
block_len = 200

def generate_ctr_data(sequence_data, lm_hist_idx,train_test_split_index, is_train):
    # print(list(lm_hist_idx.values())[:10])
    full_data = []
    total_label = []
    for uid in sequence_data.keys():
        start = lm_hist_idx
        item_seq, rating_seq = sequence_data[str(uid)]
        if not is_train:
            start = 0
        for idx in range(start, len(item_seq)):
            seq_id = idx
            label = 1 if rating_seq[idx] > rating_threshold else 0
            if is_train:
                block_id = int(seq_id/block_len)-1
            else:
                seq_id = train_test_split_index[uid]+idx
                block_id = int(seq_id/block_len)-1
            if block_id < 0:
                continue

            #------------------------
            # 去掉block数少于2的blcok
            # if block_id <=2:
            #     continue
            #------------------------
            full_data.append([uid, block_id, seq_id, label])
            total_label.append(label)
    print('user num', len(sequence_data), 'data num', len(full_data), 'pos ratio',
          sum(total_label) / len(total_label))
    print(full_data[:5])
    return full_data


def generate_rerank_data(sequence_data, lm_hist_idx, uid_set, item_set):
    full_data = []
    for uid in uid_set:
        start_idx = lm_hist_idx[str(uid)]
        item_seq, rating_seq = sequence_data[str(uid)]
        idx = start_idx
        seq_len = len(item_seq)
        while idx < seq_len:
            end_idx = min(idx + rerank_item_from_hist, seq_len)
            chosen_iid = item_seq[idx:end_idx]
            neg_sample_num = rerank_list_len - len(chosen_iid)
            neg_sample = random.sample(item_set, neg_sample_num)
            candidates = chosen_iid + neg_sample
            chosen_rating = rating_seq[idx:end_idx]
            candidate_lbs = [1 if rating > rating_threshold else 0 for rating in
                             chosen_rating] + [0 for _ in range(neg_sample_num)]
            list_zip = list(zip(candidates, candidate_lbs))
            random.shuffle(list_zip)
            candidates[:], candidate_lbs[:] = zip(*list_zip)
            full_data.append([uid, idx, candidates, candidate_lbs])
            idx = end_idx
    print('user num', len(uid_set), 'data num', len(full_data))
    print(full_data[:5])
    return full_data


def generate_hist_prompt(sequence_data, item2attribute, datamap, lm_hist_idx, dataset_name):
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2user = datamap['id2user']
    if dataset_name == 'ml-1m':
        user2attribute = datamap['user2attribute']
    hist_prompts = {}
    print('item2attribute', list(item2attribute.items())[:10])
    block2id = {}
    for uid, item_rating in sequence_data.items():
        user = id2user[uid]
        item_seq, rating_seq = item_rating
        if user not in block2id:
            block2id[user] = {}
        for i in range(0,len(item_seq),block_len):
            cur_idx = i
            block_id = int(cur_idx/block_len)
            if block_id not in block2id[user]:
                block2id[user][block_id] = []
            hist_item_seq = item_seq[cur_idx:cur_idx+block_len]
            hist_rating_seq = rating_seq[cur_idx:cur_idx+block_len]
            history_texts = []
            for iid, rating in zip(hist_item_seq, hist_rating_seq):
                if dataset_name == 'amz':
                    block2id[user][block_id].append([itemid2title[str(iid)],attrid2name[str(item2attribute[str(iid)][1])],int(rating)])
                    tmp = 'book title: {}, category: {}, user\'s rating: {} stars; '.format(itemid2title[str(iid)], attrid2name[str(item2attribute[str(iid)][1])],int(rating))
                elif dataset_name == 'ml-1m':
                    block2id[user][block_id].append([itemid2title[str(iid)], attrid2name[str(item2attribute[str(iid)][0])],int(rating)])
                    tmp = '"movie title: {}", category: {}, user\'s rating: {} stars; '.format(itemid2title[str(iid)], attrid2name[str(item2attribute[str(iid)][0])],int(rating))
                history_texts.append(tmp)
            if dataset_name == 'amz':
            # prompt = 'Analyze user\'s preferences on product (consider factors like genre, functionality, quality, ' \
            #          'price, design, reputation. Provide clear explanations based on ' \
            #          'relevant details from the user\'s product viewing history and other pertinent factors.'
            # hist_prompts[uid] = 'Given user\'s product rating history: ' + ''.join(history_texts) + prompt
                user_text = 'User\'s previous book interest summary and current book viewing history' \
                        ' are listed below.  Book viewing history includes the Book title, category, and the user\'s rating. If the user\'s rating is greater than 4, it is considered that the user likes the book; if the rating is less than or equal to 4, it is considered that the user dislikes the book.'

                question_sum = 'Analyze user\'s preferences on books about factors like genre, author, writing style, theme, ' \
                     'setting, length and complexity, time period, literary quality, critical acclaim '\
                        '(Provide clear explanations based on relevant details from the user\'s book ' \
                       'viewing history and previous book interest summary. '\
                       'If the user\'s interests have changed, describe the content of this change and analyze the reasons behind the shift in interests.'
                # question_shift = 'User previous explanations: [[{}}]], User current explanations: [[{}}]].'\
                #         ' Are there any new preferences in user current explanations that differs from user previous explanations? If yes, list these preferences on movies'\
                #         '(consider factors like genre, author, writing style, theme, ' \
                #      'setting, length and complexity, time period, literary quality, critical acclaim (Provide ' \
                #      'clear explanations based on relevant details from the user\'s book viewing history and other ' \
                #      'pertinent factors.'
                # hist_prompts[user] = 'Given user\'s book rating history: ' + ''.join(history_texts) + prompt

                if user not in hist_prompts:
                    hist_prompts[user] = {}
                hist_prompts[user][str(block_id)] = [user_text, history_texts, question_sum]
            elif dataset_name == 'ml-1m':
                gender, age, occupation = user2attribute[user]
                user_text = 'Given a {} user who is aged {} and {}, this user\'s previous movie interest summary and current movie viewing history' \
                        ' are listed below.  Movie viewing history includes the movie title, category, and the user\'s rating. If the user\'s rating is greater than or equal to 4, it is considered that the user likes the movie; if the rating is less than 4, it is considered that the user dislikes the movie.'.format(GENDER_MAPPING[gender], AGE_MAPPING[age],
                                                    OCCUPATION_MAPPING[occupation])
                question_sum = 'Analyze user\'s preferences on movies (consider factors like genre, director/actors, time ' \
                       'period/country, character, plot/theme, mood/tone, critical acclaim/award, production quality, ' \
                       'and soundtrack). Provide clear explanations based on relevant details from the user\'s movie ' \
                       'viewing history and previous movie interest summary. '\
                       'If the user\'s interests have changed, describe the content of this change and analyze the reasons behind the shift in interests.'
                # question_shift = 'User previous explanations: [[{}}]], User current explanations: [[{}}]].'\
                #         ' Are there any new preferences in user current explanations that differs from user previous explanations? If yes, list these preferences on movies'\
                #         '(consider factors like genre, director/actors, time ' \
                #         'period/country, character, plot/theme, mood/tone, critical acclaim/award, production quality, ' \
                #         'and soundtrack).'
                # if user not in hist_prompts:
                #     hist_prompts[user] = [{
                #         'block_id': block_id,
                #         'hist_prompt':[user_text, history_texts, question]
                #     }]
                # else:
                #     hist_prompts[user].append(
                #         {
                #         'block_id': block_id,
                #         'hist_prompt':[user_text, history_texts, question]
                #     }
                #     )
                if user not in hist_prompts:
                    hist_prompts[user] = {}
                hist_prompts[user][str(block_id)] = [user_text, history_texts, question_sum]

            else:
                raise NotImplementedError
    print('data num', len(hist_prompts))
    print(list(hist_prompts.items())[0])
    return hist_prompts,block2id


def generate_item_prompt(item2attribute, datamap, dataset_name):
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']

    id2item = datamap['id2item']
    item_prompts = {}
    for iid, title in itemid2title.items():
        item = id2item[iid]
        if dataset_name == 'amz':
            brand, cate = item2attribute[str(iid)]
            # brand_name = attrid2name[str(brand)]
            cate_name = attrid2name[str(cate)]
            item_prompts[item] = 'Introduce book {}, which is {} and describe its attributes including but' \
                                ' not limited to genre, author, writing style, theme, setting, length and complexity, ' \
                                'time period, literary quality, critical acclaim.'.format(title, cate_name)
            # item_prompts[iid] = 'Introduce product {}, which is from brand {} and describe its attributes (including but' \
            #                     ' not limited to genre, functionality, quality, price, design, reputation).'.format(title, brand_name)
        elif dataset_name == 'ml-1m':
            item_prompts[item] = 'Given a movie which title is {} and categoty is {}, introduce and describe its attributes (including but not limited to genre, ' \
                                'director/cast, country, character, plot/theme, mood/tone, critical ' \
                                'acclaim/award, production quality, and soundtrack).'.format(title,attrid2name[str(item2attribute[str(iid)][0])])
        else:
            raise NotImplementedError
    print('data num', len(item_prompts))
    print(list(item_prompts.items())[0])
    return item_prompts


if __name__ == '__main__':
    random.seed(12345)
    DATA_DIR = '../data/'
    # DATA_SET_NAME = 'amz'
    DATA_SET_NAME = 'ml-1m'
    if DATA_SET_NAME == 'ml-1m':
        rating_threshold = 3
    else:
        rating_threshold = 4
    
    # BLOCK_LEN = 

    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data','block_len_'+str(block_len))
    SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'sequential_data.json')
    TRAIN_SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'train.json')
    TEST_SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'test.json')
    TRAIN_TEST_SPLIT_INDEX = os.path.join(PROCESSED_DIR, 'train_test_split_index.json')

    ITEM2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'item2attributes.json')
    DATAMAP_PATH = os.path.join(PROCESSED_DIR, 'datamaps.json')
    # SPLIT_PATH = os.path.join(PROCESSED_DIR, 'train_test_split.json')

    all_sequence_data = load_json(SEQUENCE_PATH)
    train_sequence_data = load_json(TRAIN_SEQUENCE_PATH)
    test_sequence_data = load_json(TEST_SEQUENCE_PATH)
    train_test_split_index = load_json(TRAIN_TEST_SPLIT_INDEX)

    # train_test_split = load_json(SPLIT_PATH)
    item2attribute = load_json(ITEM2ATTRIBUTE_PATH)
    item_set = list(map(int, item2attribute.keys()))
    print('final loading data')
    # print(list(item2attribute.keys())[:10])

    print('generating ctr train dataset')
    train_ctr = generate_ctr_data(train_sequence_data, start_idx,train_test_split_index,True)
    print('generating ctr test dataset')
    test_ctr = generate_ctr_data(test_sequence_data, start_idx,train_test_split_index,False)
    print('save ctr data')
    save_pickle(train_ctr, PROCESSED_DIR + '/ctr.train')
    save_pickle(test_ctr, PROCESSED_DIR + '/ctr.test')
    train_ctr, test_ctr = None, None
    # exit(0)
    # print('generating reranking train dataset')
    # train_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'],
    #                                     train_test_split['train'], item_set)
    # print('generating reranking test dataset')
    # test_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'],
    #                                    train_test_split['test'], item_set)
    # print('save reranking data')
    # save_pickle(train_rerank, PROCESSED_DIR + '/rerank.train')
    # save_pickle(test_rerank, PROCESSED_DIR + '/rerank.test')
    train_rerank, test_rerank = None, None

    datamap = load_json(DATAMAP_PATH)

    statis = {
        'rerank_list_len': rerank_list_len,
        'attribute_ft_num': datamap['attribute_ft_num'],
        'rating_threshold': rating_threshold,
        'item_num': len(datamap['id2item']),
        'attribute_num': len(datamap['id2attribute']),
        'rating_num': 5,
        'dense_dim': 0,
    }
    save_json(statis, PROCESSED_DIR + '/stat.json')

    print('generating item prompt')
    item_prompt = generate_item_prompt(item2attribute, datamap, DATA_SET_NAME)
    print('generating history prompt')
    hist_prompt,block2id = generate_hist_prompt(all_sequence_data, item2attribute, datamap,
                                       block_len, DATA_SET_NAME)

    save_json(block2id, PROCESSED_DIR + '/block2item.json')
    print('save prompt data')
    print(len(item_prompt))
    save_json(item_prompt, PROCESSED_DIR + '/prompt.item')
    print(len(hist_prompt))
    save_json(hist_prompt, PROCESSED_DIR + '/all_prompt.hist')
    item_prompt, hist_prompt = None, None

