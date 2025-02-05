# use BERT/chatGLM to encode the knowledge generated by LLM
import os
import json

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel
from utils import save_json, get_paragraph_representation

device = 'cuda'


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_history_text(data_path):
    raw_data = load_data(data_path)
    users, blocks, summarys, all = [], [], {},[]
    cur_prompts, cur_sums = [],[]
    for piece in raw_data:
        user, block, targets, cur_sum, pre_sum, cur_prompt = piece['user'],piece['block'],piece['targets'],piece['cur_sum'],piece['pre_sum'],piece['cur_prompt']
        users.append(user)
        blocks.append(block)
        # all_str = '\n'.join([cur_prompt,pre_sum,cur_sum])
        # all.append(all_str)        
        cur_prompts.append(pre_sum)
        cur_sums.append(cur_sum)
    return users,blocks, cur_prompts, cur_sums

# def get_history_text(data_path):
#     raw_data = load_data(data_path)
#     users, blocks, summarys, all = [], [], {},[]
#     for piece in raw_data:
#         user, block, targets, cur_sum, pre_sum, cur_prompt = piece['user'],piece['block'],piece['targets'],piece['cur_sum'],piece['pre_sum'],piece['cur_prompt']
#         users.append(user)
#         blocks.append(block)
#         all_str = '\n'.join([cur_prompt,pre_sum,cur_sum])
#         all.append(all_str)        
#     return users,blocks, all


def get_text_data_loader(data_path, batch_size):
    users, blocks, cur_prompts, cur_sums = get_history_text(os.path.join(data_path, 'prm_data.json'))
    # print('chatgpt.hist 1', history[1], 'hist len', len(history))
    # print('chatgpt.item 1', items[1], 'item len', len(items))
    cur_prompt_loader = DataLoader(cur_prompts, batch_size, shuffle=False)
    cur_sum_loader = DataLoader(cur_sums, batch_size, shuffle=False)
    # item_loader = DataLoader(items, batch_size, shuffle=False)
    return cur_prompt_loader, cur_sum_loader, users, blocks


def remap_hist(item_idxes,block_idxes, item_vec1,item_vec2):
    item_vec_map = {}
    for idx, block_id,vec_1,vec_2 in zip(item_idxes, block_idxes, item_vec1,item_vec2):
        if idx not in item_vec_map:
            item_vec_map[idx] = {}
        item_vec_map[idx][block_id] = [vec_1,vec_2]
    return item_vec_map

def remap_item(item_idxes, item_vec):
    item_vec_map = {}
    for idx, vec in zip(item_idxes, item_vec):
        item_vec_map[idx] = vec
    return item_vec_map

def inference(model, tokenizer, dataloader, model_name, aggregate_type):
    pred_list = []
    if model_name != 'bge':
        model.eval()
    with torch.no_grad():
        for x in tqdm(dataloader):
            torch.cuda.empty_cache()
            if model_name == 'chatglm' or model_name == 'chatglm2':
                x = tokenizer(x, padding=True, truncation=True, return_tensors="pt",
                              return_attention_mask=True).to(device)
                mask = x['attention_mask']
                x.pop('attention_mask')
                outputs = model.transformer(**x, output_hidden_states=True, return_dict=True)
                outputs.last_hidden_state = outputs.last_hidden_state.transpose(1, 0)
            elif model_name == 'longformer':
                x = tokenizer(x, padding=True, truncation=True, max_length=4096, return_tensors="pt",
                              return_attention_mask=True).to(device)
                mask = x['attention_mask']
                outputs = model(**x, output_hidden_states=True, return_dict=True)
            elif model_name == 'bge':
                # x = tokenizer(x, padding=True, truncation=True, max_length=8192, return_tensors="pt",
                #               return_attention_mask=True).to(device)
                # mask = x['attention_mask']
                # outputs = model(**x, output_hidden_states=True, return_dict=True)
                outputs = model.encode(x, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
            else:
                x = tokenizer(x, padding=True, truncation=True, max_length=512, return_tensors="pt",
                              return_attention_mask=True).to(device)
                mask = x['attention_mask']
                outputs = model(**x, output_hidden_states=True, return_dict=True)
            if model_name == 'bge':
                pred_list.extend(outputs.tolist())
            else:
                pred = get_paragraph_representation(outputs, mask, aggregate_type)
                pred_list.extend(pred.tolist())
    return pred_list


def main(knowledge_path, data_path, model_name, batch_size, aggregate_type):
    cur_prompt_loader, cur_sum_loader, hist_idxes, block_idxes = get_text_data_loader(knowledge_path, batch_size)

    if model_name == 'chatglm':
        checkpoint = '../llm/chatglm-6b' if os.path.exists('../../llm/chatglm-6b') else 'chatglm-6b'
    elif model_name == 'chatglm2':
        checkpoint = '../llm/chatglm-v2' if os.path.exists('../../llm/chatglm-v2') else 'chatglm-v2'
    elif model_name == 'bert':
        checkpoint = '/llm/bert-base-uncased' if os.path.exists('/llm/bert-base-uncased') else 'bert-base-uncased'
    elif model_name == 'longformer':
        checkpoint = '../llm/longformer-base-4096' if os.path.exists('/llm/longformer-base-4096') else 'longformer-base-4096'
    elif model_name == 'bge':
        checkpoint = '/llm/bge-m3' if os.path.exists('/llm/bge-m3') else 'bge-m3'
    else:
        raise NotImplementedError

    torch.cuda.empty_cache()
    print(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,  trust_remote_code=True)
    if model_name == 'bge':
        model = BGEM3FlagModel(checkpoint,  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    else:
        model = AutoModel.from_pretrained(checkpoint,  trust_remote_code=True).half().cuda()

    # item_vec = inference(model, tokenizer, item_loader, model_name, aggregate_type)
    hist_vec1 = inference(model, tokenizer, cur_prompt_loader, model_name, aggregate_type)
    hist_vec2 = inference(model, tokenizer, cur_sum_loader, model_name, aggregate_type)
    # item_vec_dict = remap_item(item_idxes, item_vec)
    hist_vec_dict = remap_hist(hist_idxes,block_idxes, hist_vec1,hist_vec2)
    print(len(hist_vec_dict))
    # print(len(item_vec_dict))

    # save_json(item_vec_dict, os.path.join(data_path, '{}_{}_augment.item'.format(model_name, aggregate_type)))
    save_json(hist_vec_dict, os.path.join(data_path, '{}_{}_pre_cur_emb.sum'.format(model_name, aggregate_type)))


if __name__ == '__main__':
    KLG_DATA_DIR = '/PRM/ml-1m/block_len_50'
    SAVE_DATA_DIR = '/PRM/ml-1m/block_len_50'
    # DATA_SET_NAME = 'amz'
    DATA_SET_NAME = 'ml-1m'
    KLG_PATH = KLG_DATA_DIR
    DATA_PATH = SAVE_DATA_DIR
    # MODEL_NAME = 'chatglm'
    # MODEL_NAME = 'chatglm2'
    MODEL_NAME = 'bge'  # bert, chatglm, chatglm2
    AGGREGATE_TYPE = 'avg'  # last, avg, wavg, cls, ...
    BATCH_SIZE = 16 if MODEL_NAME == 'bert' or MODEL_NAME == 'bge' else 2
    main(KLG_PATH, DATA_PATH, MODEL_NAME, BATCH_SIZE, AGGREGATE_TYPE)

