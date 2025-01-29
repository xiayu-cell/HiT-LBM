# use BERT/chatGLM to encode the knowledge generated by LLM
import os
import json

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from utils import save_json, get_paragraph_representation
device = 'cuda'


def load_data(path):
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for id, value in data.items():
            res.append([id, value['prompt'], value['ans']])
    return res


def get_history_text(data_path):
    raw_data = load_data(data_path)
    idx_list, hist_text = [], []
    for piece in raw_data:
        idx, prompt, answer = piece
        pure_hist = prompt[::-1].split(';', 1)[-1][::-1]
        hist_text.append(pure_hist + '. ' + answer)
        idx_list.append(idx)
    return idx_list, hist_text


def get_item_text(data_path):
    raw_data = load_data(data_path)
    idx_list, text_list = [], []
    for piece in raw_data:
        idx, prompt, answer = piece
        text_list.append(answer)
        idx_list.append(idx)
    return idx_list, text_list


def get_text_data_loader(data_path, batch_size):
    hist_idxes, history = get_history_text(os.path.join(data_path, 'user.klg'))
    print('chatgpt.hist 1', history[1], 'hist len', len(history))
    item_idxes, items = get_item_text(os.path.join(data_path, 'item.klg'))
    print('chatgpt.item 1', items[1], 'item len', len(items))

    history_loader = DataLoader(history, batch_size, shuffle=False)
    item_loader = DataLoader(items, batch_size, shuffle=False)
    return history_loader, hist_idxes, item_loader, item_idxes


def remap_item(item_idxes, item_vec):
    item_vec_map = {}
    for idx, vec in zip(item_idxes, item_vec):
        item_vec_map[idx] = vec
    return item_vec_map


def inference(model, tokenizer, dataloader, model_name, aggregate_type):
    pred_list = []
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
            else:
                x = tokenizer(x, padding=True, truncation=True, max_length=512, return_tensors="pt",
                              return_attention_mask=True).to(device)
                mask = x['attention_mask']
                outputs = model(**x, output_hidden_states=True, return_dict=True)
            pred = get_paragraph_representation(outputs, mask, aggregate_type)
            pred_list.extend(pred.tolist())
    return pred_list


def main(knowledge_path, data_path, model_name, batch_size, aggregate_type):
    hist_loader, hist_idxes, item_loader, item_idxes = get_text_data_loader(knowledge_path, batch_size)

    if model_name == 'chatglm':
        checkpoint = '../../llm/chatglm-6b' if os.path.exists('../../llm/chatglm-6b') else 'chatglm-6b'
    elif model_name == 'chatglm2':
        checkpoint = '../../llm/chatglm-v2' if os.path.exists('../../llm/chatglm-v2') else 'chatglm-v2'
    elif model_name == 'bert':
        checkpoint = '../../llm/bert-base-uncased' if os.path.exists('../../llm/chatglm-6b') else 'bert-base-uncased'
    else:
        raise NotImplementedError

    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,  trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint,  trust_remote_code=True).half().cuda()

    item_vec = inference(model, tokenizer, item_loader, model_name, aggregate_type)
    hist_vec = inference(model, tokenizer, hist_loader, model_name, aggregate_type)
    item_vec_dict = remap_item(item_idxes, item_vec)
    hist_vec_dict = remap_item(hist_idxes, hist_vec)

    save_json(item_vec_dict, os.path.join(data_path, '{}_{}_augment.item'.format(model_name, aggregate_type)))
    save_json(hist_vec_dict, os.path.join(data_path, '{}_{}_augment.hist'.format(model_name, aggregate_type)))

    stat_path = os.path.join(data_path, 'stat.json')
    with open(stat_path, 'r') as f:
        stat = json.load(f)

    stat['dense_dim'] = 4096 if model_name == 'chatglm' or model_name == 'chatglm2' else 768
    with open(stat_path, 'w') as f:
        stat = json.dumps(stat)
        f.write(stat)


if __name__ == '__main__':
    DATA_DIR = '../data/'
    # DATA_SET_NAME = 'amz'
    DATA_SET_NAME = 'ml-1m'
    KLG_PATH = os.path.join(DATA_DIR, DATA_SET_NAME, 'knowledge')
    DATA_PATH = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    # MODEL_NAME = 'chatglm'
    # MODEL_NAME = 'chatglm2'
    MODEL_NAME = 'bert'  # bert, chatglm, chatglm2
    AGGREGATE_TYPE = 'avg'  # last, avg, wavg, cls, ...
    BATCH_SIZE = 16 if MODEL_NAME == 'bert' else 2
    main(KLG_PATH, DATA_PATH, MODEL_NAME, BATCH_SIZE, AGGREGATE_TYPE)

