import json
import os
import argparse
from torch.utils.data import DataLoader
import multiprocessing
from vllm import LLM, SamplingParams
# chat_template = '<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import tqdm as tq
import random
import math
import pdb
import torch
import torch.nn as nn
from prm_amz import MLPClassifier
from FlagEmbedding import BGEM3FlagModel
import pdb
import gc

norm_qwen_prompt = "You are a helpful assistant."
short_qwen_prompt = "You are a helpful assistant. You will provide very concise and helpful response."

def update_prm(block_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 每个blockinfer完，用prm对其进行打分
    block_interest_folder = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/amz_prm'
    if block_id >= 0:
        cur_block_interest = {}
        for filename in os.listdir(block_interest_folder):
            if filename.startswith(f'interest_{block_id}') and filename.endswith(".json"):
                file_path = os.path.join(block_interest_folder, filename)
            # 读取 JSON 文件
                with open(file_path, 'r') as file:
                    # pdb.set_trace()
                    t = json.load(file)
                    cur_block_interest.update(t)
    # print(cur_block_interest)
    users,all,user_hist_cur_sum = [],[],[]
    for user, blocks in cur_block_interest.items():
        cur_sums, pre_sum, cur_prompt = blocks[str(block_id)]['cur_sum'],blocks[str(block_id)]['pre_sum'],blocks[str(block_id)]['cur_prompt']
        for cur_sum in cur_sums:
            all_str = '\n'.join([cur_prompt,cur_sum])
            all.append(all_str)
            users.append(user)
            user_hist = cur_prompt.split('[Current Book Viewing History]')[-1].split('\n')[1]
            user_hist_cur_sum.append(user_hist+cur_sum)
            
    sum_loader = DataLoader(all, 16, shuffle=False)
    point_loader = DataLoader(user_hist_cur_sum,16,shuffle=False)
    seq_pred_list,point_pred_list = [],[]
    print(f'{torch.cuda.device_count()} 个 gpu ')
    bge_encoder = BGEM3FlagModel(bge_path,  use_fp16=True)
    seq_prm = MLPClassifier(input_dim,hidden_dim,output_dim)
    point_prm = MLPClassifier(input_dim,hidden_dim,output_dim)

    seq_prm.load_state_dict(torch.load(seq_prm_path))
    point_prm.load_state_dict(torch.load(point_prm_path))

    for x in tqdm(sum_loader): 
        outputs = bge_encoder.encode(x, 
                        max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                        )['dense_vecs']
        prm = seq_prm.to(device)
        outputs = torch.tensor(outputs).float().to(device)
        # print(outputs.shape)
        preds = prm(outputs)
        preds = softmax(preds)
        seq_pred_list.extend(preds[:,1].cpu().tolist())

    for x in tqdm(point_loader): 
        outputs = bge_encoder.encode(x, 
                        max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                        )['dense_vecs']
        prm = point_prm.to(device)
        outputs = torch.tensor(outputs).float().to(device)
        # print(outputs.shape)
        preds = prm(outputs)
        preds = softmax(preds)
        point_pred_list.extend(preds[:,1].cpu().tolist())

    # pdb.set_trace()
    for idx,seq_pred,point_pred in zip(users, seq_pred_list,point_pred_list):
        if 'seq_preds' not in cur_block_interest[idx][str(block_id)]:
            cur_block_interest[idx][str(block_id)]['seq_preds'] = []
        if 'point_preds' not in cur_block_interest[idx][str(block_id)]:
            cur_block_interest[idx][str(block_id)]['point_preds'] = []
        if 'preds' not in cur_block_interest[idx][str(block_id)]:
            cur_block_interest[idx][str(block_id)]['preds'] = []
        cur_block_interest[idx][str(block_id)]['seq_preds'].append(seq_pred)
        cur_block_interest[idx][str(block_id)]['point_preds'].append(point_pred)
        cur_block_interest[idx][str(block_id)]['preds'].append((point_pred+seq_pred)/2)

        
    with open(f'/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/amz_prm/prm_interest_{block_id}.json', 'w') as f:
        json.dump(cur_block_interest,f,ensure_ascii=False,indent=4)
    del device
    del bge_encoder
    gc.collect()
    return 0

def run_inference(model_path,block_id,batch, bs, thread_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{thread_id}'
    prompts = []
    questions = []
    users = []
    pre_interests = []
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True)

    # block_id = item['block_id']
    for item in batch:
        user_id = item['user_id']
        # block_id = item['block_id']
        hist_prompt = item['hist_prompt']
        pre_interest = item['pre_interest']
        user_desc, hist_item, question = hist_prompt[0],hist_prompt[1],hist_prompt[2]

        PRIVIOUS_INTEREST = '[Previous Interest Summary]'
        CURRENT_HIST = '[Current Book Viewing History]'
        hist_item = ' '.join(hist_item)
        prompt = '\n'.join([user_desc,PRIVIOUS_INTEREST,pre_interest, CURRENT_HIST, hist_item, question])
        messages = [
                {"role": "system", "content": norm_qwen_prompt},
                {"role": "user", "content": prompt}
            ]

        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        questions.append(prompt)
        prompts.append(text)
        users.append(user_id)
        pre_interests.append(pre_interest)

    # 初始化 VLLM 模型，指定 GPU
    #sampling_params = SamplingParams(temperature=1.0, top_p=0.9, repetition_penalty=1.05, max_tokens=512)
    # sampling_param = SamplingParams(temperature=0.5,max_tokens=4096,repetition_penalty=1.05)
    sampling_params = [SamplingParams(temperature=0.9,max_tokens=4096, seed=random.randint(0,1e9)) for _ in tq.trange(0, 10 ,desc='sample param')]

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_path,dtype='half',trust_remote_code=True)
    # 保存结果到文件
    # generated_texts = []
    res = {}

    with open(f'/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/amz_prm/interest_{block_id}_{thread_id}.json', 'w') as f:
        for i in tqdm(range(0,len(prompts),bs)):
            msg = prompts[i:min(i+bs,len(prompts))]
            ques = questions[i:min(i+bs,len(prompts))]
            u = users[i:min(i+bs,len(prompts))]
            pre_interest = pre_interests[i:min(i+bs,len(prompts))]
            # 进行批量推理
            # generated_texts = []
            # outputs = llm.generate(msg, sampling_param)
            # for num in range(len(outputs)):
            #     generated_texts.append(outputs[num].outputs[0].text)
            oots = [[] for num in range(len(msg))] 
            for j, sampling_param in enumerate(sampling_params):
                outputs = llm.generate(msg, sampling_param)
                for num in range(len(outputs)):
                    oots[num].append(outputs[num].outputs[0].text)
            # 处理输出结果
            quess = []
            for user, q,pre_interest, output in zip(u,ques,pre_interest,oots):
                res[user] = {}
                res[user][str(block_id)] = {}
                res[user][str(block_id)]['pre_sum'] =  pre_interest
                res[user][str(block_id)]['cur_sum'] =  output
                res[user][str(block_id)]['cur_prompt'] = q

                # quess.append(q)
        json.dump(res,f,ensure_ascii=False,indent=4)

        # with open(f'/mmu_nlp_hdd/xiayu12/LIBER_ours_train/preference_generation/ml-1m/summary/question_{block_id}_{thread_id}.json', 'w') as file:
        #     json.dump(quess,file,ensure_ascii=False,indent=4)
            # break

if __name__ == "__main__":
    prompt_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/data/amz/proc_data/block_len_50/all_prompt.hist'
    model_path = '/share/ad/xiayu12/Open-World-Knowledge-Augmented-Recommendation_Gang/checkpoints/Qwen/Qwen2___5-7B-Instruct'
    seq_prm_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/checkpoint/amz/cls_epoch_2_threshold_0.5.pth'
    point_prm_path = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/ckp/amz/cls_epoch_0_threshold_0.5.pth'
    bge_path = '/mmu_nlp_hdd/xiayu12/LIBER/llm/bge-m3'
    with open(prompt_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    # data = {'4683':data['4683']}
    ng = torch.cuda.device_count()

    ng = 8
    multiprocessing.set_start_method('spawn')
    #batch_size = total_prompts // 8
    input_dim = 1024  # 输入维度，与嵌入维度一致
    hidden_dim = 128  # 隐藏层维度
    output_dim = 2 
    
    softmax = nn.Softmax(dim=-1)

    print('Load successful')
    for block_id in range(0,200):
        data_list = []

        block_interest_folder = '/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/amz_prm'

        if block_id > 0:
            pre_block_interest = {}
            for filename in os.listdir(block_interest_folder):
                if filename.startswith(f'prm_interest_{block_id-1}') and filename.endswith(".json"):
                    file_path = os.path.join(block_interest_folder, filename)
                # 读取 JSON 文件
                    with open(file_path, 'r') as file:
                        # pdb.set_trace()
                        t = json.load(file)
                        pre_block_interest.update(t)

        for user_id, blocks in data.items():
            
            if str(block_id) in blocks:
                if block_id == 0:
                    pre_interest = "None"
                else:
                    # pdb.set_trace()
                    preds = pre_block_interest[user_id][str(block_id-1)]['preds']
                    max_index = preds.index(max(preds))
                    pre_interest = pre_block_interest[user_id][str(block_id-1)]['cur_sum'][max_index]

                data_list.append({
                    'user_id':user_id,
                    'block_id': str(block_id),
                    'hist_prompt': blocks[str(block_id)],
                    'pre_interest': pre_interest
                })

        # 计算每份数据的大小
        print(len(data_list))
        if len(data_list) ==0:
            break
        total_prompts = len(data_list)
        
        batch_size = math.ceil(total_prompts / ng)

        # 分割数据
        batches = [data_list[i:min(i + batch_size, total_prompts)] for i in range(0, total_prompts, batch_size)]
        # 如果数据不能均匀分割，最后一组可能会有剩余的数据
        if len(batches) > ng:
            batches[ng - 1].extend(batches[ng])
            batches = batches[:ng]
        print("总数据: {}".format(total_prompts))
        print("每份数据:", [len(val) for val in batches])
        if len(batches) < ng:
            ng = len(batches)
        processes = []

        for i in range(ng):
            p = multiprocessing.Process(target=run_inference, args=(model_path,block_id,batches[i], 1000, i))
            processes.append(p)
            p.start()
        for p in processes:
            p.join() # 阻塞进程，当多线程执行完，再进入外循环

        update_prm(block_id)
        
        
            
        print(f"All processes for block {block_id} are finished.")
