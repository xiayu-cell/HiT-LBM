import json
import os
import argparse

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

norm_qwen_prompt = "You are a helpful assistant."
short_qwen_prompt = "You are a helpful assistant. You will provide very concise and helpful response."

def run_inference(model_path,block_id,batch, bs, thread_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{4+thread_id}'
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
        CURRENT_HIST = '[Current Movie Viewing History]'
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
    sampling_param = SamplingParams(temperature=0.5,max_tokens=4096,repetition_penalty=1.05)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_path,dtype='half',trust_remote_code=True)
    # 保存结果到文件
    # generated_texts = []
    res = {}

    with open(f'/preference_generation/ml-1m/block_len_50/summary/interest_{block_id}_{thread_id}.json', 'w') as f:
        for i in tqdm(range(0,len(prompts),bs)):
            msg = prompts[i:min(i+bs,len(prompts))]
            ques = questions[i:min(i+bs,len(prompts))]
            u = users[i:min(i+bs,len(prompts))]
            pre_interest = pre_interests[i:min(i+bs,len(prompts))]
            # 进行批量推理
            generated_texts = []
            outputs = llm.generate(msg, sampling_param)
            for num in range(len(outputs)):
                generated_texts.append(outputs[num].outputs[0].text)
            # 处理输出结果
            quess = []
            for user, q,pre_interest, output in zip(u,ques,pre_interest,generated_texts):
                res[user] = {}
                res[user][str(block_id)] = {}
                res[user][str(block_id)]['pre_sum'] =  pre_interest
                res[user][str(block_id)]['cur_sum'] =  output
                res[user][str(block_id)]['cur_prompt'] = q

                # quess.append(q)
       

if __name__ == "__main__":
    prompt_path = '/data/ml-1m/proc_data/block_len_50/all_prompt.hist'
    model_path = '/checkpoints/Qwen/Qwen2___5-7B-Instruct'

    with open(prompt_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    import torch
    ng = torch.cuda.device_count()

    ng = 4

    #batch_size = total_prompts // 8

    for block_id in range(0,200):
        data_list = []

        block_interest_folder = '/preference_generation/ml-1m/block_len_50/summary'

        if block_id > 0:
            pre_block_interest = {}
            for filename in os.listdir(block_interest_folder):
                if filename.startswith(f'interest_{block_id-1}') and filename.endswith(".json"):
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
                    pre_interest = pre_block_interest[user_id][str(block_id-1)]['cur_sum']
                data_list.append({
                    'user_id':user_id,
                    'block_id': str(block_id),
                    'hist_prompt': blocks[str(block_id)],
                    'pre_interest': pre_interest
                })

        # pdb.set_trace()
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

        # pdb.set_trace()


        for i in range(ng):
            p = multiprocessing.Process(target=run_inference, args=(model_path,block_id,batches[i], 1000, i))
            processes.append(p)
            p.start()
        for p in processes:
            p.join() # 阻塞进程，当多线程执行完，再进入外循环
        print(f"All processes for block {block_id} are finished.")
    # for i, batch in enumerate(batches):
    #     run_inference(model_path, batch, 20, i, 10)
    
    # for p in processes:
    #     p.join()
