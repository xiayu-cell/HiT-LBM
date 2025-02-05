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

def run_inference(model_path,batch, bs, thread_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{4+thread_id}'
    prompts = []
    questions = []
    item_ids = []
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True)

    # block_id = item['block_id']
    for item in batch:
        item_id = item['item_id']
        messages = [
                {"role": "system", "content": norm_qwen_prompt},
                {"role": "user", "content": item['prompt']}
            ]

        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        questions.append(item['prompt'])
        prompts.append(text)
        item_ids.append(item_id)

    # 初始化 VLLM 模型，指定 GPU
    #sampling_params = SamplingParams(temperature=1.0, top_p=0.9, repetition_penalty=1.05, max_tokens=512)
    sampling_param = SamplingParams(temperature=0,max_tokens=4096,repetition_penalty=1.05)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_path,dtype='half',trust_remote_code=True)
    # 保存结果到文件
    # generated_texts = []
    res = {}

    with open(f'/preference_generation/ml-1m/block_len_100/item_knowledge/item_knowledge_{thread_id}.json', 'w') as f:
        for i in tqdm(range(0,len(prompts),bs)):
            msg = prompts[i:min(i+bs,len(prompts))]
            ques = questions[i:min(i+bs,len(prompts))]
            item_id = item_ids[i:min(i+bs,len(prompts))]

            # 进行批量推理
            generated_texts = []
            outputs = llm.generate(msg, sampling_param)
            for num in range(len(outputs)):
                generated_texts.append(outputs[num].outputs[0].text)
            # 处理输出结果
            quess = []
            for iid, q, output in zip(item_id,ques,generated_texts):
                res[iid] = output
        json.dump(res,f,ensure_ascii=False,indent=4)
        

if __name__ == "__main__":
    prompt_path = '/data/ml-1m/proc_data/block_len_100/prompt.item'
    model_path = '/checkpoints/Qwen/Qwen2___5-7B-Instruct'

    with open(prompt_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    
    import torch
    ng = torch.cuda.device_count()

    ng = 4
    data_list = []
    for k,v in data.items():
        data_list.append({
            'item_id': k,
            'prompt': v
        })
    print(len(data_list))
    if len(data_list) ==0:
        exit(0)
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
        p = multiprocessing.Process(target=run_inference, args=(model_path,batches[i], 1000, i))
        processes.append(p)
        p.start()
    # for p in processes:
    #     p.join() # 阻塞进程，当多线程执行完，再进入外循环
    # print(f"All processes for block {block_id} are finished.")
    # for i, batch in enumerate(batches):
    #     run_inference(model_path, batch, 20, i, 10)
    
    # for p in processes:
    #     p.join()