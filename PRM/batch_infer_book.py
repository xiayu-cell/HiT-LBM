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
import random
norm_qwen_prompt = "You are a helpful assistant."
short_qwen_prompt = "You are a helpful assistant. You will provide very concise and helpful response."

def run_inference(model_path,batch, bs, thread_id):
    # os.environ["CUDA_VISIBLE_DEVICES"] = f'{thread_id*2},{thread_id*2+1}'
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{thread_id}'
    prompts = []
    questions = []
    users = []
    blocks = []
    labels = []
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True)

    for item in batch:
        user_id = item['user']
        block_id = item['block']
        targets = item['targets']
        cur_sum = item['cur_sum']
        pre_sum = item['pre_sum']
        cur_prompt = item['cur_prompt']
        user_hist = cur_prompt.split('[Current Movie Viewing History]')[-1].split('\n')[1]
        prefix = 'User\'s current book viewing history are listed below.  Book viewing history includes the Book title, category, and the user\'s rating. If the user\'s rating is greater than 4, it is considered that the user likes the book; if the rating is less than or equal to 4, it is considered that the user dislikes the book.'

        prompt = 'Based on the user\'s reading history, please determine whether the user will like the target book. Output only Yes or No'
        
        current_sum = '[Current book viewing history]'
        target_item = '[Target Movie]'
        # targets = [targets[0]]
        for t in targets:
            title, category, lb = t
            ti = f'book: {title}, category: {category}'
            prompt = '\n'.join([prefix,current_sum,user_hist, target_item, ti,prompt])
            messages = [
                {"role": "system", "content": norm_qwen_prompt},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            questions.append(prompt)
            prompts.append(text)
            users.append(user_id)
            blocks.append(block_id)
            labels.append(lb)

    # 初始化 VLLM 模型，指定 GPU
    #sampling_params = SamplingParams(temperature=1.0, top_p=0.9, repetition_penalty=1.05, max_tokens=512)
    sampling_param = SamplingParams(temperature=0,max_tokens=10,repetition_penalty=1.05,logprobs=5)
    vocab = tokenizer.get_vocab()
    yes_token_ids = [token_id for token, token_id in vocab.items() if ("yes" in token) or ("Yes" in token)]
    no_token_ids = [token_id for token, token_id in vocab.items() if ("no" in token) or ("No" in token)]
    # Input the model name or path. Can be GPTQ or AWQ models.
    # llm = LLM(model=model_path,dtype='half',trust_remote_code=True,tensor_parallel_size=2)
    llm = LLM(model=model_path,dtype='half',trust_remote_code=True)
    # 保存结果到文件
    # generated_texts = []
    res = {}

    with open(f'/PRM_point/amz/prediction_{thread_id}.json', 'w') as f:
        for i in tqdm(range(0,len(prompts),bs)):
            msg = prompts[i:min(i+bs,len(prompts))]
            ques = questions[i:min(i+bs,len(prompts))]
            u = users[i:min(i+bs,len(prompts))]
            b = blocks[i:min(i+bs,len(prompts))]
            l = labels[i:min(i+bs,len(prompts))]

            # 进行批量推理
            generated_texts = []
            probs = []
            outputs = llm.generate(msg, sampling_param)
            # for num in range(len(outputs)):
            #     generated_texts.append(outputs[num].outputs[0].text)
            # 处理输出结果
            for user, q,bb, ll, o in zip(u,ques,b,l,outputs):
                if True:
                    token_ids = o.outputs[0].token_ids
                    logprobs = o.outputs[0].logprobs
                    target_p = logprobs[0]
                    target = token_ids[0]
                    yes_prob , no_prob = 1e-9,1e-9
                    for i in range(len(logprobs)):
                        logprobs_record = logprobs[i]
                        if token_ids[i] in (9454, 2753):
                            if 9454 in logprobs_record:
                                yes_prob = math.exp(logprobs_record[9454].logprob)
                            else:
                                print("error: yes_id is missing!")
                            if 2753 in logprobs_record:
                                no_prob = math.exp(logprobs_record[2753].logprob)
                            else:
                                print("error: no_id is missing!")
                            break
                    # logit = yes_prob/(yes_prob+no_prob)
                    logit = math.exp(yes_prob)/(math.exp(yes_prob)+math.exp(no_prob))
                    print(o.outputs[0].text)
                if user not in res:
                    res[user] = {}
                if bb not in res[user]:
                    res[user][str(bb)] = []
                res[user][str(bb)].append([logit,ll])

                # quess.append(q)
        json.dump(res,f,ensure_ascii=False,indent=4)


if __name__ == "__main__":
    prompt_path = '/PRM_point/amz/prm_data.json'
    model_path = '/checkpoints/Qwen/Qwen2___5-7B-Instruct'

    with open(prompt_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    import torch
    ng = torch.cuda.device_count()

    ng = 8

    #batch_size = total_prompts // 8
    data_list = data

    # prm_data = random.sample(data_list,2000)

    print(len(data_list))
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
