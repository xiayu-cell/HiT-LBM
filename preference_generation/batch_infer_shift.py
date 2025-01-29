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

def run_inference(model_path,block_idx,batch, bs, thread_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{thread_id}'
    prompts = []
    questions = []
    users = []
    block_ids = []
    pre_sum = []
    cur_sum = []
    sum_prompts = []
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True)

    # block_id = item['block_id']
    for item in batch:
        user = item['user']
        sum_prompt = item['pre_interest']
        pre_interest = item['pre_interest']
        cur_interest = item['cur_interest']
        shift_prompt = item['shift_prompt']
        sum_prompt = item['sum_prompt']
        # PRIVIOUS_INTEREST = '[Previous Interest Summary]'
        # CURRENT_HIST = '[Current Movie Viewing Data]'
        prompt = 'User previous explanations: [{}], User current explanations: [{}]. Are there any new preferences in user current explanations that differs from user previous explanations? If yes, list these preferences on movies(consider factors like genre, author, writing style, theme, setting, length and complexity, time period, literary quality, critical acclaim (Provide clear explanations based on relevant details from the user\'s book viewing history and other pertinent factors.'.format(pre_interest,cur_interest)
        # print(prompt)
        messages = [
                {"role": "system", "content": norm_qwen_prompt},
                {"role": "user", "content": prompt}
            ]

        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        questions.append(prompt)
        prompts.append(text)
        users.append(user)
        pre_sum.append(pre_interest)
        cur_sum.append(cur_interest)
        sum_prompts.append(sum_prompt)

    # 初始化 VLLM 模型，指定 GPU
    #sampling_params = SamplingParams(temperature=1.0, top_p=0.9, repetition_penalty=1.05, max_tokens=512)
    sampling_param = SamplingParams(temperature=0.5,max_tokens=4096,repetition_penalty=1.05)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_path,dtype='half',trust_remote_code=True)
    # 保存结果到文件
    # generated_texts = []
    res = {}

    with open(f'/mmu_nlp_hdd/xiayu12/LIBER_ours/preference_generation/amz/shift/shift_{block_idx}_{thread_id}.json', 'w') as f:
        for i in tqdm(range(0,len(prompts),bs)):
            msg = prompts[i:min(i+bs,len(prompts))]
            ques = questions[i:min(i+bs,len(prompts))]
            u = users[i:min(i+bs,len(prompts))]
            pre = pre_sum[i:min(i+bs,len(prompts))]
            cur = cur_sum[i:min(i+bs,len(prompts))]
            sum_pro = sum_prompts[i:min(i+bs,len(prompts))]

            # 进行批量推理
            generated_texts = []
            outputs = llm.generate(msg, sampling_param)
            for num in range(len(outputs)):
                generated_texts.append(outputs[num].outputs[0].text)
            # 处理输出结果
            quess = []
            for user, q,pre_interest,cur_interest, sp, output in zip(u,ques,pre,cur,sum_pro,generated_texts):
                res[user] = {}
                res[user][str(block_idx)] = {
                    'sum_prompt': sp,
                    'pre_sum': pre_interest,
                    'cur_sum': cur_interest,
                    'shift': output
                }
                quess.append(q)
        json.dump(res,f,ensure_ascii=False,indent=4)
        with open(f'/mmu_nlp_hdd/xiayu12/LIBER_ours/preference_generation/amz/shift/question_shift_{block_idx}_{thread_id}.json', 'w') as file:
            json.dump(quess,file,ensure_ascii=False,indent=4)

            # break

if __name__ == "__main__":
    prompt_path = '/mmu_nlp_hdd/xiayu12/LIBER/data/amz/proc_data/all_prompt.hist'
    model_path = '/share/ad/xiayu12/Open-World-Knowledge-Augmented-Recommendation_Gang/checkpoints/Qwen/Qwen2___5-7B-Instruct'

    # with open(prompt_path,'r',encoding='utf-8') as f:
    #     data = json.load(f)
    import torch
    ng = torch.cuda.device_count()

    ng = 8

    #batch_size = total_prompts // 8
    block_interest_folder = '/mmu_nlp_hdd/xiayu12/LIBER_ours/preference_generation/amz/summary'
    all_summary = {}
    for filename in os.listdir(block_interest_folder):
        if filename.startswith(f'summary') and filename.endswith(".json"):
            file_path = os.path.join(block_interest_folder, filename)
            # 读取 JSON 文件
            with open(file_path, 'r') as file:
                # pdb.set_trace()
                t = json.load(file)
                for user,blocks in t.items():
                    if user in all_summary:
                        all_summary[user].update(t[user])
                    else:
                        all_summary[user] = t[user]
    for block_id in range(1,100):
        data_list = []
        for user, blocks in all_summary.items():
            
            if str(block_id) in blocks:
                if block_id == 0:
                    break
                else:
                    # pdb.set_trace()
                    pre_interest = all_summary[user][str(block_id-1)]['summary']
                    shift_prompt = all_summary[user][str(block_id-1)]['shift_prompt']
                    cur_interest = all_summary[user][str(block_id)]['summary']
                    sum_prompt = all_summary[user][str(block_id)]['prompt']
                data_list.append({
                    'user':user,
                    'block_id': str(block_id),
                    'pre_interest': pre_interest,
                    'cur_interest': cur_interest,
                    'shift_prompt': shift_prompt,
                    'sum_prompt': sum_prompt
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