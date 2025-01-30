import json
import os
import math

with open('/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m/block_len_50/sft_test_final_prm_data.json','r',encoding='utf-8') as f:
    data_list = json.load(f)

# 计算每份数据的大小
import torch
ng = torch.cuda.device_count()

total_prompts = len(data_list)
batch_size = total_prompts // ng

# 分割数据
batches = [data_list[i:i + batch_size] for i in range(0, total_prompts, batch_size)]

# 如果数据不能均匀分割，最后一组可能会有剩余的数据
if len(batches) > ng:
    batches[ng-1].extend(batches[ng])
    batches = batches[:ng]

import multiprocessing
from vllm import LLM, SamplingParams
# chat_template = '<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
norm_qwen_prompt = "You are a helpful assistant."
short_qwen_prompt = "You are a helpful assistant. You will provide very concise and helpful response."

def run_inference(model_path,batch, bs, thread_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{thread_id}'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True)
    # 初始化 VLLM 模型，指定 GPU
    sampling_params = SamplingParams(temperature=0.0, repetition_penalty=1.05, max_tokens=10,logprobs=5)
    vocab = tokenizer.get_vocab()
    yes_token_ids = [token_id for token, token_id in vocab.items() if ("yes" in token) or ("Yes" in token)]
    no_token_ids = [token_id for token, token_id in vocab.items() if ("no" in token) or ("No" in token)]

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_path,dtype='half',trust_remote_code=True)
    
    # 将 batch 中的 prompt 提取出来
    # prompts = [chat_template.format(item['question']) for item in batch]
    prompts = []
    questions = []
    for item in batch:
        instruction = item['instruction']
        input = item['input']
        output = item['output']
        messages = [
                {"role": "system", "content": norm_qwen_prompt},
                {"role": "user", "content": instruction +' '+ input}
            ]

        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        prompts.append(text)

    labels = [item['output'] for item in batch]
    
    # 保存结果到文件
    generated_texts = []
    with open(f'/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM/ml-1m_prm/llm_prm/untrain_test_{thread_id}.json', 'w') as f:
        for i in tqdm(range(0,len(labels),bs)):
            msg = prompts[i:min((i+bs),len(labels))]
            label = labels[i:min((i+bs),len(labels))]
            # 进行批量推理
            outputs = llm.generate(msg, sampling_params)

            for m,o,l in zip(msg,outputs,labels):
                # generated_text = output.outputs[0].text
                generated_text = []
                probs = []
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

                generated_texts.append({'prompt':m,'ans':o.outputs[0].text,'prob':logit,'label':l})
        
        json.dump(generated_texts,f,ensure_ascii=False,indent=4)
            # break

if __name__ == "__main__":
    processes = []
    # model_path = '/mmu_nlp_ssd/xiayu12/LLaMA-Factory/saves/Qwen2.5_7B_Instruct/full/sft/checkpoint-48'
    model_path = '/share/ad/xiayu12/Open-World-Knowledge-Augmented-Recommendation_Gang/checkpoints/Qwen/Qwen2___5-7B-Instruct'
    for i in range(ng):
        p = multiprocessing.Process(target=run_inference, args=(model_path,batches[i], 1000, i))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()