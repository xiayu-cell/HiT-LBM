import tiktoken
import json
# 初始化分词器（例如，使用 GPT-4 的分词器）
encoding = tiktoken.encoding_for_model("gpt-4")

# 示例 prompts
prompts = [

]

with open('/mmu_nlp_ssd/xiayu12/LIBER_ours_train/PRM_point/ml-1m/prm_data.json','r',encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    cur_prompt = item['cur_prompt']
    # user_hist = cur_prompt.split('[Current Movie Viewing History]')[-1].split('\n')[1]
    # print(user_hist)
    prompts.append(cur_prompt)

# 计算每个 prompt 的 token 长度
token_lengths = [len(encoding.encode(prompt)) for prompt in prompts]

# 打印每个 prompt 的 token 长度
for i, (prompt, length) in enumerate(zip(prompts, token_lengths)):
    print(f"Prompt {i+1}: {prompt}\nToken Length: {length}\n")

# 计算平均 token 长度
average_token_length = sum(token_lengths) / len(token_lengths)
print(f"Average Token Length: {average_token_length}")