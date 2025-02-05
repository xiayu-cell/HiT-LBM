import json

path = '/data/ml-1m/proc_data/sequential_data.json'

with open(path,'r',encoding='utf-8') as f:
    data = json.load(f)

user_n = 0
hist_n = 0
user_hist = []
for k,v in data.items():
    user_n +=1
    hist_n += len(v[0])
    user_hist.append(len(v[0]))

print(user_n)
print(hist_n/user_n)
print(min(user_hist))
print(max(user_hist))

import matplotlib.pyplot as plt
import numpy as np
# 生成一个随机整数数组（示例）
np.random.seed(42)  # 固定随机种子，保证结果可复现
data = np.array(user_hist)  # 在20到2400之间生成1000个随机整数
# filtered_data = [x for x in data if x <= 500]
# print(len(filtered_data))
# # 绘制直方图
# plt.figure(figsize=(10, 6))
# plt.hist(filtered_data, bins=50, color='skyblue', edgecolor='black')  # bins表示分为多少段
# plt.title('Distribution of Integer Array (20-500)', fontsize=16)
# plt.xlabel('Value Range (20-500)', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.savefig('./book.jpg', format='jpg', dpi=300)
# plt.close()

# 定义区间边界
bins = np.linspace(50, 2550, 51)  # 50等距分组，范围是50到500

# 计算每个区间的出现次数
counts, _ = np.histogram(data, bins=bins)

# 打印结果
print("区间边界:", bins)
print("每个区间的出现次数:", counts)

# 提取你关心的区间（50-100, 100-150, 150-200）
target_intervals = [(50, 100), (100, 150), (150, 200)]
for i, (start, end) in enumerate(target_intervals):
    index = np.where((bins[:-1] >= start) & (bins[:-1] < end))[0][0]
    print(f"区间 {start}-{end} 的出现次数: {counts[index]}")
