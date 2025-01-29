#!/bin/bash

# 模型名称列表
models=(
"DIEN" 
"DIN" 
"DCNv2"
"DeepFM"
"xDeepFM"
"AutoInt"
"FiBiNet"
"FiGNN"
)

# 初始化gpuid
gpuid=0

# 循环遍历模型名称
for model in "${models[@]}"; do
    # 运行Python脚本，并将输出重定向到日志文件
    nohup python /mmu_nlp_ssd/xiayu12/LIBER_ours_train/RS/run_ctr_xiayu_ml.py --gpuid $gpuid --model_name "$model" > /mmu_vcg2_wjc_ssd/xiayu12/LIBER_ours_train/RS/MLP/ml/"${model}_MLP_bge_5.log" &
    
    # 打印运行信息
    echo "Running model $model with gpuid $gpuid, log saved to ${model}_final_prm_bge_5.log"
    
    # gpuid递增
    gpuid=$((gpuid + 1))
done