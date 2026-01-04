#!/bin/bash

# 设置GPU
export CUDA_VISIBLE_DEVICES=1

# 激活环境
source /mnt/shared-storage-user/tangjia/miniconda3/bin/activate eff

# 切换到工作目录
cd /mnt/shared-storage-user/tangjia/eff/LightZero

# 运行 Pong
# echo "========================================"
# echo "Starting Pong training..."
# echo "========================================"
# python zoo/atari/config/atari_unizero_segment_config.py --env PongNoFrameskip-v4 --seed 0

# 运行 Qbert
echo "========================================"
echo "Starting Qbert training..."
echo "========================================"
python zoo/atari/config/atari_unizero_segment_config.py --env QbertNoFrameskip-v4 --seed 0

echo "========================================"
echo "All training completed!"
echo "========================================"
