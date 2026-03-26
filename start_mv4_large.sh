#!/bin/bash
# 启动 MobileNetV4-Large pretrained 训练

cd ~/codes/head-pose-estimation
source venv/bin/activate

# 清理残留进程
pkill -f "main.py.*mobilenetv4_large" 2>/dev/null
sleep 2

nohup python main.py \
    --data data \
    --network mobilenetv4_large_pretrained \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.0001 \
    --lr-scheduler MultiStepLR \
    --milestones 40 70 \
    --gamma 0.1 \
    --num-workers 4 \
    --save-path weights \
    > train_mobilenetv4_large_pretrained.log 2>&1 &

echo "PID=$!"
disown
