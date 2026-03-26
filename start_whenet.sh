#!/bin/bash
# 清掉残留进程
pkill -f train_whenet 2>/dev/null
sleep 2

cd ~/codes/head-pose-estimation
source venv/bin/activate

nohup python train_whenet.py \
    --data data \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-4 \
    --milestones 40 70 \
    --gamma 0.1 \
    --num-workers 4 \
    --save-path weights \
    > train_whenet.log 2>&1 &

echo "PID=$!"
disown
