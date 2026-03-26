#!/bin/bash
LOG_FILE=~/codes/head-pose-estimation/train_mobilenetv4_medium_pretrained.log

# 获取最新信息
LATEST=$(tail -100 "$LOG_FILE" | grep -E 'Epoch: \[|Yaw:|completed\.' | tail -15)
CURRENT_EPOCH=$(echo $LATEST | grep -oP 'Epoch: \[\K[0-9]+' | tail -1)
BEST_MAE=$(echo $LATEST | grep "Best mean angular error" | tail -1 | grep -oP 'error:\s+\K[0-9.]+')
GPU_STATUS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits)

if [ -n "" ]; then
  echo "📊 **MobileNetV4 Medium 训练进度**"
  echo ""
  echo "Epoch: ${CURRENT_EPOCH:-?}/80"
  [ -n "$BEST_MAE" ] && echo "Best MAE: **${BEST_MAE}°** 🎯"
  echo ""
  echo "GPU: ${GPU_STATUS}"
else
  echo "训练尚未开始或日志文件不存在"
fi
