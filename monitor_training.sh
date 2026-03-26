#!/bin/bash
LOG_DIR="$HOME/codes/head-pose-estimation"
MSG_FILE="/tmp/training_status.txt"

# 检查训练是否还在运行
MV4_PRE=$(ps aux | grep "mobilenetv4_small_pretrained" | grep -v grep | wc -l)
MV2=$(ps aux | grep "mobilenetv2" | grep -v grep | grep main.py | wc -l)

if [ "$MV4_PRE" -eq 0 ] && [ "$MV2" -eq 0 ]; then
  echo "所有训练已完成" > "$MSG_FILE"
  exit 0
fi

# 收集状态
{
  echo "📊 训练进度报告 ($(date +%H:%M))"
  echo ""
  
  if [ "$MV4_PRE" -gt 0 ]; then
    echo "=== MobileNetV4 有预训练 ==="
    tail -100 "$LOG_DIR/train_mv4_small_pretrained.log" | grep -E "Epoch.*Summary|Yaw:.*MAE|Best mean" | tail -3
    echo ""
  fi
  
  if [ "$MV2" -gt 0 ]; then
    echo "=== MobileNetV2 (论文复现) ==="
    tail -100 "$LOG_DIR/train_mobilenetv2.log" | grep -E "Epoch.*Summary|Yaw:.*MAE|Best mean" | tail -3
  fi
} > "$MSG_FILE"

# 通过 OpenClaw 发送通知（如果有 openclaw CLI）
if command -v openclaw &> /dev/null; then
  openclaw notify --message "$(cat $MSG_FILE)" 2>/dev/null || true
fi

cat "$MSG_FILE"
