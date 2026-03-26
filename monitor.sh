#!/bin/bash
while true; do
  LATEST=$(tail -20 ~/codes/head-pose-estimation/train_mv4_small.log | grep -E "Epoch.*Summary|Yaw:" | tail -3)
  if [ -n "$LATEST" ]; then
    echo "[$(date +%H:%M)] $LATEST"
  fi
  sleep 600
done
