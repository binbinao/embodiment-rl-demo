#!/usr/bin/env bash
# 具身智能 RL Demo — 训练脚本（宇树 Go2 速度跟踪）
# 用法:
#   bash scripts/train.sh flat    # 平地训练（300 iter，约 3 分钟）
#   bash scripts/train.sh rough   # 粗糙地形训练（1500 iter，约 1.5 小时）
#   bash scripts/train.sh flat --max_iterations 500   # 覆盖迭代次数

set -euo pipefail

# 激活环境
source "$(dirname "$0")/activate.sh"

# 参数解析
TASK_TYPE="${1:-rough}"
shift || true

case "$TASK_TYPE" in
  flat)
    TASK="Isaac-Velocity-Flat-Unitree-Go2-v0"
    ;;
  rough)
    TASK="Isaac-Velocity-Rough-Unitree-Go2-v0"
    ;;
  *)
    echo "错误: 未知任务类型 '$TASK_TYPE'，可选: flat | rough" >&2
    exit 1
    ;;
esac

cd "$ISAACLAB_PATH"

echo "[train] 任务: $TASK"
echo "[train] 并行环境数: 4096"

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" \
  --num_envs 4096 \
  --headless \
  "$@"
