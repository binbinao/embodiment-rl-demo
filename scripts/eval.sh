#!/usr/bin/env bash
# 具身智能 RL Demo — 评估脚本（加载 checkpoint 录评测视频）
# 用法:
#   bash scripts/eval.sh rough /path/to/model_1499.pt    # 指定 checkpoint 录视频
#   bash scripts/eval.sh rough                           # 自动用已归档的 artifacts/rough/model_1499.pt
#
# 可选环境变量:
#   NUM_ENVS=50     并行环境数（默认 50）
#   VIDEO_LENGTH=500  视频帧数（默认 500）

set -euo pipefail

# 激活环境
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/activate.sh"

# 参数解析
TASK_TYPE="${1:-rough}"
CHECKPOINT="${2:-}"

case "$TASK_TYPE" in
  flat)
    TASK="Isaac-Velocity-Flat-Unitree-Go2-Play-v0"
    DEFAULT_CKPT="$SCRIPT_DIR/../artifacts/flat/model_299.pt"
    ;;
  rough)
    TASK="Isaac-Velocity-Rough-Unitree-Go2-Play-v0"
    DEFAULT_CKPT="$SCRIPT_DIR/../artifacts/rough/model_1499.pt"
    ;;
  *)
    echo "错误: 未知任务类型 '$TASK_TYPE'，可选: flat | rough" >&2
    exit 1
    ;;
esac

# checkpoint 解析：显式指定 > 归档产物
if [ -z "$CHECKPOINT" ]; then
  CHECKPOINT="$DEFAULT_CKPT"
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "错误: checkpoint 不存在: $CHECKPOINT" >&2
  exit 1
fi

NUM_ENVS="${NUM_ENVS:-50}"
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"

cd "$ISAACLAB_PATH"

echo "[eval] 任务: $TASK"
echo "[eval] checkpoint: $CHECKPOINT"
echo "[eval] 并行环境数: $NUM_ENVS, 视频帧数: $VIDEO_LENGTH"

python scripts/reinforcement_learning/rsl_rl/play.py \
  --task "$TASK" \
  --num_envs "$NUM_ENVS" \
  --video --video_length "$VIDEO_LENGTH" \
  --checkpoint "$CHECKPOINT" \
  --headless
