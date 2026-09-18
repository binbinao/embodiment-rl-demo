#!/usr/bin/env bash
# 具身智能 RL Demo — 一键启动脚本（环境激活 + 模型训练整合）
#
# 用途:
#   1) 一键完成「环境激活 → GPU 状态快照 → 模型训练 → GPU 状态快照」
#   2) 训练期间持续占用 GPU，便于 NVIDIA 软件监控测试
#      (nvidia-smi / DCGM / nvtop / 外部监控平台均可观测到持续的显存与算力占用)
#
# 用法:
#   bash run.sh flat      # 平地训练（~3 分钟，适合快速监控冒烟）
#   bash run.sh rough     # 粗糙地形训练（~1.5 小时，适合长时间监控）
#   bash run.sh flat --max_iterations 500   # 覆盖迭代次数
#   ITER_LOOP=3 bash run.sh flat            # 连续训练 3 轮（保持 GPU 长时间满载）
#
# 可选环境变量:
#   ITER_LOOP       连续训练轮数（默认 1；>1 时轮间 GPU 不停歇，用于持续压测监控）
#   NUM_ENVS        并行环境数（默认 4096）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 1. 环境激活 ----------
# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/activate.sh"

# ---------- 2. 参数解析 ----------
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

NUM_ENVS="${NUM_ENVS:-4096}"
ITER_LOOP="${ITER_LOOP:-1}"

cd "$ISAACLAB_PATH"

# ---------- 3. GPU 状态快照（训练前，监控基线） ----------
gpu_snapshot() {
  local tag="$1"
  echo ""
  echo "==================== NVIDIA GPU 状态 [$tag] ===================="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader 2>&1 || true
  else
    echo "[warn] nvidia-smi 不可用，跳过 GPU 快照"
  fi
  echo "================================================================"
  echo ""
}

gpu_snapshot "训练前"

# ---------- 4. 训练（ITER_LOOP 轮） ----------
echo "======================================================================"
echo "[run] 任务: $TASK"
echo "[run] 并行环境数: $NUM_ENVS"
echo "[run] 训练轮数: $ITER_LOOP"
echo "======================================================================"

for i in $(seq 1 "$ITER_LOOP"); do
  echo ""
  echo "---------------------- 训练轮次 $i/$ITER_LOOP ----------------------"
  python scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --headless \
    "$@"
done

# ---------- 5. GPU 状态快照（训练后） ----------
gpu_snapshot "训练后"

echo "[run] 完成。"
