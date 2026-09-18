#!/usr/bin/env bash
# 具身智能 RL Demo — Profiling 脚本（Nsight Systems / Nsight Compute 封装）
#
# 用法:
#   bash scripts/profile.sh nsys  [flat|rough] [--train-arg ...]   # 全局时间线分析
#   bash scripts/profile.sh ncu   [flat|rough] [--train-arg ...]   # kernel 级深挖（过滤 gemm）
#   bash scripts/profile.sh ncu-names [flat|rough]                 # 先看 kernel 名字再精确过滤
#
# 说明:
#   - profiler 包裹在 run.sh 外层，run.sh 无需改动
#   - 输出 .nsys-rep / .ncu-rep 报告到 /tmp/，拷回本地 GUI 打开
#
# 可选环境变量:
#   OUT_DIR        报告输出目录（默认 /tmp）
#   LAUNCH_SKIP    跳过的启动 kernel 数（ncu 用，默认 200）
#   LAUNCH_COUNT   采样 kernel 数（ncu 用，默认 10）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 参数解析 ----------
MODE="${1:-}"
TASK_TYPE="${2:-flat}"
shift 2 || true

OUT_DIR="${OUT_DIR:-/tmp}"
LAUNCH_SKIP="${LAUNCH_SKIP:-200}"
LAUNCH_COUNT="${LAUNCH_COUNT:-10}"

case "$MODE" in
  nsys)      : ;;
  ncu)       : ;;
  ncu-names) : ;;
  *)
    echo "错误: 未知模式 '$MODE'，可选: nsys | ncu | ncu-names" >&2
    echo "用法: bash scripts/profile.sh <nsys|ncu|ncu-names> [flat|rough]" >&2
    exit 1
    ;;
esac

case "$TASK_TYPE" in
  flat|rough) : ;;
  *)
    echo "错误: 未知任务类型 '$TASK_TYPE'，可选: flat | rough" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

cd "$(dirname "$SCRIPT_DIR")"   # 回到项目根目录，run.sh 在此

case "$MODE" in
  nsys)
    echo "[profile/nsys] 全局时间线分析 (任务: $TASK_TYPE)"
    echo "[profile/nsys] 报告: $OUT_DIR/go2_${TASK_TYPE}_${TS}.nsys-rep"
    nsys profile \
      --trace=cuda,nvtx,osrt \
      --output="$OUT_DIR/go2_${TASK_TYPE}_${TS}" \
      --force-overwrite=true \
      bash run.sh "$TASK_TYPE" "$@"
    echo "[profile/nsys] 完成。查看摘要: nsys stats $OUT_DIR/go2_${TASK_TYPE}_${TS}.nsys-rep"
    ;;

  ncu)
    echo "[profile/ncu] kernel 级深挖 PPO 学习 (任务: $TASK_TYPE)"
    echo "[profile/ncu] 报告: $OUT_DIR/go2_${TASK_TYPE}_${TS}.ncu-rep"
    ncu \
      --launch-skip "$LAUNCH_SKIP" \
      --launch-count "$LAUNCH_COUNT" \
      --set full \
      --export "$OUT_DIR/go2_${TASK_TYPE}_${TS}.ncu-rep" \
      --kernel-name "regex:.*(gemm|mma|elementwise|softmax|reduce).*" \
      bash run.sh "$TASK_TYPE" "$@"
    echo "[profile/ncu] 完成。拷回本地用 Nsight Compute GUI 打开: $OUT_DIR/go2_${TASK_TYPE}_${TS}.ncu-rep"
    ;;

  ncu-names)
    echo "[profile/ncu-names] 先看 kernel 名字（任务: $TASK_TYPE）"
    ncu \
      --launch-skip "$LAUNCH_SKIP" \
      --launch-count 5 \
      --set basic \
      bash run.sh "$TASK_TYPE" "$@"
    echo "[profile/ncu-names] 完成。从上方输出找 'Kernel Name' 列，再用 ncu 模式 + --kernel-name 精确过滤。"
    ;;
esac
