#!/usr/bin/env bash
# 具身智能 RL Demo — 环境激活脚本
# 用法: source scripts/activate.sh
# 说明: 设置 Isaac Lab 环境所需的所有环境变量（路径硬编码为本机实际安装位置）

set -euo pipefail

# micromamba 根目录
export MAMBA_ROOT_PREFIX=/home/ubuntu/micromamba

# Isaac Lab Python 3.11 环境（等价于 conda activate isaaclab）
export CONDA_PREFIX=/home/ubuntu/micromamba/envs/isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"

# Isaac Sim EULA 接受标志（否则首次运行会交互式阻塞）
export OMNI_KIT_ACCEPT_EULA=Y
export ACCEPT_EULA=Y

# Isaac Lab 源码根目录
export ISAACLAB_PATH=/data/IsaacLab

echo "[activate] CONDA_PREFIX=$CONDA_PREFIX"
echo "[activate] ISAACLAB_PATH=$ISAACLAB_PATH"
echo "[activate] python=$(command -v python)"
