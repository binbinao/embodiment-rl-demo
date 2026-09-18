# 具身智能 RL Demo

宇树 Go2 四足机器人 · 在线强化学习（PPO）· 速度跟踪任务

在物理模拟器中从零训练一个四足机器人学会稳定行走并跟踪速度指令。基于 NVIDIA Isaac Lab + Isaac Sim，零自研算法代码，全流程可复现。

## 成果一览

| 阶段 | 任务 | 迭代 | 耗时 | 速度跟踪奖励 `track_lin_vel_xy_exp` |
|---|---|---|---|---|
| 平地 | Flat-Unitree-Go2 | 300 | ~3 分钟 | 0.004 → **1.432** |
| 粗糙地形 | Rough-Unitree-Go2 | 1500 | ~1.5 小时 | 0.005 → **1.228** |

最终策略在粗糙地形课程（terrain_level 5.4）上稳定站立 + 按指令行走，存活率 ~88%。

## 快速开始（评估已有策略）

无需重新训练，直接加载已归档的 checkpoint 录评测视频：

```bash
# 1. 激活环境
source scripts/activate.sh

# 2. 评估粗糙地形策略（自动用 artifacts/rough/model_1499.pt）
bash scripts/eval.sh rough
```

视频输出到 Isaac Lab 的日志目录（`$ISAACLAB_PATH/logs/rsl_rl/unitree_go2_rough/.../videos/play/`）。

也可以手动播放已归档的视频：

```bash
# 用任意播放器打开
open artifacts/video/rl-video-step-0.mp4   # macOS
xdg-open artifacts/video/rl-video-step-0.mp4  # Linux
```

## 重新训练

```bash
source scripts/activate.sh

# 平地（快速，~3 分钟）
bash scripts/train.sh flat

# 粗糙地形（完整，~1.5 小时）
bash scripts/train.sh rough

# 覆盖迭代次数
bash scripts/train.sh flat --max_iterations 500
```

训练产物默认写入 `$ISAACLAB_PATH/logs/rsl_rl/`（TensorBoard 曲线 + checkpoint）。

## 查看训练曲线

```bash
source scripts/activate.sh
tensorboard --logdir $ISAACLAB_PATH/logs/rsl_rl
```

浏览器打开 `http://localhost:6006`，查看速度跟踪奖励、跟踪误差、存活率等指标。

## 目录结构

```
.
├── TRAINING_LOG.md                  # 完整训练记录（环境搭建/踩坑/收敛指标）
├── scripts/
│   ├── activate.sh                  # 环境激活（环境变量 + EULA）
│   ├── train.sh                     # 训练（flat/rough）
│   └── eval.sh                      # 评估 + 录视频
└── artifacts/
    ├── flat/model_299.pt            # 平地策略 checkpoint
    ├── rough/model_1499.pt          # 粗糙地形策略 checkpoint
    └── video/rl-video-step-0.mp4    # 评测视频
```

## 依赖环境

| 组件 | 版本 | 说明 |
|---|---|---|
| Isaac Lab | v2.3.2 | 源码位于 `/data/IsaacLab` |
| Isaac Sim | 5.1.0 | pip 包 |
| PyTorch | 2.7.0+cu128 | |
| rsl-rl-lib | 3.1.2 | PPO 实现 |
| Python | 3.11 | 隔离于 `~/micromamba/envs/isaaclab` |
| 硬件 | NVIDIA L20 48GB | Vulkan 渲染 |

> 环境已在本机装好，`scripts/activate.sh` 会设置所有必需的环境变量。
> 如需从零重建环境，完整步骤见 [TRAINING_LOG.md](./TRAINING_LOG.md) 第 5 节。

## 切换其他机器人

Isaac Lab 内置了多种足式机器人任务，改 `scripts/train.sh` 中的 `--task` 即可：

| 机器人 | 任务 registry id |
|---|---|
| 宇树 Go2 四足 | `Isaac-Velocity-Rough-Unitree-Go2-v0` |
| ANYmal-C 四足 | `Isaac-Velocity-Rough-Anymal-C-v0` |
| 宇树 H1 双足 | `Isaac-Velocity-Rough-Unitree-H1-v0` |
| 宇树 G1 双足 | `Isaac-Velocity-Rough-Unitree-G1-v0` |

> 双足人形机器人难度显著更高，需要更多迭代与显存。

## 验收指标说明

足式运动没有 0/1 "成功率"，业界标准是指令跟踪误差：

- `track_lin_vel_xy_exp` — 水平速度指令跟踪奖励（**越大越好**）
- `track_ang_vel_z_exp` — 转向角速度指令跟踪奖励（越大越好）
- `error_vel_xy` — 水平速度跟踪误差（**越小越好**）
- `error_vel_yaw` — 转向角速度跟踪误差（越小越好）
- `time_out`（存活率）— 机器人不跌倒跑满整个 episode 的比例（越大越好）

## 更多信息

完整的训练过程、踩坑记录、网络约束、依赖冲突处理，见 [TRAINING_LOG.md](./TRAINING_LOG.md)。
