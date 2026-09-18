# 具身智能强化学习 Demo — 完整训练记录

> 宇树 Go2 四足机器人 · 在线强化学习（PPO）· 速度跟踪任务
> 执行日期：2026-09-18　|　执行环境：NVIDIA L20 48GB + AMD EPYC 96 核 + Ubuntu 24.04

---

## 1. 项目目标

为具身智能强化学习场景搭建一个可运行、可复现的 demo：

1. 在本机搭建 RL 训练环境（Isaac Sim + Isaac Lab）
2. 训练一个四足机器人（宇树 Go2）的速度跟踪策略
3. 产出策略权重（checkpoint）+ 评测视频 + 收敛指标

---

## 2. 需求澄清结论（苏格拉底式问答）

| 维度 | 最终决策 | 说明 |
|---|---|---|
| 任务域 | 足式/双足运动 | 最终选四足（宇树 Go2） |
| 训练范式 | **在线 RL** | 从零开始，不依赖 HF 预训练模型与外部数据集 |
| 验收标准 | 速度跟踪误差 + 存活率 + 评测视频 | 足式运动无 0/1 成功率，业界标准是指令跟踪误差 |
| 时间预算 | 半天~过夜 | 实际仅用 ~2 小时 |
| 经验水平 | 会用 PyTorch，RL 不熟 | 采用官方默认超参，零自研算法代码 |

> **关键澄清**：用户最初提出的"从 HuggingFace 找模型 + 找公共训练数据集"与最终选定的"在线 RL"存在矛盾。
> 在线 RL 不需要预训练模型，也不需要训练数据集。经确认后按最经典路线执行：
> **纯在线 RL（PPO），agent 在物理模拟器中靠奖励函数自主探索学习。**

---

## 3. 硬件与环境探测（实测）

| 项 | 实测值 | 结论 |
|---|---|---|
| GPU | NVIDIA L20 48GB（Ada, compute 8.9） | 足式 RL 完全够用 |
| 驱动 / CUDA | 驱动 580.126.20 / CUDA runtime 13.0 / nvcc 12.8 | 需配 torch cu128 |
| 内存 | 92 GB | 充足 |
| CPU | AMD EPYC 9K84 96 核（cgroup 限 32） | 充足 |
| 磁盘 | `/data` 478 GB 空闲 | Isaac Lab + 资产放这里 |
| GLIBC | 2.39 | 满足 Isaac Sim pip 要求（≥2.35） |
| Vulkan | NVIDIA ICD 在位 | headless 离线渲染（录视频）可用 |
| Python | 系统 3.12.3（无 pip、无 conda、无 uv） | ⚠️ 需装 micromamba 建 3.11 隔离环境 |

### 网络连通性（关键约束）

- ✅ 可达：`raw.githubusercontent.com`、`api.github.com`、`codeload.github.com`、`pypi.org`、`pypi.nvidia.com`、`download.pytorch.org`、`conda.anaconda.org`
- ❌ 不可达（超时）：`github.com` 主站、git 协议（`git ls-remote` 超时）
- 影响：Isaac Lab 源码无法 `git clone`，改用 `codeload.github.com` tarball 绕过

---

## 4. 软件版本选型（联网核实，非记忆）

| 组件 | 锁定版本 | 依据 |
|---|---|---|
| Isaac Lab | v2.3.2（稳定版） | v3.0.0 仍在 beta/EA |
| Isaac Sim | 5.1.0（pip 包） | 官方 v2.3.2 文档 pin |
| Python | 3.11 | Isaac Sim 5.x 硬要求 |
| PyTorch | 2.7.0 + torchvision 0.22.0, cu128 | 官方 pin |
| RSL-RL | rsl-rl-lib 3.1.2 | train.py 校验 ≥3.0.1 |
| micromamba | 2.9.0 | 环境管理器 |

---

## 5. 环境搭建（完整步骤）

### 5.1 安装 micromamba

```bash
# 下载安装脚本（github.com 主站不可达，直接下 release 二进制）
curl -fsSL -o ~/micromamba/bin/micromamba \
  "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"
chmod +x ~/micromamba/bin/micromamba
export MAMBA_ROOT_PREFIX=~/micromamba
```

### 5.2 创建 Python 3.11 环境

```bash
~/micromamba/bin/micromamba create -n isaaclab -c conda-forge python=3.11 -y
```

### 5.3 安装 PyTorch（cu128）

```bash
~/micromamba/bin/micromamba run -n isaaclab python -m pip install \
  torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### 5.4 安装 Isaac Sim（pip 包）

```bash
~/micromamba/bin/micromamba run -n isaaclab python -m pip install \
  "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### 5.5 下载 Isaac Lab 源码（tarball 绕过 git）

```bash
cd /data
curl -fsSL -o IsaacLab-v2.3.2.tar.gz \
  "https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/refs/tags/v2.3.2"
tar xzf IsaacLab-v2.3.2.tar.gz
mv IsaacLab-2.3.2 IsaacLab
rm IsaacLab-v2.3.2.tar.gz
```

### 5.6 安装 Isaac Lab 扩展 + rsl_rl

```bash
# 前置：cmake + build-essential
sudo apt-get install -y --no-install-recommends cmake build-essential

# 激活环境上下文，运行官方安装脚本
export CONDA_PREFIX=/home/ubuntu/micromamba/envs/isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=Y
cd /data/IsaacLab
./isaaclab.sh --install rsl_rl
```

---

## 6. 踩坑记录（环境搭建）

| # | 问题 | 根因 | 解决方案 |
|---|---|---|---|
| 1 | `github.com` 主站 + git 协议超时 | 网络限制 | 用 `codeload.github.com` tarball 下载源码 |
| 2 | `isaaclab.sh` 报 `tabs: terminal type 'dumb'` | 脚本 `tabs 4` 需真实终端 | 用 `pty: true` 运行 |
| 3 | `flatdict==4.0.1` 构建失败：`ModuleNotFoundError: pkg_resources` | setuptools 84 移除了 `pkg_resources` | 降级 `setuptools<81`，再 `--no-build-isolation` 单独装 flatdict |
| 4 | 核心 `isaaclab` 包未安装（只装了 assets/contrib/mimic/rl/tasks） | 官方脚本首次未装上核心包 | 手动 `pip install --editable . --no-build-isolation` |
| 5 | 裸 `import pxr` 报 `ModuleNotFoundError` | **预期行为**，pxr 由 Isaac Sim 内核在 AppLauncher 启动后注入 sys.path | 无需处理，用 `create_empty.py` 验证模拟器启动 |
| 6 | Isaac Sim 首次运行要求接受 EULA | 交互式提示阻塞 | 设 `OMNI_KIT_ACCEPT_EULA=Y` 环境变量 |

### 已知无害的依赖冲突

- `wheel 0.48.0 requires packaging>=24.0, but you have packaging 23.0`（Isaac Sim 强降级 packaging）
- `fastapi requires starlette<0.46.0, but you have starlette 0.49.1`（Isaac Sim 强 pin starlette 0.49.1）

两者均为 Isaac Sim 主动 pin 导致，不影响训练运行。

---

## 7. 环境验证

### 7.1 Isaac Sim 内核启动验证

```bash
export CONDA_PREFIX=/home/ubuntu/micromamba/envs/isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=Y
cd /data/IsaacLab
python scripts/tutorials/00_sim/create_empty.py --headless
```

**验证通过证据**（日志输出）：
- Vulkan 识别 L20 48GB
- PhysX 插件就绪
- `[INFO]: Setup complete...`
- 脚本常驻运行（不退出，属预期，`create_empty.py` 启动模拟器后等待）

---

## 8. 正式训练

### 8.1 冒烟训练（3 迭代）

```bash
cd /data/IsaacLab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 64 --max_iterations 3 --headless
```

结果：`exit=0`，`4097 steps/s`，全链路通。

### 8.2 阶段 A — 平地训练（300 迭代）

```bash
cd /data/IsaacLab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 4096 --headless
```

| 项 | 值 |
|---|---|
| 耗时 | 3 分 3 秒（167.72s） |
| 吞吐 | 180094 steps/s |
| 并行环境数 | 4096 |
| 最终迭代 | 299/300 |
| checkpoint | `model_299.pt` |

### 8.3 阶段 B — 粗糙地形训练（1500 迭代）

```bash
cd /data/IsaacLab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --num_envs 4096 --headless
```

| 项 | 值 |
|---|---|
| 耗时 | 1 小时 31 分钟 |
| 吞吐 | ~60k steps/s |
| 并行环境数 | 4096 |
| 地形课程 | 推进到 terrain_level 5.6 |
| 最终迭代 | 1499/1500 |
| checkpoint | `model_1499.pt` |

### 8.4 PPO 超参（官方默认，未改动）

来源：`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py`

| 超参 | 值 |
|---|---|
| num_steps_per_env | 24 |
| num_learning_epochs | 5 |
| num_mini_batches | 4 |
| learning_rate | 1.0e-3 |
| max_iterations（rough） | 1500 |
| max_iterations（flat） | 300 |
| save_interval | 50 |
| experiment_name | unitree_go2_rough / unitree_go2_flat |

### 8.5 策略网络结构（模型加载时输出）

```
Actor MLP:  Linear(235→512) → ELU → Linear(512→256) → ELU
            → Linear(256→128) → ELU → Linear(128→12)
Critic MLP: Linear(235→512) → ELU → Linear(512→256) → ELU
            → Linear(256→128) → ELU → Linear(128→1)
```

- 输入 235 维（观测：本体状态 + 关节 + 指令 + 高度扫描）
- 输出 12 维（Go2 12 个关节自由度）

---

## 9. 训练收敛结果

### 阶段 A（平地）

| 指标 | iter 0 | iter 150 | iter 299 |
|---|---|---|---|
| `track_lin_vel_xy_exp`（奖励↑） | 0.004 | 1.200 | **1.432** |
| `error_vel_xy`（误差↓） | 0.018 | 0.422 | **0.178** |
| `error_vel_yaw`（误差↓） | 0.019 | 0.496 | **0.340** |
| 存活率 time_out | — | — | **0.9975** |

### 阶段 B（粗糙地形）

| 指标 | iter 0 | iter 750 | iter 1499 |
|---|---|---|---|
| `track_lin_vel_xy_exp`（奖励↑） | 0.005 | 1.078 | **1.228** |
| `error_vel_xy`（误差↓） | 0.015 | 0.473 | **0.353** |
| `error_vel_yaw`（误差↓） | 0.017 | 0.492 | **0.416** |
| 存活率 time_out | ~0.03 | — | **0.916** |

**结论**：机器人在粗糙地形课程（难度逐步增加）上学会了稳定站立 + 按指令速度行走。
误差曲线先升后降是正常的——curriculum 会逐步加大指令幅度，中段波动后回归收敛。

---

## 10. 评估与评测视频

### 10.1 评测命令

```bash
cd /data/IsaacLab
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
  --num_envs 50 \
  --video --video_length 500 \
  --checkpoint /data/IsaacLab/logs/rsl_rl/unitree_go2_rough/2026-09-18_09-22-18/model_1499.pt \
  --headless
```

### 10.2 评测视频元数据

| 项 | 值 |
|---|---|
| 路径 | `/data/IsaacLab/logs/rsl_rl/unitree_go2_rough/2026-09-18_09-22-18/videos/play/rl-video-step-0.mp4` |
| 时长 | 9.98 秒 |
| 帧数 | 499 帧 |
| 分辨率 | 1280×720 |
| 帧率 | 50 fps |
| 编码 | h264 |
| 大小 | 1.4 MB |

> 使用 PLAY 环境（`Isaac-Velocity-Rough-Unitree-Go2-Play-v0`）：50 个环境、无随机扰动、无推挤事件，用于干净评测。

---

## 11. 交付物清单

| 产物 | 路径 | 大小 |
|---|---|---|
| 平地最终 checkpoint | `/data/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-09-18_09-18-52/model_299.pt` | 983 KB |
| 粗糙地形最终 checkpoint | `/data/IsaacLab/logs/rsl_rl/unitree_go2_rough/2026-09-18_09-22-18/model_1499.pt` | 6.9 MB |
| 评测视频 | `/data/IsaacLab/logs/rsl_rl/unitree_go2_rough/2026-09-18_09-22-18/videos/play/rl-video-step-0.mp4` | 1.4 MB |
| TensorBoard 事件 | `/data/IsaacLab/logs/rsl_rl/unitree_go2_{flat,rough}/**/events.out.tfevents.*` | — |

---

## 12. 复现命令速查

```bash
# ===== 环境激活 =====
export MAMBA_ROOT_PREFIX=/home/ubuntu/micromamba
export CONDA_PREFIX=/home/ubuntu/micromamba/envs/isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=Y

# ===== 训练 =====
cd /data/IsaacLab
# 平地
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --headless
# 粗糙地形
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 4096 --headless

# ===== 评估（录视频） =====
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
  --num_envs 50 --video \
  --checkpoint <checkpoint.pt> --headless

# ===== 查看训练曲线 =====
tensorboard --logdir /data/IsaacLab/logs/rsl_rl
```

### 任务 registry 名称（可切换的机器人/地形）

| 任务 | registry id |
|---|---|
| Go2 平地 | `Isaac-Velocity-Flat-Unitree-Go2-v0` |
| Go2 粗糙地形 | `Isaac-Velocity-Rough-Unitree-Go2-v0` |
| ANYmal-C 粗糙地形 | `Isaac-Velocity-Rough-Anymal-C-v0` |
| 宇树 H1 双足 | `Isaac-Velocity-{Flat,Rough}-Unitree-H1-v0` |
| 宇树 G1 双足 | `Isaac-Velocity-{Flat,Rough}-Unitree-G1-v0` |

---

## 13. 后续可选项

1. **继续训练**：提高 `--max_iterations` 或调 reward 权重，进一步降低跟踪误差
2. **换任务**：双足 H1/G1（难度更高，需更久训练）
3. **motion imitation**：引入 AMP + mocap 参考轨迹（会用到最初想要的"参考数据集"）
4. **sim2real**：将策略部署到真实 Go2（Isaac Lab 提供 sim2sim 工具链）

---

*文档由训练全流程自动整理生成，所有命令、版本号、指标均来自实际执行记录。*
