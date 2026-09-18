# 从 0 到 1 手工搭建具身智能 RL 训练环境

> 目标：在一台裸机上，从零手工搭建 Isaac Sim + Isaac Lab 训练环境，训练宇树 Go2 四足机器人学会按速度指令行走。
> 读者：会用 PyTorch、对强化学习不熟的学习者。
> 特点：每一步不仅告诉你怎么做，更告诉你**为什么**——业务的来龙去脉。

---

## 第 0 章 · 先建立全局认知（最重要）

在动手之前，先回答三个问题，否则后面每一步都会显得莫名其妙。

### 0.1 我们要做什么？

让一个四足机器人在物理模拟器里，通过**试错**学会走路。这不是写规则控制每个关节，而是让算法自己探索出"如何协调 12 个关节"。

### 0.2 为什么是"在线强化学习"，而不是"下载模型 + 训练数据"？

这是本项目第一个关键决策，也是新手最容易走错的地方。

```
典型的 CV/NLP 项目：  下载预训练模型 + 下载标注数据集 → 微调
具身智能在线 RL：     无预训练模型、无训练数据集 → agent 在模拟器里自己探索
```

**业务目的**：四足行走没有现成的"标注答案"。每一帧该迈哪条腿，取决于当前姿态、地形、指令——这是一个**连续决策**问题，不是"输入图片输出标签"的分类问题。所以：

- 不需要 HuggingFace 模型（足式策略跨机器人迁移极难，收益小）
- 不需要训练数据集（agent 用"奖励函数"当老师，好步态得分高、摔倒得分低）

唯一需要的"老师"是**奖励函数**：走对了加分，摔倒了扣分。

### 0.3 五个组件各司其职（数据流）

```
┌─────────────────────────────────────────────────────┐
│  Isaac Lab（任务定义 + 环境管理 + 胶水层）             │
│                                                     │
│  ┌──────────┐   观测/奖励    ┌──────────────────┐   │
│  │  Isaac Sim│ ◄──────────► │  rsl-rl (PPO 算法)│   │
│  │  渲染+物理 │   动作/指令    │                  │   │
│  └──────────┘               └──────────────────┘   │
│        ▲                           │               │
│        │ Vulkan 渲染               │ torch 张量      │
│        │ PhysX 物理                │ GPU 训练        │
│        ▼                           ▼               │
│  NVIDIA GPU (L20)             PyTorch (cu128)       │
└─────────────────────────────────────────────────────┘
```

| 组件 | 角色 | 一句话理解 |
|---|---|---|
| **Isaac Sim** | 世界 | 渲染画面 + PhysX 物理引擎，机器人在里面摔倒、行走 |
| **Isaac Lab** | 任务 | 定义"Go2 速度跟踪任务"：观测有哪些、奖励怎么算、环境怎么随机化 |
| **rsl-rl** | 算法 | PPO 实现，读观测 → 输出动作 → 根据奖励更新策略 |
| **PyTorch** | 计算 | 神经网络的张量计算，跑在 GPU 上 |
| **micromamba** | 环境隔离 | 给整套工具一个独立的 Python 3.11 沙盒 |

**数据流闭环**：Isaac Sim 产生观测（姿态/速度/地形）→ rsl-rl 策略网络输出动作（12 个关节力矩）→ Isaac Sim 执行动作、算奖励 → rsl-rl 用奖励更新网络 → 循环。

---

## 第 1 章 · 环境准备

### 1.1 确认硬件与驱动

**业务目的**：Isaac Sim 强依赖 NVIDIA GPU 做渲染和物理仿真。GPU 型号、显存、驱动版本、Vulkan 支持，直接决定后面能不能跑、跑多快。

```bash
nvidia-smi                          # 看 GPU 型号、显存、驱动版本
nvidia-smi --query-gpu=driver_version,compute_cap,name,memory.total --format=csv
ls /usr/share/vulkan/icd.d/         # 确认 Vulkan 渲染后端在位
ldd --version                       # GLIBC ≥2.35 才能用 pip 版 Isaac Sim
free -h                             # 内存
df -h                               # 磁盘空间（Isaac Sim 资产要几十 GB）
```

本项目实测环境：

| 项 | 值 | 为什么重要 |
|---|---|---|
| GPU | NVIDIA L20 48GB | 显存决定能开多少并行环境 |
| 驱动 | 580.126.20 | 需 ≥ Isaac Sim 要求的最低驱动 |
| GLIBC | 2.39 | ≥2.35，pip 安装可行 |
| Vulkan | NVIDIA ICD 在位 | headless 录视频必需 |
| 系统 Python | 3.12.3（无 pip/conda） | ⚠️ 与 Isaac Sim 要求的 3.11 冲突，见 1.3 |

### 1.2 检查网络连通性

**业务目的**：训练需要下载几十 GB 的包和资产。如果某个关键域名不可达，后面会卡死且报错难懂。**提前探测能省大量时间。**

```bash
# 逐个探测关键域名
curl -sI https://pypi.org                     # Python 包
curl -sI https://pypi.nvidia.com              # Isaac Sim 专用源
curl -sI https://download.pytorch.org/whl/cu128  # PyTorch
curl -sI https://github.com                   # ⚠️ 本项目实测不可达
curl -sI https://codeload.github.com          # GitHub tarball 下载
curl -sI https://conda.anaconda.org           # conda 包
```

**本项目的关键发现**：`github.com` 主站和 git 协议**超时不可达**，但 `codeload.github.com`（GitHub 的 tarball 下载通道）可达。

> 业务含义：下载 Isaac Lab 源码不能用 `git clone`，必须用 `codeload.github.com` 的 tarball。这个坑提前发现，后面第 3 章直接用绕行方案。

### 1.3 安装 micromamba（环境隔离器）

**业务目的**：系统 Python 是 3.12.3，而 Isaac Sim 5.x **硬性要求 Python 3.11**。你不能动系统 Python（Ubuntu 24.04 系统工具依赖它），也不能把几十个包塞进系统环境污染它。所以需要一套独立的 Python 环境管理工具。

为什么选 micromamba 而不是 conda/uv？
- conda 体积大、装得慢
- uv 装 Isaac Lab 有兼容性问题（官方文档标注 experimental）
- micromamba 轻量、无 sudo、装到用户目录即可

```bash
# 直接下载 release 二进制（绕过 github.com 主站）
mkdir -p ~/micromamba/bin
curl -fsSL -o ~/micromamba/bin/micromamba \
  "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"
chmod +x ~/micromamba/bin/micromamba
export MAMBA_ROOT_PREFIX=~/micromamba
~/micromamba/bin/micromamba --version   # 验证
```

### 1.4 创建 Python 3.11 环境

**业务目的**：为 Isaac Sim 5.x 提供匹配的 Python 3.11 沙盒，与系统 3.12 完全隔离。

```bash
~/micromamba/bin/micromamba create -n isaaclab -c conda-forge python=3.11 -y
~/micromamba/bin/micromamba run -n isaaclab python -V   # 应输出 Python 3.11.x
```

环境名 `isaaclab` 是自定义的，后面所有命令都用这个名字。

---

## 第 2 章 · 安装 PyTorch

### 2.1 装 CUDA 版 PyTorch

**业务目的**：策略网络是神经网络，需要 GPU 加速。而 PyTorch 的 GPU 版本必须和本机 CUDA 架构匹配——驱动是 580.126（CUDA 13.0 runtime），但 Isaac Lab v2.3.2 官方 pin 的是 **cu128**（CUDA 12.8）的 torch 2.7.0，这是经过官方测试的组合，**不要自己乱升版本**。

```bash
~/micromamba/bin/micromamba run -n isaaclab python -m pip install \
  torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

**验证**（关键，别跳过）：

```bash
~/micromamba/bin/micromamba run -n isaaclab python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

应输出 `2.7.0+cu128 True NVIDIA L20`。`cuda.is_available() == True` 是所有后续 GPU 训练的前提。

---

## 第 3 章 · 安装 Isaac Sim

### 3.1 装 Isaac Sim（世界引擎）

**业务目的**：Isaac Sim 提供两样东西——**渲染**（Vulkan 画画面，录视频用）和 **PhysX 物理引擎**（模拟重力、碰撞、关节力矩，机器人"摔倒"就是它在算）。没有它，RL 就没有"世界"可以试错。

```bash
~/micromamba/bin/micromamba run -n isaaclab python -m pip install \
  "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

- `[all]`：装所有扩展
- `[extscache]`：缓存扩展资产，避免首次运行从云端拉取（首次拉取会慢 10 分钟+）
- `--extra-index-url https://pypi.nvidia.com`：Isaac Sim 专用 PyPI 源，不在这装不到

### 3.2 接受 EULA

**业务目的**：Isaac Sim 是 NVIDIA 专有软件，首次运行会弹交互式许可协议（EULA）提示阻塞。非交互环境下用环境变量静默接受。

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

验证：

```bash
~/micromamba/bin/micromamba run -n isaaclab python -c "import isaacsim; from importlib.metadata import version; print(version('isaacsim'))"
```

应输出 `5.1.0.0`。

---

## 第 4 章 · 安装 Isaac Lab（任务层）

### 4.1 装编译工具

**业务目的**：Isaac Lab 的部分依赖（如 `flatdict`）是源码包，安装时需要现场编译，需要 cmake + build-essential。

```bash
sudo apt-get install -y --no-install-recommends cmake build-essential
```

### 4.2 下载 Isaac Lab 源码

**业务目的**：Isaac Lab 是任务的"定义层"，内置了 Go2 速度跟踪任务（观测/奖励/随机化都写好了）。它需要从源码安装（editable 模式），因为你要看、也可能要改任务定义。

**⚠️ 关键：不能用 git clone**（第 1.2 章已探测 github.com 不可达），用 tarball 绕行：

```bash
cd /data
curl -fsSL -o IsaacLab-v2.3.2.tar.gz \
  "https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/refs/tags/v2.3.2"
tar xzf IsaacLab-v2.3.2.tar.gz
mv IsaacLab-2.3.2 IsaacLab
rm IsaacLab-v2.3.2.tar.gz
```

> 为什么 pin v2.3.2？因为 v3.0.0 当时还在 beta 阶段，做 demo 要稳定不要新。

### 4.3 安装 Isaac Lab 扩展 + rsl-rl（算法层）

**业务目的**：Isaac Lab 仓库里包含了核心包（`isaaclab`）、任务包（`isaaclab_tasks`）、算法包（`isaaclab_rl`）。rsl-rl 是 PPO 算法的实现。这一步把它们以 editable 模式装进环境，同时装 rsl-rl 算法框架。

```bash
export CONDA_PREFIX=/home/ubuntu/micromamba/envs/isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=Y
cd /data/IsaacLab
./isaaclab.sh --install rsl_rl
```

**⚠️ 踩坑 1**：`isaaclab.sh` 里有 `tabs 4` 命令，在无真实终端的环境会报 `terminal type 'dumb'` 错误。**解决：用带 pty 的终端运行**。

**⚠️ 踩坑 2**：这一步可能只装上了 assets/contrib/mimic/rl/tasks，**核心 `isaaclab` 包漏装**。验证：

```bash
pip list | grep -i isaaclab
# 必须看到 isaaclab 这行，否则继续下面修复
```

### 4.4 两个必须手动处理的坑

**坑 A — `flatdict` 构建失败**

症状：`ModuleNotFoundError: No module named 'pkg_resources'`

**业务目的/根因**：新版本 setuptools（≥84）移除了 `pkg_resources` 模块，而 `flatdict==4.0.1` 是个老源码包，构建时还依赖它。

```bash
pip install "setuptools<81"                      # 降级，恢复 pkg_resources
pip install flatdict==4.0.1 --no-build-isolation  # 禁用隔离单独装
```

**坑 B — 核心 `isaaclab` 包缺失**

```bash
cd /data/IsaacLab/source/isaaclab
pip install --editable . --no-build-isolation
```

装完验证（关键）：

```bash
python -c "import isaaclab; print(isaaclab.__version__)"   # 应输出 0.54.2
python -c "import isaaclab_tasks, isaaclab_rl, isaaclab_assets"
python -c "import rsl_rl"                                   # PPO 算法
```

### 4.5 关于 `pxr` 报错的解释（重要认知）

如果你裸跑 `python -c "import pxr"`，会报 `ModuleNotFoundError: No module named 'pxr'`。

**这不是错误，是预期行为**。`pxr`（OpenUSD 的 Python 绑定）由 Isaac Sim 内核在启动时动态注入 `sys.path`，只有通过 `AppLauncher` 启动模拟器后才可用。所以验证 Isaac Sim 能不能用，**不能靠裸 import，要靠启动模拟器**（见第 5 章）。

---

## 第 5 章 · 验证环境

### 5.1 启动模拟器（第一道验证）

**业务目的**：确认"世界引擎"能真正启动——GPU 被识别、PhysX 就绪、渲染管线正常。这一步过了，才谈得上训练。

```bash
export CONDA_PREFIX=/home/ubuntu/micromamba/envs/isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=Y
cd /data/IsaacLab
python scripts/tutorials/00_sim/create_empty.py --headless
```

**成功标志**（日志里找这些）：
- `Graphics API: Vulkan` + `NVIDIA L20` 被列出
- `[INFO]: Setup complete...`
- 脚本**常驻不退出**（这是正常的，它启动模拟器后等待，手动 Ctrl+C 结束）

### 5.2 冒烟训练（第二道验证，全链路打通）

**业务目的**：用极少的迭代数跑通"模拟器 → 环境 → PPO → 日志"全链路，确认没有任何环节断掉，再投入长时间训练。

```bash
cd /data/IsaacLab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 64 --max_iterations 3 --headless
```

**成功标志**：输出 3 个 `Learning iteration`，每个都有 reward 和 `error_vel_xy` 等指标，最后 `exit=0`。

---

## 第 6 章 · 正式训练

### 6.1 两阶段课程（平地 → 粗糙地形）

**业务目的**：这是强化学习的**课程学习（curriculum learning）**思想——先让 agent 在简单任务（平地）学会基本步态，再迁移到困难任务（粗糙地形），比直接上困难任务收敛更快更稳。

```bash
# 阶段 A：平地（~3 分钟，快速出成果）
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --headless

# 阶段 B：粗糙地形（~1.5 小时，最终交付）
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 4096 --headless
```

### 6.2 关键超参为什么这么设

| 超参 | 值 | 业务目的 |
|---|---|---|
| `--num_envs 4096` | 4096 个并行环境 | PPO 是 on-policy 算法，**极度依赖大量并行采样**。环境越多，每次更新收集的经验越多样，收敛越快。L20 48GB 能轻松扛住 |
| `--headless` | 不渲染画面 | 训练不需要看画面，关掉渲染省显存、提速度。只有录视频才开 |
| `num_steps_per_env=24` | 每环境采 24 步 | 控制每次策略更新的样本量（4096×24≈10 万样本） |
| `num_learning_epochs=5` | 每批数据学 5 轮 | 数据利用率 |
| `learning_rate=1e-3` | 学习率 | 官方调好的默认值，demo 不动它 |

> 这些超参在 `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py` 里，官方已经调好，新手先不动。

### 6.3 训练过程观察

训练时会输出关键指标，理解它们：

```
Metrics/base_velocity/error_vel_xy   # 水平速度跟踪误差（越小越好，核心指标）
Metrics/base_velocity/error_vel_yaw  # 转向角速度误差（越小越好）
Episode_Termination/time_out         # 存活率（机器人跑满整个 episode 不倒的比例）
Curriculum/terrain_levels            # 地形难度（rough 任务特有，随训练递增）
```

**一个重要的认知**：训练中 `error_vel_xy` 可能先降后升再降，这是**正常的**——课程学习会逐步加大指令幅度和地形难度，agent 需要不断适应更难的任务。

---

## 第 7 章 · 评估与录视频

### 7.1 用 PLAY 环境评测

**业务目的**：训练时环境有随机扰动（推挤、地形随机化）来增强鲁棒性；评估时要用干净的 `-Play` 环境（无扰动、无推挤），得到策略的真实能力。

```bash
cd /data/IsaacLab
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
  --num_envs 50 \
  --video --video_length 500 \
  --checkpoint <你的checkpoint.pt> \
  --headless
```

- `--video` 会开相机渲染并录视频（这是唯一需要渲染的环节）
- 视频输出到 checkpoint 同目录的 `videos/play/`

### 7.2 验收指标（足式运动的标准）

**业务目的**：足式运动**没有"成功率"**这个概念（走路没有 0/1 成败），业界标准是**指令跟踪误差**——你让机器人以 0.5 m/s 前进，它实际走得多接近。

| 指标 | 方向 | 含义 |
|---|---|---|
| `track_lin_vel_xy_exp` | 越大越好 | 水平速度跟踪奖励 |
| `track_ang_vel_z_exp` | 越大越好 | 转向角速度跟踪奖励 |
| `error_vel_xy` | 越小越好 | 水平速度跟踪误差 |
| `error_vel_yaw` | 越小越好 | 转向角速度跟踪误差 |
| `time_out`（存活率） | 越大越好 | 不跌倒跑满 episode 的比例 |

---

## 第 8 章 · 回看全局（收尾）

现在回头看第 0 章的数据流图，你已经知道每一环是怎么搭起来的：

```
环境隔离（micromamba + Python 3.11）
    ↓
计算（PyTorch cu128）
    ↓
世界（Isaac Sim：渲染 + 物理）
    ↓
任务（Isaac Lab：观测/奖励/随机化）
    ↓
算法（rsl-rl：PPO）
    ↓
训练（flat → rough 课程学习）
    ↓
评估（PLAY 环境 + 视频 + 跟踪误差指标）
```

**核心思想一句话**：具身智能的在线 RL，是让 agent 在"物理世界"（模拟器）里通过"奖励信号"（走路加分、摔倒扣分）自己学会"如何运动"（协调 12 个关节），而不是教它每个动作该怎么做。

---

## 附录 · 完整命令速查

```bash
# ===== 环境激活（每次新终端都要做）=====
export MAMBA_ROOT_PREFIX=/home/ubuntu/micromamba
export CONDA_PREFIX=/home/ubuntu/micromamba/envs/isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=Y

# ===== 训练 =====
cd /data/IsaacLab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --headless   # 平地
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 4096 --headless  # 粗糙地形

# ===== 评估 =====
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
  --num_envs 50 --video \
  --checkpoint <checkpoint.pt> --headless

# ===== 看曲线 =====
tensorboard --logdir /data/IsaacLab/logs/rsl_rl
```

> 本项目已把这些命令封装成 `scripts/` 下的可复现脚本（`activate.sh` / `train.sh` / `eval.sh`），见 [README.md](./README.md)。
