# 具身智能强化学习训练指南：宇树 Go2 速度跟踪（从 0 到 1 手工构建）

> 目标：在一台裸机上，从零手工搭建 Isaac Sim + Isaac Lab 训练环境，训练宇树 Go2 四足机器人学会按速度指令行走。
> 读者：会用 PyTorch、对强化学习不熟的学习者。
> 交付物：可加载的策略 checkpoint + 评测视频 + 收敛指标。
> 本指南每步都附**测试案例**（怎么验证这步成功）与**实测结果**，全部来自本机 2026-09-18 的真实执行记录。

---

## 0. 成果总览

| 阶段 | 任务 | 迭代 | 耗时 | 速度跟踪奖励 `track_lin_vel_xy_exp` |
|---|---|---|---|---|
| 平地 | Flat-Unitree-Go2 | 300 | ~3 分钟 | 0.004 → **1.432** |
| 粗糙地形 | Rough-Unitree-Go2 | 1500 | ~1.5 小时 | 0.005 → **1.228** |

最终策略在粗糙地形课程（terrain_level **5.6**）上稳定站立 + 按指令行走，存活率 **~92%**。

---

## 1. 前置认知：为什么要这样搭

### 1.1 这是一个"在线强化学习"问题，不是"下载模型 + 训练数据"

```
典型的 CV/NLP 项目：  下载预训练模型 + 下载标注数据集 → 微调
具身智能在线 RL：     无预训练模型、无训练数据集 → agent 在模拟器里自己探索
```

四足行走没有现成的"标注答案"：每一帧该迈哪条腿，取决于当前姿态、地形、指令——这是**连续决策**问题。唯一需要的"老师"是**奖励函数**：走对了加分，摔倒了扣分。因此：

- 不需要 HuggingFace 预训练模型（足式策略跨机器人迁移极难）
- 不需要训练数据集（agent 靠奖励信号自主探索）

### 1.2 五个组件各司其职

| 组件 | 角色 | 一句话理解 |
|---|---|---|
| **Isaac Sim** | 世界 | Vulkan 渲染 + PhysX 物理引擎，机器人在里面摔倒、行走 |
| **Isaac Lab** | 任务 | 定义"Go2 速度跟踪任务"：观测、奖励、随机化都写好了 |
| **rsl-rl** | 算法 | PPO 实现：读观测 → 输出动作 → 按奖励更新策略 |
| **PyTorch** | 计算 | 策略网络的张量计算，跑在 GPU 上 |
| **micromamba** | 环境隔离 | 给整套工具一个独立的 Python 3.11 沙盒 |

**数据流闭环**：Isaac Sim 产生观测（姿态/速度/地形）→ rsl-rl 策略网络输出动作（12 个关节）→ Isaac Sim 执行动作、算奖励 → rsl-rl 用奖励更新网络 → 循环。

---

## 2. 构建测试案例总览

整个构建过程拆成 8 个可独立验证的测试案例。**每过一个，才进入下一个**，避免在几十 GB 的安装中途卡死却找不到断点。

| # | 测试案例 | 通过判据 |
|---|---|---|
| T1 | 硬件探测 | GPU 型号/显存/驱动/Vulkan/GLIBC 满足要求 |
| T2 | 网络探测 | 关键下载域名可达（本机 `github.com` 不可达，需绕行） |
| T3 | Python 环境 | `python -V` = 3.11.x，且与系统 3.12 隔离 |
| T4 | PyTorch | `torch.cuda.is_available() == True` |
| T5 | Isaac Sim | `import isaacsim` 输出版本，EULA 已接受 |
| T6 | Isaac Lab + rsl-rl | `import isaaclab / isaaclab_tasks / isaaclab_rl / rsl_rl` 全通 |
| T7 | 模拟器启动 | 日志出现 Vulkan 识别 GPU + `Setup complete` |
| T8 | 冒烟训练 | 3 迭代跑完，输出 reward，`exit=0` |

---

## 3. 第 1 步 · 环境准备（T1~T3）

### 3.1 硬件探测（T1）

**业务目的**：Isaac Sim 强依赖 NVIDIA GPU 做渲染和物理仿真，硬件不达标后面必挂。

```bash
nvidia-smi                          # GPU 型号、显存、驱动版本
nvidia-smi --query-gpu=driver_version,compute_cap,name,memory.total --format=csv
ls /usr/share/vulkan/icd.d/         # Vulkan 渲染后端在位
ldd --version                       # GLIBC ≥2.35 才能用 pip 版 Isaac Sim
free -h && df -h                    # 内存 / 磁盘（Isaac Sim 资产要几十 GB）
```

**本机实测**：

| 项 | 值 | 结论 |
|---|---|---|
| GPU | NVIDIA L20 48GB（compute 8.9） | 足式 RL 够用 |
| 驱动 | 580.126.20（CUDA runtime 13.0 / nvcc 12.8） | 配 torch cu128 |
| GLIBC | 2.39 | ≥2.35，pip 安装可行 |
| Vulkan | NVIDIA ICD 在位 | headless 录视频必需 |
| 系统 Python | 3.12.3（无 pip/conda） | ⚠️ 与 Isaac Sim 要求的 3.11 冲突，见 3.3 |

### 3.2 网络探测（T2）

**业务目的**：训练要下载几十 GB 的包和资产，关键域名不可达会卡死且报错难懂。**提前探测省大量时间。**

```bash
curl -sI https://pypi.org
curl -sI https://pypi.nvidia.com              # Isaac Sim 专用源
curl -sI https://download.pytorch.org/whl/cu128
curl -sI https://github.com                   # ⚠️ 本机实测不可达
curl -sI https://codeload.github.com          # GitHub tarball 下载通道
curl -sI https://conda.anaconda.org
```

**本机关键发现**：`github.com` 主站和 git 协议**超时不可达**，但 `codeload.github.com`（tarball 通道）可达。

> 业务含义：下载 Isaac Lab 源码不能用 `git clone`，必须用 tarball。这个坑在第 4 步直接绕行。

### 3.3 安装 micromamba + 创建 Python 3.11 环境（T3）

**业务目的**：系统 Python 是 3.12.3，而 Isaac Sim 5.x **硬性要求 3.11**。不能动系统 Python（Ubuntu 24.04 系统工具依赖它），所以需要独立环境。选 micromamba 而非 conda/uv：conda 重，uv 装 Isaac Lab 有兼容性问题（官方标注 experimental），micromamba 轻量无 sudo。

```bash
mkdir -p ~/micromamba/bin
curl -fsSL -o ~/micromamba/bin/micromamba \
  "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"
chmod +x ~/micromamba/bin/micromamba
export MAMBA_ROOT_PREFIX=~/micromamba

~/micromamba/bin/micromamba create -n isaaclab -c conda-forge python=3.11 -y
~/micromamba/bin/micromamba run -n isaaclab python -V   # 通过判据：Python 3.11.x
```

---

## 4. 第 2~4 步 · 安装 PyTorch / Isaac Sim / Isaac Lab（T4~T6）

### 4.1 PyTorch（T4）

**业务目的**：策略网络需要 GPU 加速，PyTorch 的 GPU 版本必须和 CUDA 架构匹配。本机驱动是 580.126（CUDA 13.0），但 Isaac Lab v2.3.2 官方 pin 的是 **cu128**（CUDA 12.8）的 torch 2.7.0，这是官方测试过的组合，**不要自己乱升**。

```bash
~/micromamba/bin/micromamba run -n isaaclab python -m pip install \
  torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

**验证（T4，别跳过）**：

```bash
~/micromamba/bin/micromamba run -n isaaclab python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**通过判据**：`2.7.0+cu128 True NVIDIA L20`。`cuda.is_available() == True` 是所有后续 GPU 训练的前提。

### 4.2 Isaac Sim（T5）

**业务目的**：Isaac Sim 提供**渲染**（Vulkan，录视频用）和 **PhysX 物理引擎**（重力/碰撞/关节力矩，机器人"摔倒"就是它在算）。没有它，RL 就没有"世界"可试错。

```bash
~/micromamba/bin/micromamba run -n isaaclab python -m pip install \
  "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

- `[all]`：装所有扩展
- `[extscache]`：缓存扩展资产，避免首次运行从云端拉取（会慢 10 分钟+）
- `--extra-index-url https://pypi.nvidia.com`：Isaac Sim 专用源，不在这装不到

**接受 EULA**（否则首次运行交互式阻塞）：

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

**验证（T5）**：

```bash
~/micromamba/bin/micromamba run -n isaaclab python -c \
  "import isaacsim; from importlib.metadata import version; print(version('isaacsim'))"
# 通过判据：5.1.0.0
```

### 4.3 Isaac Lab + rsl-rl（T6）

**业务目的**：Isaac Lab 是任务的"定义层"（内置 Go2 速度跟踪任务的观测/奖励/随机化），rsl-rl 是 PPO 算法实现。两者都以 editable 模式从源码装。

```bash
# 前置：源码包依赖需要现场编译
sudo apt-get install -y --no-install-recommends cmake build-essential

# 下载源码（⚠️ 不能用 git clone，本机 github.com 不可达）
cd /data
curl -fsSL -o IsaacLab-v2.3.2.tar.gz \
  "https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/refs/tags/v2.3.2"
tar xzf IsaacLab-v2.3.2.tar.gz
mv IsaacLab-2.3.2 IsaacLab
rm IsaacLab-v2.3.2.tar.gz

# 安装扩展 + rsl-rl
export CONDA_PREFIX=/home/ubuntu/micromamba/envs/isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=Y
cd /data/IsaacLab
./isaaclab.sh --install rsl_rl
```

> 为什么 pin v2.3.2？v3.0.0 当时还在 beta，做 demo 要稳定不要新。

**两个必须手动处理的坑**：

**坑 A — `flatdict` 构建失败**（`ModuleNotFoundError: pkg_resources`）：setuptools ≥84 移除了 `pkg_resources`，而老源码包 `flatdict==4.0.1` 构建时还依赖它。

```bash
pip install "setuptools<81"                       # 降级，恢复 pkg_resources
pip install flatdict==4.0.1 --no-build-isolation   # 禁用隔离单独装
```

**坑 B — 核心 `isaaclab` 包漏装**：官方脚本首次可能只装上 assets/contrib/mimic/rl/tasks，核心包漏掉。

```bash
cd /data/IsaacLab/source/isaaclab
pip install --editable . --no-build-isolation
```

**验证（T6）**：

```bash
python -c "import isaaclab; print(isaaclab.__version__)"   # 0.54.2
python -c "import isaaclab_tasks, isaaclab_rl, isaaclab_assets"
python -c "import rsl_rl"
```

### 4.4 关于 `pxr` 报错（重要认知）

裸跑 `import pxr` 会报 `ModuleNotFoundError`——**这是预期行为，不是错误**。`pxr`（OpenUSD 的 Python 绑定）由 Isaac Sim 内核在 `AppLauncher` 启动时动态注入 `sys.path`。所以验证 Isaac Sim 能不能用，**不能靠裸 import，要靠启动模拟器**（见第 5 步 T7）。

---

## 5. 第 5 步 · 环境验证（T7~T8）

### 5.1 启动模拟器（T7）

```bash
export CONDA_PREFIX=/home/ubuntu/micromamba/envs/isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=Y
cd /data/IsaacLab
python scripts/tutorials/00_sim/create_empty.py --headless
```

**通过判据**（日志里找）：
- `Graphics API: Vulkan` + `NVIDIA L20` 被列出
- PhysX 插件就绪
- `[INFO]: Setup complete...`
- 脚本**常驻不退出**（正常，模拟器启动后等待，手动 Ctrl+C 结束）

### 5.2 冒烟训练（T8，全链路打通）

**业务目的**：用极少迭代数跑通"模拟器 → 环境 → PPO → 日志"全链路，确认没有环节断掉，再投入长时间训练。

```bash
cd /data/IsaacLab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 64 --max_iterations 3 --headless
```

**通过判据**：输出 3 个 `Learning iteration`，每个带 reward 和 `error_vel_xy` 指标，最后 `exit=0`。本机实测 `4097 steps/s`。

---

## 6. 第 6 步 · 正式训练

### 6.1 两阶段课程（平地 → 粗糙地形）

**业务目的**：课程学习（curriculum learning）——先让 agent 在平地学会基本步态，再迁移到粗糙地形，比直接上困难任务收敛更快更稳。

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
| `--num_envs 4096` | 4096 个并行环境 | PPO 是 on-policy，极度依赖大量并行采样；环境越多经验越多样，收敛越快 |
| `--headless` | 不渲染画面 | 训练不需要看画面，关渲染省显存提速；只有录视频才开 |
| `num_steps_per_env=24` | 每环境采 24 步 | 控制每次策略更新样本量（4096×24≈10 万样本） |
| `num_learning_epochs=5` | 每批数据学 5 轮 | 数据利用率 |
| `learning_rate=1e-3` | 学习率 | 官方调好的默认值，demo 不动 |

> 超参在 `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py`，官方已调好，新手先不动。

**策略网络结构**（模型加载时输出）：

```
Actor MLP:  Linear(235→512) → ELU → Linear(512→256) → ELU
            → Linear(256→128) → ELU → Linear(128→12)
Critic MLP: Linear(235→512) → ELU → Linear(512→256) → ELU
            → Linear(256→128) → ELU → Linear(128→1)
```

- 输入 235 维（观测：本体状态 + 关节 + 指令 + 高度扫描）
- 输出 12 维（Go2 的 12 个关节自由度）

### 6.3 训练过程观察指标

```
Metrics/base_velocity/error_vel_xy   # 水平速度跟踪误差（越小越好，核心）
Metrics/base_velocity/error_vel_yaw  # 转向角速度误差（越小越好）
Episode_Termination/time_out         # 存活率（跑满 episode 不倒的比例）
Curriculum/terrain_levels            # 地形难度（rough 特有，随训练递增）
```

**重要认知**：`error_vel_xy` 可能先降后升再降——这是**正常的**。课程学习会逐步加大指令幅度和地形难度，agent 需要不断适应更难的任务。

---

## 7. 测试结果（全量实测，来自 TensorBoard 事件）

### 7.1 平地（300 迭代，3 分 3 秒）

| 指标 | iter 0 | iter 150 | iter 299 |
|---|---|---|---|
| `track_lin_vel_xy_exp`（奖励↑） | 0.004 | 1.200 | **1.432** |
| `error_vel_xy`（误差↓） | 0.018 | 0.422 | **0.178** |
| `error_vel_yaw`（误差↓） | 0.019 | 0.496 | **0.340** |
| 存活率 `time_out` | 0.011 | 0.984 | **0.9975** |

### 7.2 粗糙地形（1500 迭代，1 小时 31 分钟）

| 指标 | iter 0 | iter 750 | iter 1499 |
|---|---|---|---|
| `track_lin_vel_xy_exp`（奖励↑） | 0.005 | 1.078 | **1.228** |
| `track_ang_vel_z_exp`（奖励↑） | — | — | **0.578** |
| `error_vel_xy`（误差↓） | 0.015 | 0.473 | **0.353** |
| `error_vel_yaw`（误差↓） | 0.017 | 0.492 | **0.416** |
| 存活率 `time_out` | ~0.03 | — | **0.916** |
| 地形课程 `terrain_levels` | 0 | — | **5.614** |

**结论**：机器人在粗糙地形课程（难度逐步增加）上学会了稳定站立 + 按指令速度行走。误差曲线先升后降是正常的——curriculum 逐步加大指令幅度，中段波动后回归收敛。

---

## 8. 第 7 步 · 评估与录视频

### 8.1 用 PLAY 环境评测

**业务目的**：训练时环境有随机扰动（推挤、地形随机化）来增强鲁棒性；评估时用干净的 `-Play` 环境（无扰动、无推挤），得到策略的真实能力。

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

### 8.2 评测视频元数据（实测）

| 项 | 值 |
|---|---|
| 路径 | `logs/rsl_rl/unitree_go2_rough/.../videos/play/rl-video-step-0.mp4` |
| 时长 | 9.98 秒 |
| 帧数 | 499 帧 |
| 分辨率 | 1280×720 |
| 帧率 | 50 fps |
| 编码 | h264 |
| 大小 | 1.4 MB |

### 8.3 验收指标（足式运动的标准）

足式运动**没有"成功率"**（走路没有 0/1 成败），业界标准是**指令跟踪误差**——你让机器人以 0.5 m/s 前进，它实际走得多接近。

| 指标 | 方向 | 含义 |
|---|---|---|
| `track_lin_vel_xy_exp` | 越大越好 | 水平速度跟踪奖励 |
| `track_ang_vel_z_exp` | 越大越好 | 转向角速度跟踪奖励 |
| `error_vel_xy` | 越小越好 | 水平速度跟踪误差 |
| `error_vel_yaw` | 越小越好 | 转向角速度跟踪误差 |
| `time_out`（存活率） | 越大越好 | 不跌倒跑满 episode 的比例 |

---

## 9. 交付物清单

| 产物 | 路径 | 大小 |
|---|---|---|
| 平地 checkpoint | `artifacts/flat/model_299.pt` | 983 KB |
| 粗糙地形 checkpoint | `artifacts/rough/model_1499.pt` | 6.9 MB |
| 导出 PyTorch 策略 | `artifacts/rough/exported/policy.pt` | 1.2 MB |
| 导出 ONNX 策略 | `artifacts/rough/exported/policy.onnx` | 1.1 MB |
| 评测视频 | `artifacts/video/rl-video-step-0.mp4` | 1.4 MB |
| TensorBoard 事件 | `logs/rsl_rl/unitree_go2_{flat,rough}/**/events.out.tfevents.*` | — |

checkpoint 校验（实测）：`model_299.pt` → `iter=299`，`model_1499.pt` → `iter=1499`，均含 `model_state_dict`。

---

## 10. 复现命令速查

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

### 任务 registry 名称（可切换的机器人/地形）

| 任务 | registry id |
|---|---|
| Go2 平地 | `Isaac-Velocity-Flat-Unitree-Go2-v0` |
| Go2 粗糙地形 | `Isaac-Velocity-Rough-Unitree-Go2-v0` |
| ANYmal-C 粗糙地形 | `Isaac-Velocity-Rough-Anymal-C-v0` |
| 宇树 H1 双足 | `Isaac-Velocity-{Flat,Rough}-Unitree-H1-v0` |
| 宇树 G1 双足 | `Isaac-Velocity-{Flat,Rough}-Unitree-G1-v0` |

---

## 附录 · 踩坑清单

| # | 问题 | 根因 | 解决方案 |
|---|---|---|---|
| 1 | `github.com` + git 协议超时 | 网络限制 | 用 `codeload.github.com` tarball 下载源码 |
| 2 | `isaaclab.sh` 报 `terminal type 'dumb'` | 脚本 `tabs 4` 需真实终端 | 用带 pty 的终端运行 |
| 3 | `flatdict` 构建失败 `pkg_resources` | setuptools 84 移除该模块 | 降级 `setuptools<81` + `--no-build-isolation` |
| 4 | 核心 `isaaclab` 包漏装 | 官方脚本首次未装上核心包 | 手动 `pip install --editable .` |
| 5 | 裸 `import pxr` 报错 | 预期行为，AppLauncher 启动后才注入 | 用 `create_empty.py` 验证，非裸 import |
| 6 | 首次运行交互式阻塞 | EULA 提示 | 设 `OMNI_KIT_ACCEPT_EULA=Y` |

**已知无害的依赖冲突**（Isaac Sim 主动 pin 导致，不影响训练）：
- `wheel 0.48.0 requires packaging>=24.0`（Isaac Sim 强降级 packaging 至 23.0）
- `fastapi requires starlette<0.46.0`（Isaac Sim 强 pin starlette 0.49.1）

---

> 本项目已把激活/训练/评估封装成 `scripts/` 下的可复现脚本（`activate.sh` / `train.sh` / `eval.sh` / `profile.sh`）与一键启动 `run.sh`，见 [README.md](./README.md)。完整环境搭建踩坑记录见 [SETUP_GUIDE.md](./SETUP_GUIDE.md)，逐条执行日志见 [TRAINING_LOG.md](./TRAINING_LOG.md)。
