# NVIDIA Nsight Compute 使用指南

> 目标：指导用户用 Nsight Compute (ncu) / Nsight Systems (nsys) 调试当前 Go2 训练进程，细看每一步 GPU 资源消耗，并理解结果背后的原因。
> 适用范围：本项目 `run.sh` 驱动的宇树 Go2 速度跟踪 RL 训练（Isaac Sim + Isaac Lab + rsl-rl PPO）。

---

## 0. 先说结论（最重要）

**不要直接 `ncu bash run.sh`，而是把 profiler 包在 `run.sh` 外面，并用 `--kernel-name` 过滤目标 kernel。**

`run.sh` **一个字都不用改**。原因和正确姿势见下文。

---

## 1. 背景认知：为什么不能"直接 ncu 整个 run.sh"

### 1.1 实测数据（本项目真实测得）

| 指标 | 正常 `run.sh` | `ncu` 直接包裹整个进程 |
|---|---|---|
| 吞吐 | 180094 steps/s | **270 steps/s（慢 600+ 倍）** |
| 单 iter 时间 | 0.55s | **362.93s** |

### 1.2 根本原因

这个训练进程的 GPU 时间**不是一段连续的 kernel**，而是两类碎片化的负载交替：

```
┌─────────────────────────────────────────────────────────┐
│  [环境 rollout]        [PPO 学习]      [环境 rollout]     │
│  PhysX 物理仿真  ←→  小 MLP 前向/反向  ←→  PhysX 物理仿真 │
│  (海量小 kernel)      (几个 gemm)      (海量小 kernel)    │
└─────────────────────────────────────────────────────────┘
```

- **环境 rollout 阶段**：Isaac Sim 的 PhysX 物理引擎跑大量小 kernel（碰撞、关节动力学），**不是**你想看的对象，但它们占 GPU 时间的绝大部分。
- **PPO 学习阶段**：策略网络是**小 MLP**（48→128→128→128→12，几十万参数），只有几个矩阵乘 kernel。

`ncu` 的设计目标是"深挖单个大 kernel"——它会**重放采样每个 kernel**。直接包裹整个进程，等于让它逐个重放成百上千个无关的 PhysX 物理 kernel，所以慢 600 倍，且采到的数据大部分是噪声。

---

## 2. 前置条件

本机工具已装好（无需额外安装）：

| 工具 | 版本 | 用途 |
|---|---|---|
| Nsight Compute (`ncu`) | 2025.1.1 | kernel 级：SM 占用率、内存带宽、指令吞吐 |
| Nsight Systems (`nsys`) | 2024.6.2 | 系统级：时间线、GPU/CPU 利用率、时序 |

```bash
ncu --version   # 确认存在
nsys --version  # 确认存在
```

> 注意：`nvvp`（NVIDIA Visual Profiler）已在 CUDA 12 移除，且是 GUI 工具在 headless 服务器无法启动。**本环境使用 `ncu`/`nsys` 的 CLI。**

---

## 3. 正确的使用方式

### 3.1 先粗看全局：用 Nsight Systems（推荐第一步）

**业务目的**：`ncu` 是"显微镜"（看单个 kernel 细节），`nsys` 是"广角镜"（看整个训练周期的资源消耗时间线）。先用 nsys 找到"哪个阶段在烧 GPU、哪个阶段 GPU 空闲、瓶颈在哪"，再决定要不要用 ncu 深挖。

```bash
cd /data/workdir/embodiment-rl-demo

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=/tmp/go2_timeline \
  --force-overwrite=true \
  bash run.sh flat --max_iterations 5
```

生成 `/tmp/go2_timeline.nsys-rep`，分析：

```bash
# CLI 查看 GPU kernel 耗时汇总
nsys stats /tmp/go2_timeline.nsys-rep

# 查看 GPU 利用率时间线摘要
nsys stats --report cuda_gpu_kern_sum /tmp/go2_timeline.nsys-rep
```

**结果解读**（你会看到）：
- 时间线上 GPU 利用率**呈脉冲状**：环境 rollout 时 PhysX kernel 密集，PPO 学习时只有几个短促的矩阵乘。
- 这说明训练**不是 GPU 计算密集型**，而是物理仿真与学习的交替。GPU 资源消耗的"每一步"其实是碎片化的，没有单一的大 kernel 独占。

### 3.2 深挖单个 kernel：用 Nsight Compute（过滤目标）

**业务目的**：想看 PPO 学习中策略网络的矩阵乘 kernel 到底吃了多少 SM、多少带宽。用 `--kernel-name` 正则只采样 torch 的 gemm/elementwise kernel，跳过 PhysX 噪声。

```bash
cd /data/workdir/embodiment-rl-demo

ncu \
  --launch-skip 200 \
  --launch-count 10 \
  --set full \
  --kernel-name "regex:.*(gemm|mma|elementwise|softmax).*" \
  bash run.sh flat --max_iterations 3
```

关键参数解释：

| 参数 | 值 | 作用 |
|---|---|---|
| `--launch-skip 200` | 跳过前 200 个 kernel | 跳过 Isaac Sim 启动时的资产加载噪声 |
| `--launch-count 10` | 只采 10 个 kernel | 别贪多，采到目标 kernel 即可 |
| `--set full` | 完整指标集 | 采样 SM 占用率、内存带宽、指令吞吐等全部指标 |
| `--kernel-name regex:...` | 正则过滤 kernel 名 | **只采样名字匹配的 kernel**，跳过 PhysX 物理 kernel |

### 3.3 采什么 kernel 名？

torch 的矩阵运算 kernel 名字通常包含这些关键字：

| kernel 名片段 | 对应操作 |
|---|---|
| `gemm` / `mm` | 矩阵乘（Linear 层的核心） |
| `elementwise` | 逐元素运算（激活函数 ELU、加法） |
| `softmax` | softmax（PPO 的动作分布） |
| `reduce` | 归约（loss 求和） |

可以先不设 `--kernel-name`，只 `--launch-count 5` 采样前几个 kernel 看名字，再精准过滤：

```bash
ncu --launch-skip 200 --launch-count 5 --set basic bash run.sh flat --max_iterations 2
# 看输出里 "Kernel Name" 列，找到你想深挖的名字，再回 3.2 精确过滤
```

---

## 4. 结果解读：为什么资源消耗长这样

### 4.1 你会看到的现象

深挖策略网络的 Linear 层 kernel 时，`ncu` 大概率报告：

- **SM 占用率低**（`Achieved Occupancy` 低）
- **内存吞吐占比高**（`Memory Throughput` 是主要瓶颈）
- **计算吞吐占比低**（`Compute (SM) Throughput` 不高）

### 4.2 原因（关键理解）

**这个 workload 的 GPU 计算量极小，是"内存/延迟受限"，不是"算力受限"。**

| 网络层 | 形状 | 参数量 |
|---|---|---|
| 输入 → 隐层1 | 48 → 128 | ~6k |
| 隐层1 → 隐层2 | 128 → 128 | ~16k |
| 隐层2 → 隐层3 | 128 → 128 | ~16k |
| 隐层3 → 输出 | 128 → 12 | ~1.5k |

- 策略网络**总共只有几万参数**，每个 Linear 层的矩阵乘（48×128、128×128）在 L20（48GB、几百 TFLOPS）上**微秒级就完成**。
- 所以 kernel 的瓶颈不是"算不完"，而是"**数据没喂饱**"——等待从显存/缓存读权重、读 batch 的时间比实际计算长。
- 这**不是 bug，是这类小网络 RL 训练的固有特征**：控制任务用的小 MLP 天然算力需求低，GPU 大部分时间在等数据和跑物理仿真。

### 4.3 一句话总结结果

> 细看每一步资源消耗，你会看到：**GPU 计算利用率很低、内存延迟是主要瓶颈、时间被 PhysX 物理仿真和碎片化小 kernel 瓜分**。这是四足 RL 小网络训练的正常画像，说明瓶颈在"物理仿真 + 采样"而非"神经网络计算"——因此扩大 `--num_envs`（更多并行环境）比换更大 GPU 更能提升吞吐。

---

## 5. 常见问题

### Q1: `ncu` 报告 "Profiling is not supported on this device"？

L20（compute 8.9）完全支持 `ncu`，本环境实测可用。若报错，检查 CUDA driver 与 `ncu` 版本是否匹配（本机驱动 580.126 + ncu 2025.1.1 已对齐）。

### Q2: `ncu` 让训练变得极慢，正常吗？

**正常**。`ncu` 会重放采样每个匹配的 kernel，慢 100~1000 倍是预期。所以务必用 `--launch-count` 限制数量、`--kernel-name` 过滤范围，只采样几十个 kernel，别跑完整训练。

### Q3: 想监控 GPU 持续负载，用什么？

用 `run.sh` 本身即可——它启动真实训练占用 GPU，并打印训练前后的 `nvidia-smi` 快照。配合 `nvtop` 或 DCGM 可持续观测。`ncu`/`nsys` 是"快照式"分析，不是"持续监控"。

### Q4: GUI 工具（nsys-ui / ncu-ui）用不了？

headless 服务器无法开 GUI。把生成的 `.nsys-rep` / `.ncu-rep` 报告文件拷到有 GUI 的机器上打开即可。

---

## 6. 命令速查

```bash
cd /data/workdir/embodiment-rl-demo

# ① 全局时间线（先做这个，开销小）
nsys profile --trace=cuda,nvtx,osrt \
  --output=/tmp/go2_timeline --force-overwrite=true \
  bash run.sh flat --max_iterations 5

# ② 看时间线摘要
nsys stats /tmp/go2_timeline.nsys-rep

# ③ 深挖 PPO 学习的矩阵乘 kernel（过滤 PhysX）
ncu --launch-skip 200 --launch-count 10 --set full \
  --kernel-name "regex:.*(gemm|mma|elementwise|softmax).*" \
  bash run.sh flat --max_iterations 3

# ④ 先看 kernel 名字再精准过滤
ncu --launch-skip 200 --launch-count 5 --set basic \
  bash run.sh flat --max_iterations 2
```
