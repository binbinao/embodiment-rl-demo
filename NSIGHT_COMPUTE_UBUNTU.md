# Ubuntu 上 NVIDIA Nsight Compute 应用用法

> 介绍 Nsight Compute 在 Ubuntu 上的两种形态（GUI 应用 + CLI 工具）、各自适用场景、以及 headless 服务器上的正确用法。
> 结合本项目 Go2 RL 训练的实际环境（Ubuntu 24.04 headless + NVIDIA L20）。

---

## 1. 工具全景：两个形态，别混淆

Nsight Compute 在 Ubuntu 上有**两种完全不同的形态**，新手最容易混淆：

| 形态 | 命令 | 本质 | 需要图形界面？ |
|---|---|---|---|
| **GUI 桌面应用** | `ncu-ui` | Qt 图形界面，可视化查看 `.ncu-rep` 报告 | ✅ 需要 |
| **CLI 命令行工具** | `ncu` | 纯命令行，采样并输出文本报告 | ❌ 不需要 |

同样，Nsight Systems 也有对应的一对：GUI `nsys-ui` / CLI `nsys`。

**本机实测状态**：

```
$ which ncu ncu-ui nsys nsys-ui
/usr/local/cuda/bin/ncu        # CLI ✓ 可用
/usr/local/cuda/bin/ncu-ui     # GUI（Qt 应用，需图形环境）
/usr/local/cuda/bin/nsys       # CLI ✓ 可用
/usr/local/cuda/bin/nsys-ui    # GUI（需图形环境）
```

---

## 2. GUI 应用（ncu-ui）的用法与限制

### 2.1 ncu-ui 是什么

`ncu-ui` 是一个 **Qt 桌面应用**，提供：
- 打开 `.ncu-rep` 报告文件，可视化查看 kernel 的 SM 占用率、内存带宽、指令吞吐
- Source/PTX/SASS 反汇编视图
- 多 kernel 对比、roofline 分析图

### 2.2 启动方式

```bash
# 有图形界面的 Ubuntu 桌面
ncu-ui                    # 打开 GUI，然后 File → Open 加载 .ncu-rep
ncu-ui report.ncu-rep     # 直接打开某个报告
```

### 2.3 本机（headless 服务器）的限制 ⚠️

本机是无显示器的服务器：

```bash
$ echo $DISPLAY        # 空
$ echo $WAYLAND_DISPLAY # 空
```

**直接 `ncu-ui` 会失败**（报 `Cannot open display` 或 Qt 无显示错误）。

### 2.4 在 headless 服务器上用 GUI 的三种方案

| 方案 | 命令 | 适用场景 |
|---|---|---|
| **① SSH X11 转发**（本机已配好 X11 转发，`SSH_CONNECTION` 在） | `ssh -X user@server` 后 `ncu-ui` | 本地有 X 客户端（Linux/Mac 装了 XQuartz） |
| **② 拷回本地打开** | `scp report.ncu-rep local:` 后本地装 Nsight Compute 打开 | 最省事，报告文件可跨机器 |
| **③ VNC/虚拟桌面** | 装 xvfb + VNC | 需要完整桌面体验时 |

**推荐方案 ②**：报告文件 `.ncu-rep` 是自包含的，拷到任何有 Nsight Compute 的机器都能打开，不依赖服务器图形环境。

---

## 3. CLI 工具（ncu）—— headless 服务器的首选

**业务目的**：在无图形界面的服务器上，`ncu` 是唯一能直接用的 Nsight Compute 工具。它不需要 GUI，采样结果以文本/CSV 输出。

### 3.1 基本用法

```bash
# 采样一个进程的前 5 个 kernel，输出基础指标
ncu --launch-count 5 --set basic <你的程序>

# 本项目：采样 Go2 训练的 PPO 学习 kernel
cd /data/workdir/embodiment-rl-demo
ncu --launch-skip 200 --launch-count 10 --set full \
  --kernel-name "regex:.*(gemm|mma|elementwise|softmax).*" \
  bash run.sh flat --max_iterations 3
```

### 3.2 关键参数速查

| 参数 | 作用 |
|---|---|
| `--launch-skip N` | 跳过前 N 个 kernel（跳启动噪声） |
| `--launch-count N` | 只采样 N 个 kernel（控制开销） |
| `--set basic/full` | 指标集：basic 快速 / full 完整 |
| `--kernel-name regex:...` | 正则过滤 kernel 名 |
| `--section <名>` | 只采样指定 section（如 `MemoryWorkloadAnalysis`） |
| `--export report.ncu-rep` | 导出报告文件（供 GUI 打开） |
| `--page details` | 输出详细指标到 stdout |

### 3.3 导出报告给 GUI 看

```bash
# CLI 采样，同时导出 .ncu-rep 报告
ncu --launch-skip 200 --launch-count 10 --set full \
  --export /tmp/go2.ncu-rep \
  --kernel-name "regex:.*gemm.*" \
  bash run.sh flat --max_iterations 3
```

生成 `/tmp/go2.ncu-rep`，然后：
- 本地（或 SSH -X）用 `ncu-ui /tmp/go2.ncu-rep` 打开
- 或 `scp` 拷到有 GUI 的机器打开

---

## 4. 典型工作流（headless 服务器标准流程）

```
① CLI 采样（服务器上）
   ncu --export /tmp/x.ncu-rep ... bash run.sh flat
        │
        ▼
② 导出报告文件
   /tmp/x.ncu-rep
        │
        ▼
③ 拷回本地
   scp user@server:/tmp/x.ncu-rep .
        │
        ▼
④ GUI 可视化（本地有图形界面）
   ncu-ui x.ncu-rep
```

**核心思想**：**在服务器上跑 CLI 采样（不需要图形），在本地跑 GUI 分析（需要图形）**。两者通过 `.ncu-rep` 报告文件解耦。

---

## 5. 常见坑

| 问题 | 原因 | 解决 |
|---|---|---|
| `ncu-ui` 报 `Cannot open display` | headless 服务器无 X 服务 | 用 SSH -X、拷报告回本地、或改用 CLI `ncu` |
| `ncu` 报权限不足（`ERR_NVGPUCTRPERM`） | 未启用 perf counter 权限 | 关闭 MIG、或用 `sudo ncu`，生产环境看 [NVIDIA 官方权限文档](https://developer.nvidia.com/err-nvgpuctrperm) |
| `ncu` 让程序极慢 | 每个 kernel 重放采样 | 用 `--launch-count` 限制、`--kernel-name` 过滤 |
| 找不到 `ncu-ui` | 只装了 CLI（CUDA toolkit 部分安装） | 用 `ncu` 导出报告，GUI 分析放到有完整安装的机器 |

---

## 6. 一句话总结

> Ubuntu 上 Nsight Compute 有两个形态：**GUI `ncu-ui`（看报告用，需图形界面）** 和 **CLI `ncu`（采样用，headless 服务器首选）**。
> 本机是 headless 服务器，所以**采样用 `ncu` CLI，分析用本地 `ncu-ui` GUI 打开 `.ncu-rep` 报告**——两者靠报告文件解耦。

---

*详见 [NSIGHT_COMPUTE_GUIDE.md](./NSIGHT_COMPUTE_GUIDE.md) 了解本项目 Go2 训练的具体 profiling 用法与结果解读。*
