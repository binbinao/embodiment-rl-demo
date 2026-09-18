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

### 2.3 本机（xrdp 远程桌面）的限制 ⚠️

本机通过 **xrdp（RDP）远程桌面**访问，有完整 GUI 桌面（`Xorg :10` + `gnome-shell`）。

但 **`ncu-ui` GUI 仍无法启动**，实测报错：

```
qt.glx: qglx_findConfig: Failed to finding matching FBConfig ...
Could not initialize GLX
Application could not be initialized!
```

**根本原因**：`ncu-ui` 是 Qt 6 应用，需要 **OpenGL 2.0+ 的硬件 GLX 上下文**。而 xrdp 的 Xorg 后端（xorgxrdp）只提供**旧版软件 GLX**，找不到匹配的 FBConfig。这不是"没 GUI"，而是"**有桌面但缺 GPU 加速的 OpenGL**"——xrdp 的经典限制。

> 即使用 `QT_OPENGL=software LIBGL_ALWAYS_SOFTWARE=1` 强制软件渲染，实测仍报同样的 GLX 错误。

### 2.4 在本机用 GUI 的实用方案

| 方案 | 做法 | 评价 |
|---|---|---|
| **① 本地 GUI 打开报告（推荐）** | 服务器 `ncu --export report.ncu-rep` → `scp` 拷回 Mac/Windows → 本地装 Nsight Compute GUI 打开 | 官方标准工作流，报告跨平台自包含，最可靠 |
| **② 修 xrdp 的 GLX** | 改 `/etc/xrdp/xorg.conf` 启用 glamor + 驱动 | 深坑，xrdp glamor 支持不完整，成功率低，不推荐 |

**推荐方案 ①**：Nsight Compute 官方提供 macOS / Windows / Linux 三种 GUI 安装包，在 Mac 或 Windows 本地装一个，服务器上只用 CLI 导出报告。

---

## 3. CLI 工具（ncu）—— 本机 xrdp 桌面的首选

**业务目的**：在本机 xrdp 桌面（GLX 软件版）上，`ncu-ui` GUI 起不来，`ncu` CLI 是能直接用的 Nsight Compute 工具。它不需要 GUI，采样结果以文本/CSV 输出。

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

## 4. 典型工作流（本机 xrdp 标准流程）

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

**核心思想**：**在服务器上跑 CLI 采样（不依赖 GLX），在本地（Mac/Windows）跑 GUI 分析（本地有完整 GPU/OpenGL）**。两者通过 `.ncu-rep` 报告文件解耦。

---

## 5. 常见坑

| 问题 | 原因 | 解决 |
|---|---|---|
| `ncu-ui` 报 `Could not initialize GLX` | xrdp 的 Xorg 后端是软件 GLX，缺 Qt6 需要的 OpenGL 2.0+ 硬件上下文 | 拷 `.ncu-rep` 报告到本地 GUI 打开，或改用 CLI `ncu` |
| `ncu` 报权限不足（`ERR_NVGPUCTRPERM`） | 未启用 perf counter 权限 | 关闭 MIG、或用 `sudo ncu`，生产环境看 [NVIDIA 官方权限文档](https://developer.nvidia.com/err-nvgpuctrperm) |
| `ncu` 让程序极慢 | 每个 kernel 重放采样 | 用 `--launch-count` 限制、`--kernel-name` 过滤 |
| 找不到 `ncu-ui` | 只装了 CLI（CUDA toolkit 部分安装） | 用 `ncu` 导出报告，GUI 分析放到有完整安装的机器 |

---

## 6. 一句话总结

> Ubuntu 上 Nsight Compute 有两个形态：**GUI `ncu-ui`（看报告用，需 GPU 加速的 OpenGL）** 和 **CLI `ncu`（采样用，任何环境可用）**。
> 本机是 xrdp 远程桌面，有桌面但 GLX 是软件版，`ncu-ui` 起不来。所以**采样用 `ncu` CLI（服务器上），分析用本地 GUI（Mac/Windows 装 Nsight Compute）打开 `.ncu-rep` 报告**——两者靠报告文件解耦。

---

*详见 [NSIGHT_COMPUTE_GUIDE.md](./NSIGHT_COMPUTE_GUIDE.md) 了解本项目 Go2 训练的具体 profiling 用法与结果解读。*
