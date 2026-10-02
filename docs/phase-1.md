# 阶段 1 交付说明：MVP（2D 目标检测 + TensorRT）

SPEC 17 对阶段 1 的要求：**上传 → 校验 → FP16/INT8 → Engine → C++ → 编译 → 测试 → Docker**。

本文件记录交付内容、关键设计决策（含**设计变更**）、真实执行过的验证证据、未验证项与偏差。

---

## 1. 交付内容

| 模块 | 位置 | 状态 |
| --- | --- | --- |
| 上传接口（ONNX / 校准集 ZIP） | `backend/app/api/routes/uploads.py` | 完成 |
| 不可信输入防护（大小/扩展名/ZIP 炸弹/路径穿越/符号链接） | `backend/app/services/uploads.py` | 完成 |
| ONNX 校验链（Checker → Runtime Load → 图分析） | `backend/app/services/onnx_inspector.py` | 完成 |
| 统一预处理（letterbox/归一化/NCHW，中日文文件名安全） | `backend/app/services/preprocess.py` | 完成 |
| 校准集校验（SPEC 8.3） | `backend/app/services/calibration.py` | 完成 |
| Model Definition 校验 + YOLOv8 Model Adapter | `backend/app/adapters/definition.py`、`adapters/models/yolov8.py` | 完成 |
| TensorRT Backend Adapter（能力检查/构建/推理） | `backend/app/adapters/backends/tensorrt_adapter.py` | 完成 |
| INT8 静态量化（熵校准，显存经 cuda-python） | 同上 + `backends/cuda_memory.py` | 完成 |
| 工具链探测（CMake/Ninja/MSVC）与构建调用 | `backend/app/services/toolchain.py` | 完成 |
| GPU 开发文件探测（TensorRT/CUDA 头文件与导入库） | `backend/app/adapters/backends/tensorrt_dev.py` | 完成 |
| 代码生成（Jinja2 分层模板 + Dockerfile） | `backend/app/codegen/` | 完成 |
| 流水线各阶段（SPEC 13 全链路） | `backend/app/pipeline/` | 完成 |
| 进程内后台执行器（SPEC 3.1） | `backend/app/services/executor.py` | 完成 |
| 产物归档与报告 | `backend/app/pipeline/stages/delivery.py` | 完成 |
| 前端：上传页 / 任务详情实时日志 / 产物下载 | `frontend/src/views/` | 完成 |
| 容器产物：多阶段 Dockerfile + 启动脚本 + 镜像构建/运行 | `backend/app/codegen/docker.py`、`services/docker_build.py` | 完成（镜像已构建，容器内 GPU 推理已验证） |
| 端到端验收测试 | `tests/test_pipeline_e2e.py`、`tests/test_docker.py` | 完成（`-m gpu` / `-m docker`） |

## 2. 关键设计决策

### 2.1 【设计变更】INT8 量化改用 TensorRT 官方校准（原 SPEC 指定 PPQ）

- **问题**：SPEC 4 / 9 指定 PPQ 作为量化工具，但 PPQ 最新版仍是 **0.6.6（2023-03-13，Alpha）**，
  PyPI 元数据只声明支持 **Python 3.6~3.9**，`requires_dist` 依赖无上限的 `protobuf`。
- **实测证据**：本机 `carla_env` 中 `import ppq` 直接失败
  （`Descriptors cannot be created directly`，protobuf 版本冲突）；而现代 `onnx` 要求
  `protobuf>=4.25`，与 PPQ 0.6.6 生成的 pb2 代码所需 `protobuf<=3.20` **无法共存**。
- **影响**：PPQ 与本机 RTX 5060（SM 12.0）所需的 TensorRT 10.16 + CUDA 13 + Python 3.10 组合不兼容。
- **方案与结论（已获用户确认）**：INT8 静态量化改用 **TensorRT 自带 `IInt8EntropyCalibrator2`**。
  交付行为不变（校准集 → 校准 → INT8 Engine → 精度验证），且为 NVIDIA 官方路径。
- **已知限制**：TensorRT 10.1 起 `IInt8Calibrator` 被标记为 deprecated，官方推荐「显式量化（Q/DQ）」。
  当前实现保留该可用路径并显式忽略弃用告警（`adapters/backends/tensorrt_adapter.py` 内有注释），
  报告中记录 `calibrator_api_status`。升级到 Q/DQ 显式量化列入阶段 2 候选。

### 2.2 【容器】镜像内重新编译 + 启动时重建 Engine

三个实测约束决定了容器结构（都是踩出来的，不是推测）：

1. **平台生成的 C++ 工程在 Windows 侧编译成 `.exe`，不能在 Linux 容器里运行**
   → 镜像采用**多阶段构建**，在容器内重新编译同一份源码（生成代码本身跨平台）。
2. **TensorRT 官方镜像只带 TRT 头文件与 CUDA 运行库，不含 `cuda_runtime_api.h`**
   → 平台把自己已探测到的 CUDA 头文件随构建上下文送进容器（`cuda-include/`）；
     CMake 对每个依赖独立解析：TRT 由镜像提供，CUDA 头文件由上下文提供。
3. **TensorRT 计划文件是平台相关的**：Windows 构建的 Engine 在 Linux 容器里加载会报
   `Platform specific tag mismatch`；而 `docker build` 阶段默认**拿不到 GPU**，
   无法在构建期用 trtexec 重建
   → Engine 由生成的 `entrypoint.sh` 在**容器启动时**按需重建（目标机 GPU 可用）；
     平台侧构建的 Engine 与 `engine_metadata.json` 仍作为产物保留与追溯。

### 2.3 长任务执行：进程内后台执行器（不引 Redis）

- SPEC 3.1 要求「Web API 不直接执行长时间量化、编译、Docker 构建任务」。
  阶段 0 的 Celery eager 模式会**同步**执行，使入队请求被长任务阻塞（且前端会超时），不符合 SPEC。
- 实现 `LocalTaskExecutor`：有界线程池 + 并发信号量，`submit()` 立即返回；
  并发已满时以 `RESOURCE_LIMIT_EXCEEDED (429)` 拒绝，任务**不会**卡在 QUEUED。
- 与 Celery 完全共用同一函数 `run_pipeline`，接入 Redis 后只需把 `QFORGE_EXECUTOR_MODE` 改为 `celery`。

### 2.3 编译验证的 `build.cpp_build` 策略（auto / required / skip）

- 编译需要 TensorRT 开发文件（头文件+导入库）与 CUDA 运行时开发文件，这些**不在 pip 包内**。
- `required`：缺文件即 FAILED；`auto`（默认）：标记 `cpp_verification: BLOCKED` 并继续，
  报告中写明缺什么；`skip`：明确跳过。
- 无论哪种模式，Engine 都会用 **TensorRT Python API 真实加载并推理**，并与 onnxruntime FP32
  基准比较，产出 SPEC 9.2 的精度指标 —— 绝不把「没编译」写成「编译通过」。

### 2.4 SQLite 开启 WAL

后台线程写库与 API 请求并发时，SQLite 默认的回滚日志模式会互相阻塞。
连接级启用 `journal_mode=WAL` + `busy_timeout=15000`（`backend/app/db/base.py`）。

### 2.5 存储布局扩展

在 SPEC 14.2 的任务目录之外新增按资源归档的上传目录：
`storage/models/<model_id>/`（上传的 ONNX）与 `storage/datasets/<dataset_id>/images/`（解压后的校准集），
供多个任务复用；任务目录内仍保留模型副本（硬链接，跨卷时退化为复制）使产物自包含。

### 2.6 生成的 C++ 工程不依赖 OpenCV

为让生成的工程在任何装有 TensorRT+CUDA 的机器上都能直接编译运行，输入只支持：
PPM(P6) 图像与裸 float32 张量（平台在验证阶段自动生成）。letterbox、归一化、解码与 NMS
全部在生成代码中实现，参数由同一份 Model Definition 渲染进 `include/qforge/generated_config.h`，
同时输出可读的 `config/model.yaml`（二者同源，不会漂移）。

## 3. 真实执行过的验证

环境：conda env `qforge`（Python 3.10.21）+ TensorRT 10.16.1.11（CUDA 13.0）+ RTX 5060（SM 12.0）+ MSVC 14.29 + CMake 4.4.3 + Ninja 1.13.2。

| 验证项 | 命令 | 结果 |
| --- | --- | --- |
| 默认测试套件（CPU，已排除重型标记） | `python -m pytest -q` | **141 passed, 3 deselected** |
| FP16 端到端（真实 GPU） | `pytest tests/test_pipeline_e2e.py -m gpu -k fp16 -q -s` | **通过**：Engine 52.1 KB；MAE **1.8e-5**、RMSE 2.4e-5、余弦 **1.000000**；C++ 工程编译成功并真实运行 |
| INT8 端到端（真实 GPU） | `pytest tests/test_pipeline_e2e.py -m gpu -k int8 -q -s` | **通过**：Engine ~120 KB；熵校准 4 张图/1 批次；MAE **3.18e-4**、RMSE 4.16e-4、余弦 **0.999991**；检测 16 条 |
| **容器镜像构建与容器内推理** | `pytest tests/test_docker.py -m docker -q -s` | **通过**：基础镜像 `nvcr.io/nvidia/tensorrt:26.03-py3`（16.7 GB，拉取 58 分钟）→ 容器内多阶段编译成功 → `docker build` 成功 → `docker run --gpus all` 用 trtexec 在容器内重建 Engine（5.78 s）→ 程序真实推理 `{"status":"ok","input_shape":[1,3,64,64],"output_shape":[1,84,256],"latency_ms_avg":0.184}` |
| 产物完整性 | 端到端测试断言 | Engine / source / config / docker / report / artifact.zip（43 个文件）齐备，`task.artifact_id` 已指向归档 |
| 前端构建 | `npm run build` | 成功（新增 UploadView；`index` chunk 约 1.06 MB） |

### 3.1 真实 YOLOv8n 模型验证

用真实导出的 `yolov8n.onnx`（12.2 MB、opset 11、PyTorch 2.8 导出、输入 `[1,3,640,640]`、
输出 `[1,84,8400]`、TensorRT 解析出 **311 层**网络）跑完整流水线：

| 精度 | Engine | 构建耗时 | 稳态推理（生成的 C++ 程序） | MAE | 余弦相似度 |
| --- | --- | --- | --- | --- | --- |
| FP16 | 7.9 MB | 29.9 s | **8.06 ms/次** | 0.0090 | 0.999999 |
| INT8（16 张**合成**校准图） | 9.4 MB | 45.6 s | 9.74 ms/次 | 0.3301 | 0.998087 |

复现：`python tools/real_model_check.py <yolov8n.onnx>`（需先启动 API 与 Redis 可选）。

必须说明的四点，避免误读数据：

1. **校准集是平台生成的合成图像**（渐变 + 色块 + 噪声），不是真实场景数据。
   INT8 的 MAE 明显偏大（0.33 vs FP16 的 0.009）与此直接相关 —— 按 SPEC 9.2 的排查顺序，
   **第一步就是换真实校准数据**，而不是先怀疑量化配置或敏感层。真实校准下的 INT8 精度需单独测。
2. 两个时延口径不同：Python 侧数字包含**加载 Engine + 反序列化 + 分配显存**（约 128 ms 的一次性开销），
   C++ 程序报告的是**预热后纯推理**均值（8~10 ms）。两者都如实记录，不做混用比较。
3. 本机该模型 **FP16 略快于 INT8**（8.06 vs 9.74 ms）：模型较小、GPU 很快时，INT8 的量化/反量化开销
   可能抵消收益。**不能默认「INT8 一定更快」**，以实测为准。
4. 检测数均为 0：样例输入是合成图像（无真实目标），因此这里比较的是**原始输出张量**的数值一致性，
   不是检测精度（mAP）。检测质量需用真实图片与标注评估（阶段 2）。

### 3.2 开发过程中发现并修复的真实缺陷（均为运行/测试暴露，非推测）

| # | 缺陷 | 影响 | 修复 |
| --- | --- | --- | --- |
| 1 | 执行器「先提交再登记」竞态 | 任务过快完成时并发位**永久占用**，后续入队全部 429 | 先登记后提交（`services/executor.py`） |
| 2 | `ModelRead.input_spec` 声明为 dict，实际存 list | **上传接口 500**，模型无法上传 | 改为列表（`schemas/model.py`） |
| 3 | Model Adapter 注册未在包导入期完成 | 生产环境会误报「不支持的架构：yolov8」（被测试导入顺序掩盖） | `app/adapters/__init__.py` 导入即注册 |
| 4 | TRT 10 无 `create_config` | Engine 构建直接崩 | 改用 `create_builder_config`（兼容两种命名） |
| 5 | `layer.num_inputs` 是整数而非序列 | 能力检查崩 | 修正为 `int(...)` |
| 6 | 开发文件探测 glob 层数写错 | 文件存在却报「缺失」 | 改为带深度剪枝的 `os.walk` |
| 7 | TRT 导入库名带版本后缀（`nvinfer_10.lib`） | 误判缺失 | 按前缀+数字后缀解析真实文件名 |
| 8 | CUDA 13 的 `crt/host_defines.h` 不在 cudart 包 | 编译 `C1083` 失败 | 探测并追加 `cuda_crt` 分发包的 include 目录 |
| 9 | CMake 字符串中的 Windows 反斜杠 | `Invalid character escape '\q'`，配置失败 | 注入 CMake 的路径统一转正斜杠 |
| 10 | 带引号命令串直接交 `cmd /c` | Python 把内层引号转义成 `\"`，命令**无输出地失败**，日志只剩命令行 | 改为写 `.bat` 脚本再执行（并把 vcvars 输出并入日志） |
| 11 | 进程 PATH 无 conda `Scripts` 目录 | 误判「缺少 cmake/ninja」 | 解释器同目录/`Scripts`/`bin` 兜底查找 |
| 12 | `config.int8_calibrator` 的弃用告警被 pytest 提升为错误 | INT8 路径 `SystemError` | 显式 `catch_warnings` 抑制并记录弃用事实 |
| 13 | 浮点字面量由 double 初始化 | C4305 截断告警 | `|cpp` 过滤器输出 `f` 后缀 |
| 14 | 生成的 Dockerfile 复制 Windows `.exe` | Linux 容器里根本无法运行 | 改为多阶段构建，容器内重新编译（2.2） |
| 15 | 容器内编译缺 `cuda_runtime_api.h` | `fatal error: cuda_runtime_api.h: No such file` | CUDA 头文件随构建上下文提供（2.2） |
| 16 | 生成的 CMakeLists 把 Windows 库名（`nvinfer_10`）当默认值 | 容器内链接失败 `cannot find -lnvinfer_10` | 注入库名必须先能被解析，否则回退自动探测 |
| 17 | 平台构建的 Engine 在 Linux 容器加载失败 | `Platform specific tag mismatch`（计划文件平台相关） | 容器启动时用 trtexec 重建（2.2） |
| 18 | 报告产物登记用了错误的基准目录 | **所有报告都不会出现在产物列表里**（e2e 测试抓到） | 按 `context.reports` 动态登记，且以任务目录为基准 |

## 4. 未验证项与已知限制

| 项 | 原因 | 后续 |
| --- | --- | --- |
| 真实 Redis + Celery worker 进程 | 用户要求暂不接入 Redis | 设 `QFORGE_EXECUTOR_MODE=celery` 后验证 |
| 浏览器交互 | 未做真实点击验证 | 仅验证构建、dev server 与 `/api` 代理 |
| INT8 层间误差统计与混合精度 | SPEC 9.2 将其列为「后续版本」 | 阶段 2 |
| 动态 batch / shape | SPEC 10.1 明确 MVP 固定 shape | 阶段 2 |
| 真实 YOLOv8 权重模型 | 使用结构等价的合成模型验证链路 | 阶段 2 引入真实模型回归集 |
| Dockerfile 之外的部署产物（依赖说明） | 阶段 2 | — |

> 说明：SPEC 18 的 MVP 验收 13 条中，**第 8/9/10 条（C++ 编译、真实推理、Dockerfile 构建）
> 均已在本机真实通过**（第 10 条进一步做到了「容器内 GPU 真实推理」）。
> 仍未覆盖的是上面的限制项（真实 Redis/worker、浏览器交互、层间误差、动态 shape、真实权重模型），
> 这些不计入「已完成」。

## 5. 环境准备（可复现）

本机工具链（除 TensorRT 开发包外均可通过 pip / 公共分发站获得，无需登录）：

```powershell
# 1) Python 环境
conda create -y -n qforge python=3.10 pip
pip install -r backend/requirements.txt -r backend/requirements-dev.txt

# 2) CUDA 运行时开发文件（NVIDIA 公共分发站，无需登录）
#    cuda_cudart（cudart.lib / cudart64_13.dll）、cuda_crt（crt/host_defines.h）、cuda_cccl
#    https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.0.json
#    解压到 D:\qforge-toolchain\cuda\

# 3) TensorRT 开发包（含 include/ 与 lib/，需登录 NVIDIA 开发者账号）
#    https://developer.nvidia.com/tensorrt/download  → TensorRT 10.16.x / Windows / CUDA 13.0 / ZIP
#    解压到 D:\qforge-toolchain\tensorrt\

# 4) CMake 与 Ninja（pip 安装，无需管理员）
pip install cmake ninja
```

CUDA Python 绑定（INT8 校准与推理需要显存指针）：

```powershell
pip install cuda-python==13.4.1
```

容器验证还需要（阶段 1 已实测）：

```powershell
# Docker Desktop（daemon 需运行；本机 Docker Hub 直连不可达，daemon.json 配了国内加速镜像）
docker pull nvcr.io/nvidia/tensorrt:26.03-py3   # 16.7 GB；内含 TensorRT 10.16.0.72 / CUDA 13.2
```

工具链位置可通过 `QFORGE_TENSORRT_ROOT` / `QFORGE_CUDA_ROOT` 覆盖；
默认会在 `QFORGE_TOOLCHAIN_SEARCH_ROOTS`（默认含 `D:/qforge-toolchain`）下自动探测。

## 6. 与 SPEC 的偏差汇总（需知悉，均已记录）

1. **INT8 量化工具**：PPQ → TensorRT `IInt8EntropyCalibrator2`（见 2.1，已确认）。
2. **存储布局**：新增 `storage/models/`、`storage/datasets/`（见 2.5）。
3. **目录结构**：`adapters/` 放在 `backend/app/adapters/` 内（SPEC 附录 A 放在仓库根）；
   `workers/` 未创建 `model_analyzer` 等子包（随各自阶段加入）。
4. **量化阶段对非 INT8 精度**：状态机保持 SPEC 13 的线性推进，QUANTIZING 阶段对 FP32/FP16
   记录「无需量化」并继续，不跳阶段、不伪造结果。
5. **`tasks.artifact_id` 无外键**（阶段 0 决策延续）。
6. **新增错误码**：`UPLOAD_INVALID`、`UPLOAD_TOO_LARGE`、`ARCHIVE_INVALID`、
   `ARCHIVE_BOMB_SUSPECTED`、`BACKEND_UNAVAILABLE`（SPEC 15.1 未覆盖上传与环境不可用场景）。

## 7. 阶段 2 前置条件

1. 确认 Docker 基础镜像 tag（与 TensorRT/CUDA 版本匹配）并实测镜像构建；
2. 引入真实 YOLOv8 权重回归集（含精度基线）；
3. 决策是否接入 Redis/PostgreSQL（当前为进程内执行器 + SQLite WAL）；
4. 评估把 INT8 从弃用的 `IInt8Calibrator` 迁移到显式量化（Q/DQ）；
5. 明确层间误差分析与混合精度策略（SPEC 9.2 后续项）。
