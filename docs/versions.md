# 版本矩阵（SPEC 4.1）

SPEC 4.1 要求：版本敏感组件必须锁定并记录**具体版本**，不接受「TensorRT 8.x」这类模糊写法；
每个 Engine 必须保存构建环境元数据。

本文件区分三类：**已锁定并验证**、**已锁定未验证**、**未确定（进入下一阶段前必须确认）**。
阶段 1 现状：GPU 工具链已在本机真实构建 Engine、编译 C++ 并运行推理（见 [phase-1.md](phase-1.md)）。

## 1. 平台与工具链（阶段 1 实测，2026-10-01）

| 项目 | 版本 / 路径 | 状态 |
| --- | --- | --- |
| OS | Windows 11 家庭版（中文版），用户 `JXGM` | 已探测 |
| 命令通道 Shell | Windows PowerShell 5.1 | 已探测 |
| Python | **3.10.21**（conda env `qforge`） | 已锁定并验证 |
| conda | 25.5.1（base 为 `D:\Anaconda3`，Python 3.13.5） | 已探测 |
| pip | 26.2.1（索引：清华镜像 `pypi.tuna.tsinghua.edu.cn`） | 已验证 |
| Node.js / npm | 24.9.0 / 11.16.0 | 已验证 |
| **GPU** | NVIDIA GeForce RTX 5060 Laptop GPU，**compute capability 12.0（SM 12.0）**，显存 8 GB | 已探测 |
| **GPU 驱动** | 610.88（`nvidia-smi` 报告 CUDA 13.3） | 已探测 |
| **TensorRT** | **10.16.1.11**（pip `tensorrt-cu13` + 官方 Windows 开发包 10.16.1.11） | 已锁定并验证 |
| **CUDA 运行时** | **13.0.48**（官方 redist `cuda_cudart` / `cuda_crt` / `cuda_cccl`） | 已锁定并验证 |
| CUDA Toolkit（完整） | **未安装**（不需要：生成工程不含 CUDA kernel，`nvcc` 非必需） | 不需要 |
| **C++ 编译器** | MSVC 19.29.30159（Visual Studio Professional 2019，`D:\vs2019`） | 已锁定并验证 |
| Windows SDK | 10.0.19041.0 | 已探测 |
| **CMake** | 4.4.3（pip 安装，`<env>\Scripts\cmake.exe`） | 已锁定并验证 |
| **Ninja** | 1.13.2（pip 安装） | 已锁定并验证 |
| 工具链根目录 | `D:\qforge-toolchain\{tensorrt,cuda}`（自动探测，可用环境变量覆盖） | 已验证 |
| Docker CLI / Compose | 29.8.0 / v5.5.1 | daemon 已验证可用（server 29.8.1，linux/x86_64） |
| Docker 镜像加速 | `https://docker.m.daocloud.io`（daemon.json 已配置） | 已验证（Docker Hub 直连不可达，经镜像可拉取） |
| **容器基础镜像** | **`nvcr.io/nvidia/tensorrt:26.03-py3`** → TensorRT **10.16.0.72** / CUDA **13.2.0** | 已核对镜像配置元数据（与 Engine 构建环境同一 TensorRT 小版本线） |
| WSL | Ubuntu-22.04（Stopped） | 已探测 |
| gcc / g++（MinGW） | `D:\Visual_Studio_Code_2025\MinGW\bin` | 存在但**不使用**（TRT 官方库为 MSVC ABI） |

## 2. Python 依赖（已锁定并验证）

锁定来源：`backend/requirements.txt`、`backend/requirements-dev.txt`；下表为实测安装结果。

| 组件 | 版本 | 用途 |
| --- | --- | --- |
| fastapi | 0.142.2 | API 框架 |
| starlette | 1.7.0 | FastAPI 底层 |
| uvicorn | 0.54.0 | ASGI 服务器 |
| pydantic | 2.13.5 | 请求/响应模型 |
| pydantic-settings | 2.15.0 | 配置加载 |
| SQLAlchemy | 2.0.54 | ORM |
| alembic | 1.20.0 | schema 迁移 |
| celery | 5.6.3 | 任务队列（阶段 1 保留接线，默认走进程内执行器） |
| kombu | 5.6.2 / billiard 4.3.0 | Celery 传递依赖 |
| redis（Python 客户端） | 8.1.0 | broker 客户端（未连真实 Redis） |
| **onnx** | **1.23.1** | ONNX 解析、Checker |
| **onnxruntime** | **1.23.2** | Runtime Load Test + **FP32 精度基准** |
| **numpy** | **2.2.6** | 张量计算 |
| **opencv-python-headless** | **5.0.0.93** | 图像缩放/letterbox |
| **pillow** | **12.3.0** | 图像读取（含 EXIF 方向）与 PPM 输出 |
| **Jinja2** | **3.1.6** | 代码生成模板 |
| **PyYAML** | **6.0.3** | 配置读写 |
| **python-multipart** | **0.0.32** | 文件上传 |
| **psutil** | **7.2.2** | 资源信息 |
| **cmake** | **4.4.3** | 构建生成的 C++ 工程 |
| **ninja** | **1.13.2** | 构建后端 |
| **cuda-python** | **13.4.1**（cuda-bindings 13.4.3） | INT8 校准与推理的显存操作 |
| pytest | 9.1.1 | 测试框架 |
| httpx | 0.28.1 | 测试客户端与验证脚本 |
| protobuf | 7.36.2 | onnx 依赖（**PPQ 0.6.6 与此不兼容**，见 phase-1.md 设计变更） |

### 数据库驱动

| 组件 | 版本 | 状态 |
| --- | --- | --- |
| SQLite | Python 内置（3.10.21） | 已验证（当前默认，已开启 WAL） |
| psycopg（PostgreSQL） | — | **未安装**；接入 PostgreSQL 时先确认版本 |

### 明确未采用

| 组件 | 原因 |
| --- | --- |
| PPQ 0.6.6 | 2023 年 Alpha 版，只支持 Python 3.6~3.9，且与现代 protobuf/onnx 无法共存（实测 `import ppq` 失败） |
| torch | 原为 PPQ 依赖；改用 TensorRT 官方校准后**不需要**，故未安装 |

## 3. 前端依赖（exact 锁定，已验证构建）

来源：`frontend/package.json`（`npm install --save-exact` 写入）。

| 组件 | 版本 |
| --- | --- |
| vue | 3.5.43 |
| vue-router | 5.3.1 |
| element-plus | 2.14.7 |
| axios | 1.20.0 |
| vite | 8.3.1 |
| @vitejs/plugin-vue | 6.0.9 |

## 4. 未确定项（进入阶段 2 前必须确认）

| 组件 | 现状 | 确认方式 |
| --- | --- | --- |
| Docker 基础镜像 tag | Dockerfile 中为 `REPLACE_WITH_CONFIRMED_TAG` 占位 | 选择与 TensorRT 10.16 / CUDA 13 匹配的镜像并实测构建 |
| Redis / PostgreSQL | 未接入（用户要求暂缓） | 接入前确认镜像与驱动版本 |
| 真实 YOLOv8 权重模型 | 当前用结构等价的合成模型验证链路 | 阶段 2 引入回归模型与精度基线 |
| INT8 显式量化（Q/DQ） | 当前使用已弃用的 `IInt8Calibrator` | 阶段 2 评估迁移 |

## 5. Engine 元数据（已落地）

每个 Engine 构建时写入 `engine_metadata.json`，字段与实测值示例：

```json
{
  "backend": "tensorrt",
  "backend_version": "10.16.1.11",
  "cuda_version": "13.0",
  "cuda_driver_version": "13.3",
  "gpu_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
  "gpu_arch": "sm_120",
  "precision": "INT8",
  "input_profile": {"min": [1,3,64,64], "opt": [1,3,64,64], "max": [1,3,64,64]},
  "build_time": "2026-10-01T09:38:52+00:00",
  "build_duration_seconds": 6.1,
  "workspace_bytes": 2147483648,
  "engine_size_bytes": 119808,
  "calibration": {
    "method": "IInt8EntropyCalibrator2",
    "calibrator_api_status": "deprecated since TensorRT 10.1（推荐显式量化）",
    "images": 4, "batches": 1, "batch_size": 4, "cache_file": "calibration.cache"
  },
  "engine_file": "model.engine"
}
```
