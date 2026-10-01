# 版本矩阵（SPEC 4.1）

SPEC 4.1 要求：版本敏感组件必须锁定并记录**具体版本**，不接受「TensorRT 8.x」这类模糊写法；
每个 Engine 必须保存构建环境元数据。

本文件区分三类：**已锁定并验证**、**已锁定未验证**、**未确定（阶段 1 前必须确认）**。

## 1. 平台与工具链（阶段 0 探测，2026-10-01）

| 项目 | 版本 / 路径 | 状态 |
| --- | --- | --- |
| OS | Windows 11 家庭版（中文版），用户 `JXGM` | 已探测 |
| 命令通道 Shell | Windows PowerShell 5.1 | 已探测 |
| Python | 3.10.21（conda env `qforge`） | 已锁定并验证 |
| conda | 25.5.1（base 为 `D:\Anaconda3`，Python 3.13.5） | 已探测 |
| pip | 26.2.1 | 已探测 |
| pip 索引 | `https://pypi.tuna.tsinghua.edu.cn/simple`（清华镜像） | 已验证可用 |
| Node.js | 24.9.0 | 已探测 |
| npm | 11.16.0 | 已验证可用 |
| Docker CLI | 29.8.0（build 88096ef） | 已安装，**daemon 未运行** |
| Docker Compose | v5.5.1 | 已安装，未验证 |
| WSL | Ubuntu-22.04（Stopped） | 已探测 |
| CMake / make | **未安装** | 缺失 |
| nvcc (CUDA) | **未安装** | 缺失 |
| gcc / g++ | `D:\Visual_Studio_Code_2025\MinGW\bin`（版本未探测） | 存在，用途待确认 |

## 2. Python 依赖（已锁定并验证）

锁定来源：`backend/requirements.txt`、`backend/requirements-dev.txt`；下列为实测安装结果。

| 组件 | 版本 | 阶段 0 用途 |
| --- | --- | --- |
| fastapi | 0.142.2 | API 框架 |
| starlette | 1.7.0 | FastAPI 底层（传递依赖） |
| uvicorn | 0.54.0 | ASGI 服务器 |
| pydantic | 2.13.5 | 请求/响应模型 |
| pydantic-settings | 2.15.0 | 配置加载 |
| SQLAlchemy | 2.0.54 | ORM |
| alembic | 1.20.0 | schema 迁移 |
| celery | 5.6.3 | 任务队列 |
| kombu | 5.6.2 | Celery 消息层（传递依赖） |
| billiard | 4.3.0 | Celery 进程池（传递依赖） |
| redis (Python 客户端) | 8.1.0 | broker 客户端（阶段 0 未连真实 Redis） |
| pytest | 9.1.1 | 测试框架 |
| httpx | 0.28.1 | 测试客户端与运行时验证脚本 |
| anyio | 4.15.1 | 异步兼容层（传递依赖） |
| greenlet | 3.5.6 | SQLAlchemy 依赖 |
| python-dotenv | 1.2.3 | `.env` 加载（传递依赖） |
| watchfiles | 1.3.0 / websockets 16.1.1 / httptools 0.8.0 | uvicorn[standard] 附带的 reload/WS/解析支持 |

### 数据库驱动

| 组件 | 版本 | 状态 |
| --- | --- | --- |
| SQLite | Python 内置（3.10.21 自带） | 已验证（阶段 0 默认） |
| psycopg（PostgreSQL 驱动） | — | **未安装**；阶段 1 接入 PostgreSQL 时确认版本后再引入 |

## 3. 前端依赖（exact 锁定，已验证构建）

来源：`frontend/package.json`（使用 `npm install --save-exact` 写入）。

| 组件 | 版本 |
| --- | --- |
| vue | 3.5.43 |
| vue-router | 5.3.1 |
| element-plus | 2.14.7 |
| axios | 1.20.0 |
| vite | 8.3.1 |
| @vitejs/plugin-vue | 6.0.9 |

> 注意：`vite` 8.x 与 `vue-router` 5.x 均为较新主版本。阶段 0 只验证到「生产构建成功 +
> dev server 返回页面与 `/api` 代理 200」，**未做浏览器交互验证**；若阶段 1 出现行为差异，
> 需按 SPEC 4.1 流程确认后再调整版本，不得擅自升降级。

## 4. 未确定项（阶段 1 前必须确认并写入本文件）

| 组件 | 现状 | 确认方式 |
| --- | --- | --- |
| TensorRT | 未安装，版本未定 | 查阅 NVIDIA 官方 Release Notes 的 Python/CUDA 支持矩阵后锁定 |
| CUDA / cuDNN | 未安装，版本未定 | 与 TensorRT 版本配套锁定 |
| PPQ | 未安装 | 以其官方仓库/文档确认支持的 Python 与依赖版本 |
| CMake / 编译器 | 缺失 / 未确认 | 确定 MSVC 或 MinGW-w64 方案与具体版本（C++ 工程构建必需） |
| Redis 镜像 tag | compose 暂写 `redis:7.4-alpine` | 镜像仓库可达后拉取确认 |
| PostgreSQL 镜像 tag | compose 暂写 `postgres:16-alpine` | 镜像仓库可达后拉取确认 |
| GPU 型号与 SM 版本 | 未探测（`nvidia-smi` 未安装/不可用） | 阶段 1 Engine 元数据需要（SPEC 4.1 / 10.1） |

> 阶段 1 的 Engine 必须写入 `engine_metadata`（backend/backend_version/cuda_version/gpu_arch/
> precision/input_profile/build_time），其取值来源即本文件锁定后的版本矩阵。
