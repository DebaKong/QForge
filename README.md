# QForge

ONNX 自动化量化部署平台 —— Web 端一站式完成 ONNX 模型校验、量化校准、TensorRT Engine 构建、
C++ 推理工程生成、自动编译与运行验证。

- 需求与技术设计规格：[SPEC.md](SPEC.md)
- 协作方式与阶段纪律：[AGENTS.md](AGENTS.md)
- **当前实现阶段：阶段 1（MVP：2D 检测 + TensorRT）** —— 交付与验证证据见 [docs/phase-1.md](docs/phase-1.md)
- 阶段 0 记录：[docs/phase-0.md](docs/phase-0.md)
- **接入 Redis + Celery**：[docs/redis.md](docs/redis.md)（已实测切换：入队 0.30s 返回，任务由 worker 进程执行）
- **容器化部署（可选）**：[docs/docker.md](docs/docker.md)（`docker compose up -d --build` 一条命令跑起 api + worker + redis）
- 安装即用改造的方案与进度：[docs/portable.md](docs/portable.md)
- 版本矩阵（已锁定 / 待确认）：[docs/versions.md](docs/versions.md)

## 阶段状态

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| 阶段 0 | 脚手架与基础架构（Vue / FastAPI / Redis / Celery / 存储 / 数据库 / 任务状态机） | 已完成（Redis 与 PostgreSQL 真实链路待环境就绪） |
| 阶段 1 | MVP：2D 检测 + TensorRT（上传→校验→FP16/INT8→Engine→C++→编译→测试→Docker） | **已完成并实测通过**（FP16/INT8 全链路；Docker 镜像构建因 daemon 未运行未验证） |
| 阶段 2 | V1.0：稳定性与质量（分割、误差分析、算子检测、数据集管理） | 未开始 |
| 阶段 3 | V1.5：3D PoC | 未开始 |
| 阶段 4 | V2.0：多后端（Backend Adapter + OpenVINO） | 未开始 |

### 阶段 1 实测结果（真实 GPU）

以结构等价的合成 YOLOv8 形状模型（输入 64×64）跑完整流水线：

| 精度 | Engine | MAE（对照 FP32 基准） | 余弦相似度 | 生成的 C++ 程序 |
| --- | --- | --- | --- | --- |
| FP16 | 52.1 KB | 1.8e-5 | 1.000000 | 编译成功并真实推理 |
| INT8（熵校准，4 张校准图） | 117.0 KB | 3.18e-4 | 0.999991 | 编译成功并真实推理 |

FP32 基准 = onnxruntime CPU 推理；三种精度不混称（SPEC 9）。

## 目录结构

```text
QForge/
├── AGENTS.md                 协作与阶段纪律
├── SPEC.md                   需求与技术设计规格书 V1.0
├── docker-compose.yml        Redis + PostgreSQL 编排（已编写，未运行验证）
├── pyproject.toml            pytest / ruff 配置与解释器约束
├── backend/
│   ├── alembic.ini           迁移配置（保持 ASCII：Alembic 按 locale 编码读取）
│   ├── migrations/           Alembic 迁移（env.py + versions/）
│   ├── requirements.txt      运行期依赖（已锁定版本）
│   ├── requirements-dev.txt  测试依赖（已锁定版本）
│   └── app/
│       ├── main.py           FastAPI 应用与统一错误出口
│       ├── config/           运行期配置（QFORGE_* 环境变量）
│       ├── db/               SQLAlchemy 引擎/会话
│       ├── models/           ORM 实体（SPEC 14.1 八张表）
│       ├── schemas/          Pydantic 请求/响应模型
│       ├── services/         状态机、存储、任务编排、元数据读写
│       └── api/routes/       REST 路由
├── workers/                  Celery 应用与任务（执行真实流水线）
├── frontend/                 Vue 3 + Element Plus + Vite
├── scripts/                  本地启动脚本（API / worker）
├── tools/                    辅助脚本（合成模型生成、真实模型检查、Redis 切换检查）
├── storage/                  运行时任务目录（storage/<task_id>/...，不入库）
└── tests/                    pytest 用例
```

## 环境要求

- **必需**：Windows 10/11 + NVIDIA 显卡驱动（平台为你的 GPU 构建 Engine）
- **安装脚本会处理**：Python 3.10、依赖包、CUDA 运行时开发文件、数据目录、数据库表
- **可选**：Node.js（构建前端界面）、Docker（容器镜像 / Redis）、Visual Studio + TensorRT 开发包（C++ 编译验证）
- 数据库默认 SQLite（已开启 WAL），无需外部服务
- 版本矩阵与已锁定组件见 [docs/versions.md](docs/versions.md)

> 新机器从零到能用的步骤数：**改造前 12 步 → 现在 3 步**（装驱动 → `install.ps1` → `QForge.bat`）。
> 方案与演进记录见 [docs/portable.md](docs/portable.md)。

## 快速开始（安装即用）

新机器上只需要三步：

**Windows**：

```powershell
# 1) 安装：自动建虚拟环境 → 装依赖 → 下载 CUDA 开发文件（公开源，无需登录）
#          → 初始化数据目录 → 尽力构建前端界面 → 环境体检
powershell -ExecutionPolicy Bypass -File scripts\install.ps1

# 2) 启动：界面 + API + worker 一起起来，并自动打开浏览器
.\QForge.bat
```

**Linux / macOS**：

```bash
bash scripts/install.sh     # 同上，自动找 Python 3.10、建 .venv、装依赖、初始化
./QForge.sh                 # 启动并打开浏览器
```

也可以直接用 `qforge` 命令：`qforge serve --open`。

**不需要 Redis**：没有 Redis 时自动使用进程内执行器；若本机有 Redis（例如
`docker compose up -d redis`），启动时自动切换为队列模式并带起 worker。

**运行期不需要 Node.js**：已构建的前端由 API 直接托管（`install.ps1` 会在检测到 Node 时构建，
没有 Node 则界面显示一张说明页，API 全部可用）。

唯一无法自动化的前提是 **NVIDIA 显卡驱动**（平台要为你的 GPU 构建 Engine）。
用 `qforge doctor` 可以看到本机还缺什么、以及每条缺失项的安装命令。

### 常用命令

| 命令 | 作用 |
| --- | --- |
| `qforge doctor` | 环境体检：逐项给出「有什么/缺什么/怎么补」 |
| `qforge serve --open` | 启动服务（自动选择执行方式、自动带起 worker、自动开浏览器） |
| `qforge init` | 建数据目录并**应用数据库迁移**（首次 `serve` 也会自动完成；升级版本后再跑一次即可升级表结构） |
| `qforge build-frontend` | 构建前端界面（需要 Node.js；安装形态用 `--source <目录> --publish`） |
| `qforge fetch-cuda-headers` | 自动下载 CUDA 运行时开发文件（无需登录；`--dry-run` 只看不下载） |
| `qforge worker` | 单独启动 Celery worker |
| `qforge version` | 版本与关键路径 |

想让它像普通软件一样有图标：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create-shortcut.ps1
# 桌面 + 开始菜单各创建一个 QForge 快捷方式；撤销用 -Remove
```

### 依赖边界（决定装多大）

| 组 | 内容 | 谁需要 |
| --- | --- | --- |
| 核心（`pip install .`） | Web/队列/校验/预处理/代码生成 | **所有人**（约 300 MB，不需要 GPU） |
| GPU 额外项（`pip install ".[gpu]"`） | TensorRT 运行库 + cuda-python | **在本机构建 Engine 的人**（本机默认装它，约 1.5 GB） |
| 开发额外项（`pip install ".[dev]"`） | pytest / httpx | 跑测试的人 |

`scripts\install.ps1` 默认执行 `.[gpu]`（即完整安装）；只想跑 API/界面时用 `-NoGpu`。

### 可选能力（缺了也能跑核心流程）

| 能力 | 额外需要 | 缺了会怎样 |
| --- | --- | --- |
| 编译「生成的 C++ 工程」 | TensorRT **开发包**（需 NVIDIA 账号登录）+ MSVC | 报告里标 `cpp_verification: BLOCKED`，其余流程照常 |
| 生成容器镜像 | Docker Desktop + TensorRT 基础镜像（16.7 GB） | 任务配置 `build.docker_build=true` 时才需要 |
| Redis 队列 / 独立 worker | Docker 或本机 Redis | 自动退回进程内执行器，功能不变 |
| 前端热更新开发 | Node.js | 只用构建好的界面即可 |

### 从源码开发（可选，与原流程一致）

```powershell
conda create -y -n qforge python=3.10 pip
conda activate qforge
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
pip install ".[gpu]"          # 构建 Engine 需要（GPU 额外项）
alembic -c backend/alembic.ini upgrade head
```

前端开发服务器：`cd frontend; npm install; npm run dev`（http://127.0.0.1:5173，`/api` 代理到 8000）。

### 运行测试

```powershell
# 必须在仓库根执行（pytest 的 pythonpath 由 pyproject.toml 提供）
python -m pytest -q                 # 默认套件：143 项，纯 CPU，不需要 GPU
python -m pytest -m gpu -q -s       # 端到端：真实构建 Engine、编译 C++ 并推理（需要上面的工具链）
python -m pytest -m docker -q -s    # 容器：真实 docker build + 容器内推理（需要 Docker daemon 与基础镜像）
```

### 界面回归检查（真实浏览器点击）

后端测试全绿也可能界面点不动（上传没反应、列表点「查看」不跳转），所以要真的点一遍：

```powershell
# 需要 Playwright（只在跑这个脚本时用；默认驱动系统已装的 Edge，不用下载浏览器）
npm install --no-save playwright
node tools/ui_check.mjs <某个.onnx路径> http://127.0.0.1:8000
```

覆盖：列表→详情自动跳转、浏览器返回、详情页显示解析结果、选项目+选文件+上传成功。

## 配置

复制 [.env.example](.env.example) 为 `.env` 后修改（`.env` 不入库）。所有变量使用 `QFORGE_` 前缀：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `QFORGE_ENVIRONMENT` | `dev` | `dev` / `test` / `prod`；`prod` 时日志输出单行 JSON |
| `QFORGE_STORAGE_ROOT` | `<repo>/storage` | 任务存储根目录（SPEC 14.2） |
| `QFORGE_DATABASE_URL` | 空 → `storage/qforge.db` | 留空即 SQLite；阶段 1 切 `postgresql+psycopg://...` |
| `QFORGE_AUTO_CREATE_SCHEMA` | `true` | 启动时按元数据建表（本地便捷）；正式 schema 变更走 Alembic |
| `QFORGE_EXECUTOR_MODE` | `local` | `local`（API 进程内后台线程）或 `celery`（投递到 Redis，本机已启用） |
| `QFORGE_CELERY_TASK_ALWAYS_EAGER` | `true` | 必须为 `false` 才真正走队列；本机 `.env` 已设为 `false` |
| `QFORGE_CELERY_BROKER_URL` | `redis://127.0.0.1:6379/0` | 任务队列 |
| `QFORGE_CELERY_RESULT_BACKEND` | `redis://127.0.0.1:6379/1` | 结果后端 |
| `QFORGE_MAX_UPLOAD_MB` / `QFORGE_MAX_ZIP_*` | 见 .env.example | SPEC 15 资源与安全限制（阶段 1 起强制执行） |
| `QFORGE_TASK_TIMEOUT_SECONDS` | `3600` | 任务超时上限（已用于 Celery `task_time_limit`） |
| `QFORGE_EXECUTOR_MODE` | `local` | `local`（进程内后台执行器）或 `celery`（接入 Redis 后） |
| `QFORGE_DOCKER_BASE_IMAGE` | `nvcr.io/nvidia/tensorrt:26.03-py3` | 容器基础镜像（含 TensorRT 10.16 / CUDA 13.2） |
| `QFORGE_DOCKER_ENABLED` | `false` | 是否默认构建容器镜像（SPEC 2.1 列为可选；也可由任务配置 `build.docker_build` 打开） |
| `QFORGE_TENSORRT_ROOT` / `QFORGE_CUDA_ROOT` | 空（自动探测） | TensorRT / CUDA 开发文件所在目录 |
| `QFORGE_MAX_CONCURRENT_TASKS` | `1` | 并发上限（阶段 1 起强制执行） |

## API

| 方法 | 路径 | 行为 |
| --- | --- | --- |
| GET | `/api/health` | 应用、Python、数据库方言、存储根、阶段标记 |
| POST / GET | `/api/projects`、`/api/projects/{id}` | 项目登记与查询 |
| **POST** | **`/api/models/upload`** | **上传 ONNX（multipart）→ 大小/扩展名校验 → ONNX 校验链 → 落 Model Definition** |
| POST / GET | `/api/models`、`/api/models/{id}` | 模型元数据登记与查询（元数据方式，不含文件） |
| **POST** | **`/api/datasets/upload`** | **上传 calibration.zip → 逐条安全解压（防路径穿越 / ZIP 炸弹 / 符号链接）→ 统计图像** |
| POST / GET | `/api/datasets`、`/api/datasets/{id}` | 数据集元数据登记与查询 |
| POST / GET | `/api/tasks`、`/api/tasks/{id}` | 创建任务（状态 + TaskConfig 快照 + 存储目录）与查询 |
| GET | `/api/tasks/{id}/config` | 任务配置快照 |
| GET | `/api/tasks/{id}/transitions` | 当前状态与允许的下一步（前端不复制状态机） |
| POST | `/api/tasks/{id}/enqueue` | `CREATED -> QUEUED` 并投递执行（进程内执行器或 Celery） |
| POST | `/api/tasks/{id}/cancel` | 非终态 → `CANCELLED` |
| GET | `/api/tasks/{id}/logs` | 阶段日志（前端轮询展示） |
| GET | `/api/tasks/{id}/artifacts` | 产物列表（Engine / 源码 / 报告 / artifact.zip） |
| GET | `/api/tasks/{id}/artifacts/{artifact_id}/download` | 下载单个产物（路径受限在存储根内） |
| GET | `/api/tasks/{id}/report` | 精度验证指标（ENGINE 与 FP32 基准的 MAE/RMSE/余弦）；无结果时 501 |

错误响应统一为 `{"error_code", "message", "detail"}`，错误码见 SPEC 15.1 与 `backend/app/errors.py`。

## 任务状态机

`CREATED → QUEUED → VALIDATING → PREPROCESSING → QUANTIZING → BUILDING_ENGINE → GENERATING_CODE
→ BUILDING → TESTING → PACKAGING → SUCCESS`；任意非终态可进入 `FAILED` 或 `CANCELLED`；终态封闭。
不允许跳级、回退与自环。唯一写入口是 `backend/app/services/task_service.py` 的 `transition()`。

阶段 0 的 Worker 为**占位实现**：它登记领取记录并写日志，但**不推进阶段、不伪造 SUCCESS**，
因此入队后任务停留在 `QUEUED`，等待阶段 1 接续。

## 已知环境缺口

- **Docker 镜像构建未验证**：daemon 未运行，且 Dockerfile 的基础镜像 tag 必须与 TensorRT/CUDA 匹配（模板中为 `REPLACE_WITH_CONFIRMED_TAG` 占位），需人工确认后实测
- **Redis / PostgreSQL 未接入**（按要求暂缓）：当前用进程内后台执行器 + SQLite（已开 WAL）
- **真实 YOLOv8 权重模型未纳入回归**：阶段 1 用结构等价的合成模型验证全链路
- **INT8 使用的是已被 TensorRT 标记为 deprecated 的 `IInt8Calibrator`**：可用但官方推荐「显式量化（Q/DQ）」，已记录待阶段 2 评估
- 前端仅验证了生产构建、dev server 与 `/api` 代理，**未做浏览器交互验证**

## 阶段 1 补充环境准备（GPU 工具链）

阶段 1 的 Engine 构建 / C++ 编译 / 真实推理需要额外工具链，详见 [docs/phase-1.md](docs/phase-1.md) 第 5 节：

```powershell
pip install cmake ninja cuda-python==13.4.1        # CMake/Ninja/显存绑定（无需管理员）
# CUDA 运行时开发文件：NVIDIA 公共分发站（无需登录）
#   https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.0.json
#   取 cuda_cudart / cuda_crt / cuda_cccl 三个 windows-x86_64 包，解压到 D:\qforge-toolchain\cuda\
# TensorRT 开发包（需 NVIDIA 开发者账号）：https://developer.nvidia.com/tensorrt/download
#   TensorRT 10.16.x / Windows / CUDA 13.0 / ZIP → 解压到 D:\qforge-toolchain\tensorrt\
```

编译还需要 Visual Studio 的 C++ 工具集（本机为 VS Professional 2019，`D:\vs2019`），
平台会自动通过 vswhere 或常见路径找到 `vcvars64.bat`；也可用 `QFORGE_VCVARS_PATH` 显式指定。

工具链缺失时任务不会伪成功：`build.cpp_build=auto`（默认）会把
`cpp_verification` 标记为 **BLOCKED** 并写入缺少什么；设为 `required` 则直接 FAILED。
