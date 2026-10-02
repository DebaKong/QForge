# 容器化部署（可选方式）

本机默认走**原生安装**（README「快速开始」，3 步：装驱动 → `install.ps1` → `QForge.bat`）。
需要在服务器上跑、或想要环境隔离时，用容器：

## 一条命令

```bash
docker compose up -d --build
# 打开 http://127.0.0.1:8000
```

三个服务：

| 服务 | 镜像 | 作用 | 备注 |
| --- | --- | --- | --- |
| `redis` | `redis:7.4-alpine` | 任务队列 | 已实测（Redis 7.4.11） |
| `api` | `qforge-api`（**981 MB**） | Web API + 前端界面 | 只装核心依赖，**不需要 GPU**；前端在镜像内构建，运行期不需要 Node |
| `worker` | `qforge-worker`（约 9 GB） | 量化 / Engine 构建 / 代码生成 / 编译 / 运行验证 | 额外装 GPU 依赖（TensorRT 运行库，Linux 版单包约 3.7 GB）；需要 NVIDIA 显卡 |

### 为什么是两个镜像

依赖边界写在 `pyproject.toml` 里：

| 组 | 内容 | 谁用 |
| --- | --- | --- |
| 核心 `dependencies` | Web/队列/校验/预处理/代码生成/构建链 | **api 与 worker 都用** |
| 额外项 `gpu` | `tensorrt` + `cuda-python` | **只有 worker 用**（构建 Engine 需要） |

API 只是接收请求、入库、派发任务，不碰 TensorRT —— 所以 api 镜像 0.98 GB，而不是 9 GB。
只跑 API（例如放在没有显卡的机器上做前端/接口）时这样最省资源。

只想让**本机原生安装**的应用连上 Redis：

```bash
docker compose up -d redis
```

## 数据与路径（两个容易踩的坑）

- `qforge-data` 卷挂载到 **`/data`**：任务目录与 SQLite 数据库都在 `/data/storage`。
- 前端静态文件放在**镜像内** `/opt/qforge/web`，并由 `QFORGE_FRONTEND_DIR` 指定。
  它**不能**放在 `/data` 下——卷挂载会遮蔽镜像里同路径的内容，界面会变成说明页。

## 国内网络加速

`.env` 里设置一次，compose 会把它作为构建参数传给镜像（默认走官方 PyPI）：

```ini
QFORGE_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

## GPU

`worker` 需要显卡构建 Engine，compose 已声明 NVIDIA 设备预留。先确认本机 Docker 支持 GPU：

```bash
docker run --rm --gpus all nvcr.io/nvidia/tensorrt:26.03-py3 nvidia-smi
```

若本机 Docker 没有 GPU 支持：

- `api` 仍可启动（上传、查看任务、下载产物都可用）；
- `worker` 启动会失败 → 要么修好 GPU 支持，要么改用原生安装（README「快速开始」）。

## 镜像里有什么、没有什么（诚实标注）

- **有**：应用本体（`qforge` 命令）、前端界面、TensorRT **运行库**（pip 提供）、cuda-python。
- **没有**：CUDA/TensorRT **开发包**（头文件与导入库）、MSVC、CMake/Ninja。
  → 因此容器里**不能编译生成的 C++ 工程**，该步骤会如实标为 `cpp_verification: BLOCKED`；
  需要这一步就用原生安装（README 的「可选能力」）。

## 别和「生成的容器镜像」混淆

| | 是什么 | 在哪 |
| --- | --- | --- |
| 本文的镜像 | **平台自身**的运行环境（api + worker） | 仓库根 `Dockerfile` |
| 生成的镜像 | 每个任务的**用户推理程序**镜像（SPEC 12.2） | 任务产物里的 `docker/Dockerfile` |

## 常用命令

```bash
docker compose logs -f api worker      # 看日志
docker compose ps                      # 看状态
docker compose down                    # 停止（保留数据卷）
docker compose down -v                 # 停止并删除数据卷（会丢任务数据）
docker compose exec api qforge doctor  # 在容器里体检
docker compose exec api qforge version
```

## 验证记录（2026-10-02，本机实测）

| 项 | 结果 |
| --- | --- |
| 镜像构建 | 两个目标都成功：`qforge-api` **981 MB**（约 2.5 分钟）、`qforge-worker` 约 **9 GB**（约 22 分钟，其中 Linux 版 TensorRT 运行库单包 3.7 GB） |
| 启动 | `docker compose up -d` → `api`（healthy）/ `worker` / `redis` 三个容器 |
| **精简 api 镜像可用性** | 用 api 镜像单独起容器：`GET /api/health` ok、`GET /` 200、`GET /api/docs` 200；容器内 `qforge doctor` **如实报告** GPU/TensorRT/cuda-python 为「缺失（必需）」——这正是该镜像的定位（不构建 Engine） |
| 容器内 API | 数据目录 `/data/storage`；前端由镜像内 `/opt/qforge/web` 托管 |
| 容器内 worker 看见 GPU | `docker compose exec worker qforge doctor` → RTX 5060 / 驱动 610.88 / CUDA 13.3 / sm_120 / 8.0 GB |
| **容器内构建 Engine（真实任务）** | `python tools/docker_stack_check.py` → 入队 **0.16s** 返回；`worker_id=celery@<容器主机名>`；**9 秒 SUCCESS**；Engine 241 KB（TensorRT 10.16.1.11，构建于 Linux 容器）；精度 MAE **1.85e-5**、余弦 **0.99999996** |
| C++ 编译验证 | `BLOCKED`（镜像内没有 TensorRT/CUDA 开发文件，符合预期，不影响其它阶段） |
| 生成的容器镜像（任务的产物） | `SKIPPED`（`QFORGE_DOCKER_ENABLED=false`，需要时在任务配置里打开） |

已知限制：

1. `worker` 镜像约 9 GB —— Linux 版 TensorRT 运行库本身就 3.7 GB，属于硬成本；
   不需要在本机构建 Engine 时只跑 `api` 即可（981 MB）。
2. `qforge doctor` 的「前端界面」一项在容器里曾误报「未构建」（它没读 `QFORGE_FRONTEND_DIR`），
   已在源码修复并加断言；**镜像需重新构建才会包含该修复**——界面本身一直由 API 正常托管，不受影响。
3. 首次构建会从 PyPI 拉取 3.7 GB 的 TensorRT 包；国内网络建议在 `.env` 里设
   `QFORGE_PIP_INDEX_URL`（见上文）。
