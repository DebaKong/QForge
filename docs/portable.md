# 安装即用改造：现状基线、方案与落地清单

> 本文对应 AGENTS.md 的「产品形态要求（长期有效）」。用户要求：项目要像**应用程序**一样，
> 在一台新电脑上**安装即用**，不要一长串手工配置。本文给出实测基线、可选方案与推荐落地清单。

## 0. 现状基线：新机器从零到能用需要 12 步

| # | 步骤 | 能否自动化 | 体积 | 必须人工/登录？ |
| --- | --- | --- | --- | --- |
| 1 | 安装 Python 3.10（或 conda） | 可（winget / 便携版） | ~100 MB | 否 |
| 2 | 建虚拟环境并 `pip install` 后端依赖（含 tensorrt、cuda-python、cmake、ninja） | 可 | ~1.5 GB | 否 |
| 3 | 下载解压 CUDA 运行时开发文件（cudart / crt / cccl） | 可（NVIDIA 公共分发站，**无需登录**） | 25 MB | 否 |
| 4 | 下载解压 TensorRT Windows 开发包 | 否 | 2.3 GB | **需 NVIDIA 账号登录** |
| 5 | 安装 Visual Studio（C++ 工具集） | 可（winget） | 数 GB | 需人工确认 |
| 6 | 安装 Node.js + `npm install` + 构建前端 | 可 | ~400 MB | 否 |
| 7 | 安装 Docker Desktop（Redis / 容器构建） | 否 | ~2 GB | **需人工安装并重启** |
| 8 | 写 `.env` | 可（自动生成默认值） | — | 否 |
| 9 | 初始化数据库 | 可 | — | 否 |
| 10–11 | 启动 API、启动 worker | 可（一条命令） | — | 否 |
| 12 | 打开浏览器 | 可（自动打开） | — | 否 |

**结论：12 步，其中 4 步要下载数 GB、1 步需要登录账号、2 步必须人工交互。**

## 1. 三个决定性事实（决定了能简化到什么程度）

1. **核心流程不需要登录墙**：上传 → 校验 → 量化 → Engine 构建 → 真实推理 → 精度对比 → 打包，
   全部只用 `pip install tensorrt` 提供的库即可完成（阶段 1 已实测：Python 侧 Engine 构建 + 推理 + 精度对比均通过）。
   TensorRT **开发包**（头文件 + 导入库）**只有「编译生成的 C++ 工程」这一步才需要**。
2. **容器路线可用但重**：容器内 GPU 真实推理已验证通过，但要 16.7 GB 基础镜像，且容器内产物是 Linux 版。
   适合服务器/隔离部署，不适合当"最省事的默认路径"。
3. **无法消除的只有一条**：平台是"为用户的 GPU 构建 Engine"的工具，因此目标机必须有
   **NVIDIA 显卡 + 驱动**。这一条不能省，但可以简化成"驱动装好即可"。

## 2. 方案对比

| | 方案 A：Docker 一体化 | 方案 B：一键安装 + 一键启动（原生） | **方案 C：B 为默认 + A 可选（推荐）** |
| --- | --- | --- | --- |
| 用户步骤 | 装 Docker Desktop → `docker compose up` | 跑 `install.ps1` → 双击启动 | B 的步骤，需要隔离时再选 A |
| 下载量 | ~16.7 GB 镜像 | ~1.6 GB | 默认 ~1.6 GB |
| 需要登录 | 否 | 否（C++ 编译验证才需要，默认关闭） | 否 |
| 需要人工交互 | 是（装 Docker + 重启） | 否（除 GPU 驱动） | 否 |
| 能力损失 | 生成产物为 Linux 版；无法在 Windows 编译验证 | 无（编译验证降级为可选、明确标 BLOCKED） | 无 |
| 步骤数 | 3 | **3** | **3**（+1 可选） |

### 方案 A（Docker 一体化）
- `docker compose up` 一次拉起 redis + api + worker；前端静态文件由 API 直接提供。
- 优点：环境完全隔离，服务器部署最省心；本机已验证容器内 GPU 推理可用。
- 缺点：16.7 GB 镜像、必须装 Docker Desktop、容器内生成的 Engine 是 Linux 版（Windows 目标机仍需在 Windows 侧构建）。

### 方案 B（原生一键安装）
- `install.ps1`：检测/引导安装 Python 3.10 → 建 venv → pip 安装 → **自动下载 CUDA 头文件**（公共源）→ 初始化数据库 → （有 Node 才）构建前端 → 打印下一步。
- `QForge.bat` / `qforge serve`：启动 API（前端静态托管，运行期不需要 Node）+ 执行器（有 Redis 自动用 Celery，没有就用内置后台执行器）+ 自动打开浏览器。
- 编译验证（C++ 工程 / Docker 镜像）作为**可选增强**：环境齐备则自动启用，缺失则如实标 BLOCKED 并给出获取方式。

### 方案 C（推荐）
默认走 B（`pip` + 一条命令），同时保留 A 作为可选部署形态。两者共用同一份代码，只切换运行形态。

## 3. 推荐方案（C）的落地清单

| # | 工作项 | 说明 |
| --- | --- | --- |
| 1 | `qforge doctor` | 一条命令体检 GPU/驱动/TensorRT/CUDA 开发文件/Redis/Node/Docker，缺失项给出**可复制的安装命令** |
| 2 | 打包为可安装 Python 包 + `qforge` 命令 | `qforge serve` / `qforge init` / `qforge doctor` / `qforge version`，不再要求用户记住 uvicorn/celery 参数 |
| 3 | `install.ps1`（+`install.sh`） | 全程自动化：Python 检测→venv→依赖→CUDA 头文件→数据库→前端构建→下一步提示；不污染全局环境 |
| 4 | 一键启动器 | `QForge.bat`：起 API（静态托管前端）+ 执行器 + 打开浏览器；Redis 自动探测 |
| 5 | 配置自动生成 | 首次运行生成带默认值的 `.env`，Redis/Celery 为"探测到就用" |
| 6 | 文档重写 | README 快速开始压到**两条命令**；开发/容器/编译验证移入「可选能力」章节 |
| 7 | 保持开发路径 | 现有 pytest（141 项）、docker、前端 dev server 全部保留并继续可用 |

**目标：新机器步骤 12 → 3**（装 GPU 驱动 → 跑 `install.ps1` → 双击 `QForge.bat`）；
需要 C++ 编译验证时 +1 步（TRT 开发包，脚本给链接与放置路径）。

## 4. 风险与代价（重构前必须知悉）

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| Python 包化会改变导入路径 | 可能影响现有 141 项测试与 `app.*` 导入 | 先加兼容层（src-layout + 包路径映射），保证测试全绿再继续 |
| 前端静态托管 | SPA 深链接需回退到 `index.html`；API 前缀要与 Vite 代理一致 | 加 catch-all 路由 + 统一 `/api` 前缀；保留 dev server 供开发 |
| 自动下载 CUDA 头文件 | 依赖 NVIDIA 公共分发站可达 | 不可达时降级：跳过编译验证并明确提示（不伪造成功） |
| 首次运行生成 `.env` | 可能与用户已有配置冲突 | 只在文件不存在时生成，且打印生成的路径与内容摘要 |
| 一键启动器与现有 `.env`（celery 模式） | 两套启动方式可能不一致 | 启动器读取同一份 `.env`；`qforge doctor` 显示实际生效的执行方式 |

## 5. 进度（滚动更新）

### 已完成并实测

| 工作项 | 状态 | 证据 |
| --- | --- | --- |
| 后端可安装包 + `qforge` 命令 | 完成 | `pip install -e .` 后 `qforge version/doctor` 可用 |
| `qforge doctor` 环境体检 | 完成 | 逐项给出缺失与安装命令；必需项缺失时退出码 1 |
| `qforge serve` 一条命令启动 | 完成 | 自动选执行方式 + 自动带起 worker + 可选开浏览器；实测界面/文档/404 语义均正确 |
| 前端由 API 托管（运行期不需要 Node） | 完成 | `qforge build-frontend` 构建后 `GET /` 返回真实界面，深链接回退 200 |
| 执行方式自动（无 Redis 也能跑） | 完成 | `QFORGE_EXECUTOR_MODE=auto` 默认值；探测到 Redis 自动走 Celery |
| 数据/配置路径自动适配 | 完成 | 仓库检出用 `<repo>/storage`，安装形态用 `%LOCALAPPDATA%\QForge` |
| CUDA 开发文件自动获取（免登录） | 完成 | `qforge fetch-cuda-headers` 实测下载 3 个分发包并解压出 `cuda_runtime_api.h`/`crt/host_defines.h` |
| `scripts/install.ps1` 一键安装 | 完成 | 找 Python → 建 venv → 装依赖 → CUDA → 初始化 →（有 Node 就）构建前端 → 体检 |
| `QForge.bat` 双击启动 | 完成 | 优先用 `.venv`，否则用 PATH 上的 `qforge`；未安装时给出安装命令 |
| README 改为「两条命令」 | 完成 | 快速开始、可选能力表、环境要求均已重写 |
| **数据库迁移随包分发** | 完成 | 迁移脚本移入 `app/migrations`；`qforge init` 与 API 启动都会应用迁移。三种情况都实测：全新库→应用迁移、老库(create_all)→登记版本不报错、已最新→无动作 |
| 缺少 Python 时自动安装 | 完成 | 安装器可用 winget 装 Python 3.10（默认先询问，`-InstallPython` 跳过询问） |
| **Docker 一体化（compose：api + worker + redis）** | 完成 | `docker compose up -d --build` 一条命令；容器内 worker 真实构建 Engine 并推理（实测 9 秒 SUCCESS、MAE 1.85e-5），详见 [docker.md](docker.md) |
| CUDA 驱动加载跨平台 | 完成 | Linux 下加载 `libcuda.so.1`（原先只写 `nvcuda.dll`，容器里必然失败）；Engine 元数据新增 `platform` 字段 |

### 剩余

| 工作项 | 说明 |
| --- | --- |
| Linux/macOS 原生安装脚本（`install.sh`） | 容器化已覆盖 Linux 部署；原生脚本仍只有 Windows 版 |
| 前端构建产物是否随包分发 | 决定「新机器是否完全不需要 Node」——待用户确认 |
| 容器镜像瘦身 | 当前 9.3 GB（Linux TensorRT 运行库 3.7 GB）；只要 api 的场景可做无 tensorrt 的精简镜像 |

**步骤数：12 → 3**（装 NVIDIA 驱动 → `install.ps1` → `QForge.bat`）；
需要「编译生成的 C++ 工程」时 +1 步（TensorRT 开发包，需 NVIDIA 账号登录）。
升级到新版本后执行一次 `qforge init` 即可应用新的数据库迁移（API 启动时也会自动检查）。
