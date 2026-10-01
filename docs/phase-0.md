# 阶段 0 交付说明：脚手架与基础架构

SPEC 17 对阶段 0 的要求：**Vue / FastAPI / Redis / Celery / 存储 / 数据库 / 任务状态机**。

本文件记录该阶段的交付内容、关键决策、**实际执行过的验证**，以及未验证项与偏差。

---

## 1. 交付内容

| 模块 | 位置 | 状态 |
| --- | --- | --- |
| FastAPI 应用与统一错误出口 | `backend/app/main.py` | 完成 |
| 运行期配置（`QFORGE_*`） | `backend/app/config/settings.py` | 完成 |
| 数据库与 ORM（SPEC 14.1 八张表） | `backend/app/db/`、`backend/app/models/` | 完成 |
| schema 迁移 | `backend/migrations/`、`backend/alembic.ini` | 完成（初始迁移已生成并应用） |
| Pydantic 请求/响应模型 | `backend/app/schemas/` | 完成 |
| 任务状态机（SPEC 13） | `backend/app/services/task_state.py` | 完成（单测覆盖迁移规则） |
| 存储布局与路径安全（SPEC 14.2 / 15） | `backend/app/services/storage.py` | 完成（单测覆盖穿越与命名） |
| 任务编排与配置快照（SPEC 5.1 / 13） | `backend/app/services/task_service.py` | 完成 |
| 元数据读写（Project/Model/Dataset） | `backend/app/services/catalog.py` | 完成 |
| REST API（SPEC 16.1） | `backend/app/api/routes/` | 完成（上传类接口除外，见下） |
| Celery 应用与占位任务 | `workers/celery_app.py`、`workers/tasks.py` | 完成接线 |
| 结构化日志（SPEC 13.2） | `backend/app/logging_config.py`、`job_logs` 表 | 完成 |
| 前端骨架（SPEC 16 六个页面） | `frontend/` | 完成（两条真实链路 + 四个显式占位） |
| 基础设施编排（Redis + PostgreSQL） | `docker-compose.yml` | 已编写，**未运行验证** |
| 测试 | `tests/` | 完成（78 个用例） |

## 2. 关键决策（均经用户确认或依据 SPEC）

1. **Python 3.10.21（conda env `qforge`）**
   本机仅有 Python 3.13.5（Anaconda base）与 `carla_env`。阶段 1 要接 PPQ + TensorRT，
   而 TensorRT 的 Python 支持一贯保守，3.13 属高风险，故按用户确认新建 3.10 专用环境，
   避免阶段 1 迁移解释器并重锁依赖。

2. **阶段 0 使用 SQLite + Celery eager，Redis/PostgreSQL 编排待接**
   本机 Docker daemon 未运行且 `registry-1.docker.io:443` 不可达，无法拉起容器。
   按用户确认：阶段 0 用 `storage/qforge.db` + `task_always_eager=true` 打通全链路，
   `docker-compose.yml` 与配置项全部就绪，等环境可用后接入并验证（不会把未验证项写成已验证）。

3. **`transition()` 是任务状态的唯一写入口**
   `Task.status/progress/current_stage` 只在 `task_service.transition()` 中变更，先经状态机校验。
   这样「跳级/回退/自环」在服务层即被拒绝，而不是靠调用方自觉。

4. **TaskConfig 为不可变快照（SPEC 5.1）**
   任务创建时一次性写入 `task_configs.config`，四个使用方（前端 / API / Worker / 代码生成）
   共用同一份，避免执行期配置漂移。

5. **存储路径只由 `task_id` 生成，命名采用白名单**
   目录结构 `storage/<task_id>/{input,calibration,intermediate,engine,source,docker,logs,report}`；
   任何外部名（文件名、ZIP 条目）必须过 `sanitize_filename` / `safe_join`，越界即抛错。
   保留中文等 Unicode 字母，但拒绝空白、控制字符、分隔符、盘符与 Windows 保留名。

6. **未实现的能力显式暴露，不伪造**
   - `GET /api/tasks/{id}/report` → **501 NOT_IMPLEMENTED**（精度报告属阶段 2）
   - 占位 Worker **不把任务标记为 SUCCESS**，只登记领取记录并留下日志，任务停留在 `QUEUED`
   - 前端精度报告页与产物页显示「该功能在阶段 X 交付」的显式占位，不用假数据填充

7. **`tasks.artifact_id` 不加外键约束**
   `artifacts.task_id` 已指向 `tasks`；再让 `tasks.artifact_id` 指向 `artifacts` 会形成双向外键，
   带来建表与删除顺序耦合。阶段 0 由应用层维护该引用的一致性。

## 3. 实际执行过的验证

环境：conda env `qforge`（Python 3.10.21），仓库根 `D:\QForge`。

| 验证项 | 命令 / 方式 | 结果 |
| --- | --- | --- |
| 单元与接口测试 | `python -m pytest -q` | **78 passed** |
| 迁移生成 | `alembic -c backend/alembic.ini revision --autogenerate -m "phase 0 initial schema"` | 生成 `20261001_1125_2403cb7209da_phase_0_initial_schema.py` |
| 迁移应用 | `alembic -c backend/alembic.ini upgrade head` | 建表成功 |
| 表结构核对 | 直接查询 `sqlite_master` | `alembic_version, artifacts, backends, datasets, job_logs, models, projects, task_configs, tasks` |
| 真实 HTTP 端到端 | `uvicorn` + httpx 脚本 | 健康检查 `200 ok`；建项目/模型/数据集/任务 → 快照正确；`enqueue` → `QUEUED` 且 `worker_id` 写入；日志 3 条；`storage/<task_id>/` 8 个子目录齐备；`cancel` → `CANCELLED` |
| 错误路径 | 同上 | 重复入队 `409 CONFLICT`；报告 `501 NOT_IMPLEMENTED`；未知任务 `404 NOT_FOUND`；状态机 `QUEUED` 的允许迁移为 `CANCELLED / FAILED / VALIDATING` |
| 中文编码往返 | 同上（逐字节比对 UTF-8） | 项目描述与 `job_logs` 消息的 UTF-8 字节完全一致 |
| 前端生产构建 | `npm run build` | 成功（vite 8.3.1，产物 `frontend/dist/`） |
| 前端 dev server 与代理 | `npm run dev` + 请求 `/`、`/src/main.js`、`/api/health` | 均 `200`；`/api/health` 经代理返回 `QForge phase-0` |

测试覆盖的规则（摘要）：

- 状态机：成功路径逐级推进、跳级/回退/自环被拒、终态封闭、`FAILED`/`CANCELLED` 可达性、进度单调
- 存储：`..`、`a/b`、`a\b`、`C:`、控制字符、保留名（CON/NUL/COM1…）全部拒绝；清洗后保留扩展名；越界抛错
- 接口：项目/模型/数据集 CRUD 与冲突、404、422；任务创建、快照继承、INT8 必须有校准集、
  跨项目引用冲突、入队、重复入队、取消、日志、产物、501、列表过滤
- 接线：Celery eager 与超时配置、任务注册名、`probe` 可执行、健康检查、校验错误信封、未处理异常信封

## 4. 未验证项与已知限制

| 项 | 原因 | 后续 |
| --- | --- | --- |
| Redis 真实 broker 链路 | 本机无 Redis 服务 | 环境就绪后设 `QFORGE_CELERY_TASK_ALWAYS_EAGER=false` 并启动 worker 验证 |
| PostgreSQL | 未安装 | 阶段 1 引入 `psycopg` 并经 Alembic 迁移验证 |
| `docker-compose.yml` | daemon 未运行 + 镜像仓库不可达 | 镜像 tag 确认后 `docker compose up` 实测 |
| Celery 非 eager 模式 | 同上 | Windows 下用 `--pool=solo`；需验证并发与超时行为 |
| 前端交互 | 未做浏览器验证 | 仅验证构建与 dev server 静态响应 |
| 上传类接口 | 属阶段 1（SPEC 8.2 / 15） | 阶段 0 只有存储层命名与越界防护，**没有** ZIP 解压、炸弹防护与图像校验 |
| 资源限制强制执行 | 属阶段 1 | 配置项已就绪（`max_upload_mb`、`max_zip_*`、`max_concurrent_tasks`），尚未强制 |
| 断点续跑与并发领取 | 未实现 | 阶段 1 结合真实流水线设计 |

## 5. 与 SPEC 的偏差（需确认）

1. **未创建 `workers/model_analyzer|quantization|tensorrt|codegen|build|packaging` 子包。**
   SPEC 附录 A 是「MVP 推荐目录结构」，这些模块属阶段 1 起逐步加入；阶段 0 只建 `workers/`
   顶层包与 Celery 接线，避免空包与提前实现。
2. **前端 Element Plus 全量引入**，`index` chunk 约 1.06 MB（gzip 343 kB）。
   阶段 0 以可运行优先；如需优化，按需引入可在阶段 2 处理。
3. **`alembic.ini` 内容必须保持 ASCII。**
   实测 Alembic 以 OS locale 编码（本机 GBK）读取该文件，中文注释会导致
   `UnicodeDecodeError` 并使迁移命令完全无法启动。已在文件内注明。
4. **`QFORGE_AUTO_CREATE_SCHEMA=true` 默认开启**，用于本地便捷建表；正式 schema 变更仍以
   Alembic 迁移为准（阶段 1 起建议在非 dev 环境关闭）。

## 6. 阶段 1 前置条件（建议按序确认）

1. 确认 TensorRT / CUDA / PPQ / CMake / 编译器版本矩阵（见 [versions.md](versions.md) 第 4 节）
2. 补齐构建链：CMake、C++ 编译器、CUDA Toolkit、TensorRT Python 绑定
3. 决策并接入 Redis 与 PostgreSQL（或明确阶段 1 仍用 SQLite）
4. 确认首个 MVP 模型（如 YOLOv8 ONNX）及其 Model Definition（SPEC 6.1）
5. 明确上传与解压的安全实现方案（ZIP 炸弹、路径穿越、文件数与解压后总大小限制）
