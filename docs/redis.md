# 接入 Redis + Celery（已实测）

阶段 1 默认用**进程内后台执行器**跑长任务（无 Redis 也能用，符合 SPEC 3.1「不占请求线程」）。
本文件说明如何切到真正的 Redis + Celery worker —— **业务代码零改动**，只改配置与启动方式。

本机已实测通过：入队 0.30 秒返回，任务由 worker 进程执行并在 49 秒内 SUCCESS。

---

## 0. 本机现状（已完成切换）

| 项 | 状态 |
| --- | --- |
| Redis | 容器 `qforge-redis-1`（`redis:7.4-alpine`，Redis 7.4.11），compose 中已设 `restart: unless-stopped` → **Docker Desktop 启动后会自动拉起**，不必每次手动 `up` |
| 配置 | 仓库根 `.env` 已写入 Redis/Celery 相关项（`.env` 不入库） |
| 验证 | `python tools/redis_switch_check.py --base http://127.0.0.1:8000/api` 通过：入队 **0.41s** 返回、`worker_id=qforge-worker@<主机名>`、**47s SUCCESS** |

日常启动顺序（仓库根，三个终端；也可用 `scripts/` 下的脚本）：

```powershell
.\scripts\start-api.ps1                  # 1) API      → http://127.0.0.1:8000
.\scripts\start-worker.ps1               # 2) worker   → 串行；并行用 -Concurrency 2
cd frontend; npm run dev                 # 3) 前端      → http://127.0.0.1:5173
```

> 脚本默认用 PATH 里的 `python`；若用 conda 环境，先设 `$env:QFORGE_PYTHON="C:\Users\<你>\.conda\envs\qforge\python.exe"`。
> 停止：worker 用 Ctrl+C；Redis 用 `docker compose stop redis`（想彻底移除：`docker compose down`）。

---

## 1. 启动 Redis（用仓库里的 docker-compose）

```powershell
# 在仓库根执行；本机 Docker Hub 直连不可达，但 daemon.json 里配了国内加速镜像
docker compose up -d redis

# 验证（应输出 PONG）
docker compose exec -T redis redis-cli ping

# 查看版本与状态
docker compose exec -T redis redis-server --version
docker compose ps
```

实测：`redis:7.4-alpine` → **Redis 7.4.11**，端口 6379 映射到本机，数据卷 `qforge_redis-data`
（compose 里已开 `appendonly`，重启不丢队列）。

> 只想临时试一下、不写配置文件的话，也可以直接：
> `docker run -d --name qforge-redis -p 6379:6379 redis:7.4-alpine`

## 2. 改配置（三行）

在仓库根创建/修改 `.env`（`.env` 不入库）：

```ini
# 1) 执行方式：local（进程内线程，默认）→ celery（投递到 broker）
QFORGE_EXECUTOR_MODE=celery

# 2) 关键：关掉 eager，否则「入队」仍会在 API 进程内同步执行
QFORGE_CELERY_TASK_ALWAYS_EAGER=false

# 3) broker 与结果后端（compose 里的 Redis；/0 走队列，/1 存结果）
QFORGE_CELERY_BROKER_URL=redis://127.0.0.1:6379/0
QFORGE_CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/1
```

## 3. 启动 worker

```powershell
# 在仓库根执行（workers/_bootstrap.py 会自动把 backend/ 加入导入路径）
celery -A workers.celery_app:celery_app worker --loglevel=info --pool=solo
```

**Windows 必须用 `--pool=solo`**（或 `--pool=threads`）：Celery 的默认 prefork 池在 Windows 上不受支持。
`solo` 池是**串行**执行（一次一个任务）；本机 CPU 是 24 核，但那行 `concurrency: 24 (solo)` 不代表能并行——
要并行请用 `--pool=threads --concurrency=2`，或起多个 worker 进程（见下方「并发」）。

启动成功的标志：

```text
-------------- qforge-worker@<主机名> v5.6.3 (recovery)
.> transport:   redis://127.0.0.1:6379/0
.> results:     redis://127.0.0.1:6379/1
[tasks]
  . qforge.probe
  . qforge.run_task_pipeline
... ready.
```

## 4. 重启 API

API 只在**启动时**读配置，所以改完 `.env` 需要重启：

```powershell
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

前端不用改：Vite 仍代理到 8000，`/api/health` 里的 `celery_eager` 会变成 `false`。

## 5. 验证（有现成脚本）

```powershell
python tools/redis_switch_check.py                    # 默认打 http://127.0.0.1:8001/api
python tools/redis_switch_check.py --base http://127.0.0.1:8000/api
```

脚本判定三件事：入队是否**立即返回**、任务的 `worker_id` 是否是 **worker 主机名**（而不是
`local-executor`）、任务是否最终 SUCCESS。实测输出：

```text
后端：ok / celery_eager=False
入队返回耗时 0.30s，任务状态=QUEUED（应远小于任务总耗时）
任务最终状态：SUCCESS（总耗时 49s）
worker_id     ：qforge-worker@DESKTOP-JOTA1J5
✅ 任务由 worker 进程执行：qforge-worker@DESKTOP-JOTA1J5
精度：MAE 0.0099 / 余弦 0.9999989
产物数量：17
```

同时 worker 侧日志能看到完整流水线（任务由 `MainProcess` 执行）：

```text
Task qforge.run_task_pipeline[...] received
Celery 领取任务
流水线开始：精度=fp16 后端=tensorrt 架构=yolov8
... VALIDATING → PREPROCESSING → QUANTIZING → BUILDING_ENGINE → GENERATING_CODE
    → BUILDING → TESTING → PACKAGING → SUCCESS
Task qforge.run_task_pipeline[...] succeeded in 46.5s
```

## 6. 注意事项（都是实测踩过的）

| 事项 | 说明 |
| --- | --- |
| `QFORGE_CELERY_TASK_ALWAYS_EAGER` 必须为 `false` | 否则「入队」会退化成在 API 进程里同步执行，长任务照样卡住请求 |
| Windows 池 | 只能 `--pool=solo` / `threads`；`prefork` 不支持。solo 是串行 |
| 并发限制的归属 | `QFORGE_MAX_CONCURRENT_TASKS` 只作用于**进程内执行器**；Celery 模式下并发由 worker 的池与数量决定 |
| 数据库 | 默认仍是 SQLite（已开 WAL）。API 与 worker 是**两个进程**同时读写同一个库文件，WAL 能扛住；若要更高并发再按需切 PostgreSQL |
| 任务日志 | 仍在 `job_logs` 表与 `storage/<task_id>/logs/`，与 Redis 无关（Redis 只传消息与结果） |
| 超时 | `QFORGE_TASK_TIMEOUT_SECONDS` 已同时用于 Celery 的 `task_time_limit`/`soft_time_limit` |
| 停止 | `docker compose stop redis`；worker 用 Ctrl+C；回退到本地执行器只需把 `QFORGE_EXECUTOR_MODE` 改回 `local` 并重启 API |
| 生产建议 | Redis 加 `requirepass`、只监听内网；`docker-compose.yml` 里的镜像 tag 属版本敏感项（SPEC 4.1），升级前先在 docs/versions.md 登记 |

## 7. 可选：查看队列与运行时状态

```powershell
# 队列里堆积的消息（Redis 模式）
docker compose exec -T redis redis-cli -n 0 llen celery

# worker 是否在线 / 正在执行什么（在仓库根执行）
celery -A workers.celery_app:celery_app inspect ping
celery -A workers.celery_app:celery_app inspect active
celery -A workers.celery_app:celery_app inspect registered
```
