# QForge 应用镜像（可选的容器化部署方式；本机默认走原生安装，见 README）
#
# 设计说明：
#   1. 多阶段：Node 阶段构建前端，Python 阶段装应用本体；运行期不需要 Node；
#   2. 前端放在 /opt/qforge/web（**镜像内**）并用 QFORGE_FRONTEND_DIR 指定，
#      避免被 /data 数据卷遮蔽（卷挂载会覆盖镜像内同路径内容，是常见坑）；
#   3. 数据（任务目录、数据库）统一放 /data，作为卷持久化；
#   4. 镜像内不装 CUDA/TensorRT 开发包：Engine 构建只需要 pip 提供的 TensorRT 运行库；
#      「编译生成的 C++ 工程」这一步需要开发包，属可选能力，容器里默认缺失并如实标注。
#
# syntax=docker/dockerfile:1

# 软件源（可按网络环境覆盖，例如国内镜像）：
#   docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG NPM_REGISTRY=https://registry.npmjs.org

# ---------------- 前端构建 ----------------
FROM node:22-alpine AS frontend
ARG NPM_REGISTRY
WORKDIR /src/frontend
# 先装依赖再拷源码，利用层缓存
COPY frontend/package.json frontend/package-lock.json ./
RUN npm config set registry "${NPM_REGISTRY}" \
 && npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------------- 应用运行 ----------------
FROM python:3.10-slim AS runtime
ARG PIP_INDEX_URL

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /opt/qforge

# 只拷贝安装所需内容（其余由 .dockerignore 排除）
COPY pyproject.toml README.md ./
COPY backend/ ./backend/
COPY workers/ ./workers/
RUN pip install . \
 && rm -rf /opt/qforge/backend /opt/qforge/workers /opt/qforge/pyproject.toml

# 前端静态文件（用显式路径，避免被数据卷遮蔽）
COPY --from=frontend /src/frontend/dist /opt/qforge/web

ENV QFORGE_FRONTEND_DIR=/opt/qforge/web \
    QFORGE_DATA_DIR=/data/storage \
    QFORGE_EXECUTOR_MODE=celery \
    QFORGE_CELERY_TASK_ALWAYS_EAGER=false \
    QFORGE_CELERY_BROKER_URL=redis://redis:6379/0 \
    QFORGE_CELERY_RESULT_BACKEND=redis://redis:6379/1

VOLUME ["/data"]
EXPOSE 8000

# 默认只跑 API；worker 由 compose 的另一个服务用同一个镜像执行 `qforge worker`
CMD ["qforge", "serve", "--host", "0.0.0.0", "--port", "8000", "--no-worker"]
