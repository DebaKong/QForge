#!/usr/bin/env bash
# QForge 一键安装（Linux / macOS）
#
# 与 Windows 的 scripts\install.ps1 等价，把「新机器从零到能用」压到最少步骤：
#   1) 找 Python 3.10（缺失则打印可复制的安装命令后退出）
#   2) 在 <仓库>/.venv 建独立虚拟环境（不污染系统 Python）
#   3) 安装依赖（默认含 GPU 额外项：TensorRT 运行库 + cuda-python，无需登录 NVIDIA）
#   4) 自动下载 CUDA 运行时开发文件（公开源，无需登录；仅「编译生成的 C++ 工程」需要）
#   5) 初始化数据目录与数据库（应用迁移）
#   6) 有 Node.js 就构建前端界面（没有则跳过，界面显示说明页）
#   7) 运行环境体检，并打印下一步
#
# 任何一步失败都会明确报错并停止，不会伪造成功；CUDA 下载属可选增强，失败只警告。
#
# 用法：
#   bash scripts/install.sh [选项]
#     --python <路径>     指定解释器（默认依次尝试 python3.10 / python3 / python）
#     --venv <目录>       虚拟环境目录（默认 <仓库>/.venv）
#     --editable          以可编辑方式安装（开发用）
#     --no-gpu            不装 GPU 额外项（只跑 API/界面时用）
#     --skip-deps         只建环境，不装依赖（离线/调试）
#     --skip-cuda         不下载 CUDA 开发文件
#     --skip-frontend     不构建前端界面
#     --pip-index <url>   指定 pip 源（国内网络建议换成镜像，例：
#                         --pip-index https://pypi.tuna.tsinghua.edu.cn/simple）
#     -h, --help          显示本帮助

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON=""
VENV_DIR="$REPO/.venv"
EDITABLE=0
NO_GPU=0
SKIP_DEPS=0
SKIP_CUDA=0
SKIP_FRONTEND=0
PIP_INDEX=""

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

step() { printf '\n==> %s\n' "$1"; }
ok()   { printf '    [通过] %s\n' "$1"; }
warn() { printf '    [警告] %s\n' "$1"; }
fail() { printf '\n[失败] %s\n' "$1"; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --python)       PYTHON="${2:-}"; shift 2 ;;
        --venv)         VENV_DIR="${2:-}"; shift 2 ;;
        --editable)     EDITABLE=1; shift ;;
        --no-gpu)       NO_GPU=1; shift ;;
        --skip-deps)    SKIP_DEPS=1; shift ;;
        --skip-cuda)    SKIP_CUDA=1; shift ;;
        --skip-frontend) SKIP_FRONTEND=1; shift ;;
        --pip-index)    PIP_INDEX="${2:-}"; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        *)              printf '未知参数：%s\n\n' "$1"; usage; exit 2 ;;
    esac
done

printf 'QForge 安装程序\n'
printf '仓库位置：%s\n' "$REPO"

# --------------------------------------------------------------------------- #
# 1) 找 Python 3.10
# --------------------------------------------------------------------------- #
step "1/7 查找 Python 3.10"
PYTHON_EXE=""
PYTHON_VERSION=""

try_python() {
    local exe="$1" version
    command -v "$exe" >/dev/null 2>&1 || [ -x "$exe" ] || return 1
    version="$("$exe" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || true)"
    [ -n "$version" ] || return 1
    case "$version" in
        3.10.*) PYTHON_EXE="$exe"; PYTHON_VERSION="$version"; return 0 ;;
        *) warn "$exe 是 Python $version（需要 3.10.x，跳过）"; return 1 ;;
    esac
}

if [ -n "$PYTHON" ]; then
    try_python "$PYTHON" || warn "指定解释器不可用或不是 3.10：$PYTHON"
fi
if [ -z "$PYTHON_EXE" ]; then
    for candidate in python3.10 python3 python; do
        if try_python "$candidate"; then break; fi
    done
fi

if [ -z "$PYTHON_EXE" ]; then
    printf '\n未找到 Python 3.10。请先安装（任选一种）后重新运行本脚本：\n'
    printf '  Ubuntu/Debian : sudo apt update && sudo apt install -y python3.10 python3.10-venv\n'
    printf '  Fedora/RHEL   : sudo dnf install -y python3.10\n'
    printf '  macOS (brew)  : brew install python@3.10\n'
    printf '  通用（pyenv） : pyenv install 3.10.11 && pyenv local 3.10.11\n'
    printf '（本项目锁定 3.10，其它版本未验证）\n'
    exit 1
fi
ok "使用 $PYTHON_EXE（Python $PYTHON_VERSION）"

# --------------------------------------------------------------------------- #
# 2) 建虚拟环境
# --------------------------------------------------------------------------- #
step "2/7 准备虚拟环境：$VENV_DIR"
VENV_PYTHON="$VENV_DIR/bin/python"
if [ -x "$VENV_PYTHON" ]; then
    ok "已存在，复用"
else
    "$PYTHON_EXE" -m venv "$VENV_DIR" || fail "创建虚拟环境失败（Debian/Ubuntu 需要 python3.10-venv 包）"
    [ -x "$VENV_PYTHON" ] || fail "创建虚拟环境失败：$VENV_PYTHON 不存在"
    ok "已创建"
fi

# --------------------------------------------------------------------------- #
# 3) 安装依赖
# --------------------------------------------------------------------------- #
REQUIREMENT="."
if [ "$NO_GPU" -eq 0 ]; then
    REQUIREMENT=".[gpu]"
fi

step "3/7 安装依赖（$REQUIREMENT；首次约 1.5 GB，主要是 TensorRT 运行库）"
if [ "$SKIP_DEPS" -eq 1 ]; then
    warn "按参数要求跳过依赖安装"
else
    # 注意：bash 3.2（macOS 自带）在 set -u 下展开空数组会报错，必须用 ${arr[@]+"${arr[@]}"} 写法
    PIP_ARGS=()
    if [ -n "$PIP_INDEX" ]; then
        PIP_ARGS+=(--index-url "$PIP_INDEX")
        ok "使用 pip 源：$PIP_INDEX"
    fi
    "$VENV_PYTHON" -m pip install --upgrade pip --quiet ${PIP_ARGS[@]+"${PIP_ARGS[@]}"}
    if [ "$NO_GPU" -eq 1 ]; then
        warn "按参数要求跳过 GPU 额外项（无法构建 Engine，qforge doctor 会标为缺失）"
    fi
    if [ "$EDITABLE" -eq 1 ]; then
        "$VENV_PYTHON" -m pip install ${PIP_ARGS[@]+"${PIP_ARGS[@]}"} -e "$REQUIREMENT" \
            || fail "pip 安装失败，请检查网络后重试（或换源：--pip-index https://pypi.tuna.tsinghua.edu.cn/simple）"
    else
        "$VENV_PYTHON" -m pip install ${PIP_ARGS[@]+"${PIP_ARGS[@]}"} "$REQUIREMENT" \
            || fail "pip 安装失败，请检查网络后重试（或换源：--pip-index https://pypi.tuna.tsinghua.edu.cn/simple）"
    fi
    ok "依赖安装完成"
fi

QFORGE="$VENV_DIR/bin/qforge"
[ -x "$QFORGE" ] || fail "未找到 $QFORGE（安装未完成）"

# --------------------------------------------------------------------------- #
# 4) CUDA 运行时开发文件（公开源，无需登录）
# --------------------------------------------------------------------------- #
step "4/7 准备 CUDA 运行时开发文件（C++ 编译验证用，可选）"
if [ "$SKIP_CUDA" -eq 1 ]; then
    warn "按参数要求跳过；编译生成的 C++ 工程时会提示缺什么"
elif "$QFORGE" fetch-cuda-headers; then
    ok "CUDA 头文件与导入库已就绪"
else
    warn "CUDA 文件未准备好：核心流程不受影响，只影响 C++ 工程编译验证"
fi

# --------------------------------------------------------------------------- #
# 5) 初始化数据目录与数据库
# --------------------------------------------------------------------------- #
step "5/7 初始化数据目录与数据库"
"$QFORGE" init || fail "初始化失败"
ok "数据目录与表结构已就绪"

# --------------------------------------------------------------------------- #
# 6) 前端界面（可选）
# --------------------------------------------------------------------------- #
step "6/7 构建前端界面（可选，需要 Node.js）"
if [ "$SKIP_FRONTEND" -eq 1 ]; then
    warn "按参数要求跳过；服务仍可用，界面会显示内置说明页"
elif ! command -v npm >/dev/null 2>&1; then
    warn "未检测到 npm（Node.js）。跳过前端构建；界面将显示说明页，API 不受影响"
    warn "需要界面时：安装 Node.js 后执行 $QFORGE build-frontend --source $REPO/frontend --publish"
elif "$QFORGE" build-frontend --source "$REPO/frontend" --publish; then
    ok "前端界面已就绪并发布（由 API 直接托管）"
else
    warn "前端构建失败（不影响 API 使用）"
fi

# --------------------------------------------------------------------------- #
# 7) 体检 + 下一步
# --------------------------------------------------------------------------- #
step "7/7 环境体检"
"$QFORGE" doctor || true   # 体检不通过时也已经打印了缺什么，这里不当作安装失败

printf '\n安装完成。启动方式：\n'
printf '    ./QForge.sh                  （等价于 %s serve --open）\n' "$QFORGE"
printf '    %s serve --open\n' "$QFORGE"
printf '\n提示：没有 Redis 也能用（自动使用进程内执行器）；\n'
printf '     需要队列/独立 worker 时执行 docker compose up -d redis 即可自动切换。\n\n'

exit 0
