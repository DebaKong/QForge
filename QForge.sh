#!/usr/bin/env bash
# QForge 一键启动（Linux / macOS）—— 等价于 Windows 的 QForge.bat
#
# 首次使用请先运行： bash scripts/install.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

if [ -x "$REPO/.venv/bin/qforge" ]; then
    exec "$REPO/.venv/bin/qforge" serve --open "$@"
fi

if command -v qforge >/dev/null 2>&1; then
    exec qforge serve --open "$@"
fi

cat <<'MSG'

  未找到已安装的 QForge。

  请先安装（在仓库根执行一次）：
      bash scripts/install.sh

MSG
exit 1
