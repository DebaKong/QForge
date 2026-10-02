# 启动 API（读取仓库根的 .env；默认 http://127.0.0.1:8000）
#
# 用法（仓库根）：  .\scripts\start-api.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = if ($env:QFORGE_PYTHON) { $env:QFORGE_PYTHON } else { "python" }
Write-Host "启动 API：$repo（python=$python）" -ForegroundColor Cyan
& $python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
