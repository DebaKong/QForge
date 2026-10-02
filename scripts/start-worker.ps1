# 启动 Celery worker（执行量化 / Engine 构建 / 编译 / 运行验证等长任务）
#
# 用法（仓库根）：  .\scripts\start-worker.ps1
# 可选参数：        .\scripts\start-worker.ps1 -Concurrency 2   （用 threads 池并行）
#
# 注意：Windows 不支持 Celery 的默认 prefork 池，因此默认用 solo（串行）；
#       需要并行时用 -Concurrency N（自动切换为 threads 池）。
param(
    [int]$Concurrency = 0
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# 中文输出编码：子进程写 UTF-8，父进程也按 UTF-8 解码（否则 GBK 控制台下乱码）
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$python = if ($env:QFORGE_PYTHON) { $env:QFORGE_PYTHON } else { "python" }

if ($Concurrency -gt 1) {
    Write-Host "启动 worker（threads 池，并发 $Concurrency）" -ForegroundColor Cyan
    & $python -m celery -A workers.celery_app:celery_app worker `
        --loglevel=info --pool=threads --concurrency=$Concurrency
} else {
    Write-Host "启动 worker（solo 池，串行执行）" -ForegroundColor Cyan
    & $python -m celery -A workers.celery_app:celery_app worker --loglevel=info --pool=solo
}
