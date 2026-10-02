<#
.SYNOPSIS
  QForge 一键安装（Windows）。

.DESCRIPTION
  把「新电脑从零到能用」压到最少步骤：

    1) 找到 Python 3.10（缺失则打印可复制的安装命令后退出）
    2) 在 <仓库>\.venv 建独立虚拟环境（不污染系统 Python）
    3) 安装运行期依赖（含 TensorRT 运行库，pip 提供，**无需登录 NVIDIA**）
    4) 自动下载 CUDA 运行时开发文件（公开源，无需登录；仅「编译生成的 C++ 工程」需要）
    5) 初始化数据目录与数据库
    6) 有 Node.js 就构建前端界面（没有则跳过，服务仍可用并给出说明页）
    7) 运行环境体检，并打印下一步

  任何一步失败都会明确报错并停止，不会伪造成功。CUDA 下载属于可选增强，失败只警告。

.PARAMETER Python
  指定 Python 解释器路径。默认依次尝试 py -3.10 / python。
.PARAMETER VenvDir
  虚拟环境目录，默认 <仓库>\.venv。
.PARAMETER Editable
  以可编辑方式安装（改代码立即生效，适合开发）。
.PARAMETER SkipDeps
  只建环境，不安装依赖（离线或调试用）。
.PARAMETER SkipCuda
  不下载 CUDA 开发文件（只影响 C++ 工程编译验证）。
.PARAMETER SkipFrontend
  不构建前端界面（API 仍可用，界面显示内置说明页）。

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -SkipCuda -SkipFrontend
#>
param(
    [string]$Python = "",
    [string]$VenvDir = "",
    [switch]$Editable,
    [switch]$SkipDeps,
    [switch]$SkipCuda,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# 中文输出编码：子进程(qforge)写 UTF-8，父进程也按 UTF-8 解码（否则 GBK 控制台下乱码）
$env:PYTHONIOENCODING = "utf-8"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

function Write-Step([string]$text) {
    Write-Host ""
    Write-Host "==> $text" -ForegroundColor Cyan
}

function Write-Ok([string]$text) { Write-Host "    [通过] $text" -ForegroundColor Green }
function Write-Warn2([string]$text) { Write-Host "    [警告] $text" -ForegroundColor Yellow }
function Fail([string]$text) {
    Write-Host ""
    Write-Host "[失败] $text" -ForegroundColor Red
    exit 1
}

Write-Host "QForge 安装程序" -ForegroundColor White
Write-Host "仓库位置：$repo"

# --------------------------------------------------------------------------- #
# 1) 找 Python 3.10
# --------------------------------------------------------------------------- #
Write-Step "1/7 查找 Python 3.10"
$pythonExe = ""
$candidates = @()
if ($Python) { $candidates += @{ Exe = $Python; Args = @() } }
$candidates += @(
    @{ Exe = "py"; Args = @("-3.10") },
    @{ Exe = "python"; Args = @() },
    @{ Exe = "python3"; Args = @() }
)

foreach ($candidate in $candidates) {
    $exe = $candidate.Exe
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    try {
        $probe = & $exe @($candidate.Args) -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>$null
    } catch { continue }
    if (-not $probe) { continue }
    $parts = $probe.Trim().Split(".")
    if ([int]$parts[0] -eq 3 -and [int]$parts[1] -eq 10) {
        $pythonExe = $exe
        $pythonArgs = $candidate.Args
        Write-Ok "使用 $exe（Python $probe）"
        break
    }
    Write-Warn2 "$exe 是 Python $probe（需要 3.10.x，跳过）"
}

if (-not $pythonExe) {
    Write-Host ""
    Write-Host "未找到 Python 3.10。请任选一种方式安装后重新运行本脚本：" -ForegroundColor Yellow
    Write-Host "  winget install -e --id Python.Python.3.10"
    Write-Host "  或从 https://www.python.org/downloads/release/python-31011/ 下载安装"
    Write-Host "（安装时勾选 Add Python to PATH；本项目锁定 3.10，其它版本未验证）"
    exit 1
}

# --------------------------------------------------------------------------- #
# 2) 建虚拟环境
# --------------------------------------------------------------------------- #
if (-not $VenvDir) { $VenvDir = Join-Path $repo ".venv" }
Write-Step "2/7 准备虚拟环境：$VenvDir"
$venvPython = Join-Path $VenvDir "Scripts\python.exe"
if (Test-Path $venvPython) {
    Write-Ok "已存在，复用"
} else {
    & $pythonExe @($pythonArgs) -m venv $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) { Fail "创建虚拟环境失败" }
    Write-Ok "已创建"
}

# --------------------------------------------------------------------------- #
# 3) 安装依赖
# --------------------------------------------------------------------------- #
Write-Step "3/7 安装依赖（首次约 1.5 GB，主要是 TensorRT 运行库）"
if ($SkipDeps) {
    Write-Warn2 "按参数要求跳过依赖安装"
} else {
    & $venvPython -m pip install --upgrade pip --quiet
    $target = if ($Editable) { "-e" } else { "." }
    $installArgs = @("-m", "pip", "install")
    if ($Editable) { $installArgs += "-e" }
    $installArgs += "."
    & $venvPython @installArgs
    if ($LASTEXITCODE -ne 0) { Fail "pip 安装失败，请检查网络后重试" }
    Write-Ok "依赖安装完成"
}

$qforge = Join-Path $VenvDir "Scripts\qforge.exe"
if (-not (Test-Path $qforge)) { Fail "未找到 $qforge（安装未完成）" }

# --------------------------------------------------------------------------- #
# 4) CUDA 运行时开发文件（公开源，无需登录）
# --------------------------------------------------------------------------- #
Write-Step "4/7 准备 CUDA 运行时开发文件（C++ 编译验证用，可选）"
if ($SkipCuda) {
    Write-Warn2 "按参数要求跳过；编译生成的 C++ 工程时会提示缺什么"
} else {
    & $qforge fetch-cuda-headers
    if ($LASTEXITCODE -ne 0) {
        Write-Warn2 "CUDA 文件未准备好：核心流程不受影响，只影响 C++ 工程编译验证"
    } else {
        Write-Ok "CUDA 头文件与导入库已就绪"
    }
}

# --------------------------------------------------------------------------- #
# 5) 初始化数据目录与数据库
# --------------------------------------------------------------------------- #
Write-Step "5/7 初始化数据目录与数据库"
& $qforge init
if ($LASTEXITCODE -ne 0) { Fail "初始化失败" }
Write-Ok "数据目录与表结构已就绪"

# --------------------------------------------------------------------------- #
# 6) 前端界面（可选）
# --------------------------------------------------------------------------- #
Write-Step "6/7 构建前端界面（可选，需要 Node.js）"
if ($SkipFrontend) {
    Write-Warn2 "按参数要求跳过；服务仍可用，界面会显示内置说明页"
} elseif (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Warn2 "未检测到 npm（Node.js）。跳过前端构建；界面将显示说明页，API 不受影响"
    Write-Warn2 "需要界面时：安装 Node.js 后执行 qforge build-frontend --source `"$repo\frontend`" --publish"
} else {
    # 安装形态下应用不认识仓库路径，因此显式传源码目录，并发布到 <数据目录>/../web
    & $qforge build-frontend --source "$repo\frontend" --publish
    if ($LASTEXITCODE -ne 0) {
        Write-Warn2 "前端构建失败（不影响 API 使用）"
    } else {
        # 安装形态下界面由数据目录旁的 web/ 提供，这里把构建产物放过去
        $dataRoot = (& $venvPython -c "from app.paths import data_root; print(data_root())").Trim()
        $webRoot = Join-Path (Split-Path -Parent $dataRoot) "web"
        if (Test-Path $webRoot) { Remove-Item $webRoot -Recurse -Force }
        New-Item -ItemType Directory -Path $webRoot -Force | Out-Null
        Copy-Item (Join-Path $repo "frontend\dist\*") $webRoot -Recurse -Force
        Write-Ok "前端界面已就绪：$webRoot"
    }
}

# --------------------------------------------------------------------------- #
# 7) 体检 + 下一步
# --------------------------------------------------------------------------- #
Write-Step "7/7 环境体检"
& $qforge doctor

Write-Host ""
Write-Host "安装完成。启动方式：" -ForegroundColor Green
Write-Host "    .\QForge.bat                  （双击同目录的 QForge.bat 也可以）"
Write-Host "    $qforge serve --open"
Write-Host ""
Write-Host "提示：本机没有 Redis 也能用（自动使用进程内执行器）；"
Write-Host "      需要多任务并行/独立 worker 时执行 docker compose up -d redis 即可自动切换。"
Write-Host ""

# 显式退出码：脚本被其它工具调用时状态必须确定（子进程往 stderr 写日志不算失败）
exit 0
