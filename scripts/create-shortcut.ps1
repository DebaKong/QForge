<#
.SYNOPSIS
  在桌面与开始菜单创建 QForge 快捷方式（让它更像"一个应用程序"，而不是一堆脚本）。

.DESCRIPTION
  创建两个快捷方式，都指向仓库根的 QForge.bat：
    - 桌面
    - 开始菜单\QForge
  QForge.bat 会优先使用 <仓库>\.venv 里安装的 qforge，其次用 PATH 上的 qforge；
  因此先跑过一次 scripts\install.ps1 再用本脚本。

.PARAMETER Remove
  删除已创建的快捷方式（撤销）。

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\create-shortcut.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\create-shortcut.ps1 -Remove
#>
param(
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$target = Join-Path $repo "QForge.bat"

$desktop = [Environment]::GetFolderPath("Desktop")
$programs = [Environment]::GetFolderPath("Programs")
$startMenuDir = Join-Path $programs "QForge"

$links = @(
    (Join-Path $desktop "QForge.lnk"),
    (Join-Path $startMenuDir "QForge.lnk")
)

if ($Remove) {
    foreach ($link in $links) {
        if (Test-Path $link) {
            Remove-Item $link -Force
            Write-Host "已删除：$link"
        }
    }
    if (Test-Path $startMenuDir -and -not (Get-ChildItem $startMenuDir -Force)) {
        Remove-Item $startMenuDir -Force
    }
    Write-Host "完成。" -ForegroundColor Green
    exit 0
}

if (-not (Test-Path $target)) {
    Write-Host "[失败] 找不到 $target（请在仓库根运行本脚本）" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $repo ".venv\Scripts\qforge.exe")) -and -not (Get-Command qforge -ErrorAction SilentlyContinue)) {
    Write-Host "[警告] 还没检测到已安装的 qforge，请先运行 scripts\install.ps1" -ForegroundColor Yellow
}

if (-not (Test-Path $startMenuDir)) {
    New-Item -ItemType Directory -Path $startMenuDir -Force | Out-Null
}

$shell = New-Object -ComObject WScript.Shell
foreach ($link in $links) {
    $shortcut = $shell.CreateShortcut($link)
    $shortcut.TargetPath = $target
    $shortcut.WorkingDirectory = $repo
    $shortcut.Description = "QForge —— ONNX 自动化量化部署平台"
    # 用系统自带图标，避免额外资源文件
    $shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,13"
    $shortcut.Save()
    Write-Host "[通过] 已创建：$link" -ForegroundColor Green
}

Write-Host ""
Write-Host "双击桌面上的 QForge 即可启动（会打开浏览器进入界面）。" -ForegroundColor Cyan
Write-Host "撤销：scripts\create-shortcut.ps1 -Remove"
