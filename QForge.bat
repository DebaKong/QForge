@echo off
rem QForge 一键启动（双击本文件即可）
rem 首次使用请先运行： scripts\install.ps1
setlocal
cd /d "%~dp0"

rem 中文输出：控制台切到 UTF-8，并让 Python 也按 UTF-8 输出（否则乱码）
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

set "QFORGE_EXE=%~dp0.venv\Scripts\qforge.exe"
if exist "%QFORGE_EXE%" goto :run_local

where qforge >nul 2>nul
if errorlevel 1 goto :not_installed
qforge serve --open %*
goto :done

:run_local
"%QFORGE_EXE%" serve --open %*
goto :done

:not_installed
echo.
echo  未找到已安装的 QForge。
echo.
echo  请先安装（在仓库根执行一次）：
echo      powershell -ExecutionPolicy Bypass -File scripts\install.ps1
echo.
pause

:done
endlocal
