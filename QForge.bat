@echo off
rem QForge one-click start (double-click this file).
rem First time: run  scripts\install.ps1
rem Keep this file ASCII-only with CRLF line endings:
rem   cmd.exe needs CRLF for labels/goto and garbles UTF-8 Chinese text
rem   (see AGENTS.md "????????????").
setlocal
cd /d "%~dp0"

chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"

set "QFORGE_EXE=%~dp0.venv\Scripts\qforge.exe"
if exist "%QFORGE_EXE%" goto run_local

where qforge >nul 2>nul
if errorlevel 1 goto not_installed
qforge serve --open %*
goto done

:run_local
"%QFORGE_EXE%" serve --open %*
goto done

:not_installed
echo.
echo  QForge is not installed yet.
echo.
echo  Run this once in the repository root:
echo      powershell -ExecutionPolicy Bypass -File scripts\install.ps1
echo.
pause

:done
endlocal
