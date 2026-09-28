@echo off
setlocal
title sbai (Python 3.12) 환경

echo ======================================================================
echo  [sbai] 가상환경을 활성화합니다...
echo ======================================================================

set "CONDA_BAT="
if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"

if defined CONDA_BAT (
    call "%CONDA_BAT%" activate sbai
) else (
    call conda activate sbai 2>nul
)

if %errorlevel% neq 0 (
    echo.
    echo [경고] 'sbai' 가상환경을 활성화하지 못했습니다.
    echo 먼저 'setup_environment.bat'을 실행해 주세요.
    pause
    exit /b 1
)

echo.
echo 가상환경 활성화 완료: sbai
call python --version
echo.
echo ======================================================================
echo  이 창에서 바로 Python 및 Jupyter 관련 명령어를 실행할 수 있습니다.
echo  VS Code 실행: code .
echo  종료: exit
echo ======================================================================
echo.

cmd /k