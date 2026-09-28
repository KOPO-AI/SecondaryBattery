@echo off
setlocal enabledelayedexpansion
title Miniconda, sbai and VS Code Auto Setup

echo ======================================================================
echo   [1/5] Conda (Miniconda / Anaconda) 확인 및 설치
echo ======================================================================
echo.

set "CONDA_DIR="
set "CONDA_BAT="

if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    set "CONDA_DIR=%USERPROFILE%\anaconda3"
    set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
)
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    set "CONDA_DIR=%USERPROFILE%\miniconda3"
    set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
)

if defined CONDA_BAT (
    echo [확인] 이미 Conda가 설치되어 있습니다: !CONDA_BAT!
    goto CONDA_PATH_SETUP
)

set "CONDA_DIR=%USERPROFILE%\miniconda3"
set "MINICONDA_INSTALLER=%TEMP%\Miniconda3-latest-Windows-x86_64.exe"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

echo [설치] Miniconda 설치 프로그램을 다운로드합니다...
curl.exe -L -# -o "%MINICONDA_INSTALLER%" "%MINICONDA_URL%"
if not exist "%MINICONDA_INSTALLER%" (
    echo [오류] Miniconda 다운로드에 실패했습니다. 인터넷 연결을 확인해 주세요.
    pause
    exit /b 1
)

echo [설치] Miniconda 무인 설치를 진행 중입니다... 잠시만 기다려 주세요.
start /wait "" "%MINICONDA_INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_DIR%
if exist "%MINICONDA_INSTALLER%" del "%MINICONDA_INSTALLER%" 2>nul

set "CONDA_BAT=%CONDA_DIR%\condabin\conda.bat"
if not exist "%CONDA_BAT%" (
    echo [오류] Miniconda 설치 확인에 실패했습니다.
    pause
    exit /b 1
)
echo [완료] Miniconda 설치 완료!

:CONDA_PATH_SETUP
echo.
echo ======================================================================
echo   [2/5] Conda 환경 변수(PATH) 자동 등록
echo ======================================================================
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$dirs = @('%CONDA_DIR%', '%CONDA_DIR%\Scripts', '%CONDA_DIR%\condabin', '%CONDA_DIR%\Library\bin'); $up = [Environment]::GetEnvironmentVariable('Path', 'User'); $changed = $false; foreach ($d in $dirs) { if ($d -and (Test-Path $d) -and ($up -split ';' -notcontains $d)) { $up = \"$up;$d\"; $changed = $true } }; if ($changed) { [Environment]::SetEnvironmentVariable('Path', $up.Trim(';'), 'User'); Write-Host '  Conda 경로가 사용자 PATH에 등록되었습니다.' } else { Write-Host '  Conda 경로가 이미 PATH에 등록되어 있습니다.' }"

echo.
echo ======================================================================
echo   [3/5] 'sbai' [Python 3.12 + ipykernel] 가상환경 확인 및 생성
echo ======================================================================
echo.

:: 최신 Conda 약관(ToS) 자동 동의 처리
call "%CONDA_BAT%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>nul
call "%CONDA_BAT%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>nul
call "%CONDA_BAT%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >nul 2>nul
call "%CONDA_BAT%" tos accept --all >nul 2>nul

call "%CONDA_BAT%" env list > "%TEMP%\conda_envs.txt" 2>nul
findstr /b /c:"sbai " "%TEMP%\conda_envs.txt" >nul 2>nul
if %errorlevel% equ 0 (
    echo [확인] 'sbai' 가상환경이 이미 존재합니다.
    echo        ipykernel 패키지 상태를 점검합니다...
    call "%CONDA_BAT%" install -y -n sbai ipykernel >nul 2>nul
) else (
    echo [생성] 'sbai' Python 3.12 및 ipykernel 가상환경을 생성합니다...
    call "%CONDA_BAT%" create -y -n sbai python=3.12 ipykernel
    if !errorlevel! neq 0 (
        echo [재시도] conda-forge 채널을 이용하여 가상환경 생성을 재시도합니다...
        call "%CONDA_BAT%" create -y -n sbai -c conda-forge python=3.12 ipykernel
        if !errorlevel! neq 0 (
            echo [오류] 가상환경 생성 중 오류가 발생했습니다.
            del "%TEMP%\conda_envs.txt" 2>nul
            pause
            exit /b 1
        )
    )
)
del "%TEMP%\conda_envs.txt" 2>nul

echo [등록] Jupyter에 'sbai' 커널 등록 중...
call "%CONDA_BAT%" run -n sbai python -m ipykernel install --user --name sbai --display-name "Python 3.12 (sbai)"

echo [초기화] 터미널 연동 [conda init]...
call "%CONDA_BAT%" init cmd.exe >nul 2>nul
call "%CONDA_BAT%" init powershell >nul 2>nul

echo.
echo ======================================================================
echo   [4/5] Visual Studio Code 확인 및 자동 설치
echo ======================================================================
echo.

set "CODE_CMD="
set "VSCODE_BIN="

if exist "%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd" (
    set "CODE_CMD=%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd"
    set "VSCODE_BIN=%LOCALAPPDATA%\Programs\Microsoft VS Code\bin"
)
if not defined CODE_CMD if exist "%ProgramFiles%\Microsoft VS Code\bin\code.cmd" (
    set "CODE_CMD=%ProgramFiles%\Microsoft VS Code\bin\code.cmd"
    set "VSCODE_BIN=%ProgramFiles%\Microsoft VS Code\bin"
)
if not defined CODE_CMD (
    where code >nul 2>nul
    if !errorlevel! equ 0 (
        set "CODE_CMD=code"
    )
)

if defined CODE_CMD (
    echo [확인] VS Code가 이미 설치되어 있습니다. [설치 스킵]
    echo        경로: !CODE_CMD!
    goto VSCODE_PATH_SETUP
)

echo [안내] VS Code가 설치되어 있지 않습니다. 다운로드 및 설치를 시작합니다...
set "VSCODE_INSTALLER=%TEMP%\VSCodeUserSetup-x64.exe"
set "VSCODE_URL=https://update.code.visualstudio.com/latest/win32-x64-user/stable"

echo [설치] VS Code 설치 프로그램을 다운로드합니다...
curl.exe -L -# -o "%VSCODE_INSTALLER%" "%VSCODE_URL%"
if not exist "%VSCODE_INSTALLER%" (
    echo [오류] VS Code 다운로드에 실패했습니다.
    pause
    exit /b 1
)

echo [설치] VS Code 무인 설치를 진행 중입니다... 잠시만 기다려 주세요.
start /wait "" "%VSCODE_INSTALLER%" /VERYSILENT /NORESTART /MERGETASKS=!runcode,addtopath
if exist "%VSCODE_INSTALLER%" del "%VSCODE_INSTALLER%" 2>nul

if exist "%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd" (
    set "CODE_CMD=%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd"
    set "VSCODE_BIN=%LOCALAPPDATA%\Programs\Microsoft VS Code\bin"
    echo [완료] VS Code 설치 완료!
) else (
    echo [경고] VS Code 기본 설치 경로를 확인하지 못했습니다.
)

:VSCODE_PATH_SETUP
echo.
echo ======================================================================
echo   [5/5] VS Code 환경 변수(PATH) 등록 및 Extension 설치
echo ======================================================================
echo.

if defined VSCODE_BIN (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$vb = '%VSCODE_BIN%'; $up = [Environment]::GetEnvironmentVariable('Path', 'User'); if ($vb -and (Test-Path $vb) -and ($up -split ';' -notcontains $vb)) { [Environment]::SetEnvironmentVariable('Path', ($up + ';' + $vb).Trim(';'), 'User'); Write-Host '  VS Code 경로가 사용자 PATH에 등록되었습니다.' } else { Write-Host '  VS Code 경로가 이미 PATH에 등록되어 있습니다.' }"
)

if defined CODE_CMD (
    echo.
    echo [설치] VS Code 확장 프로그램 [Python, Jupyter] 설치 중...
    call "!CODE_CMD!" --install-extension ms-python.python
    call "!CODE_CMD!" --install-extension ms-toolsai.jupyter
    echo [완료] Extension 설정 완료!
) else (
    echo [경고] VS Code 명령어를 찾지 못해 확장을 설치하지 못했습니다.
)

echo.
echo ======================================================================
echo  [성공] 모든 설정이 성공적으로 완료되었습니다!
echo ======================================================================
echo.
echo  - 가상환경: sbai
echo  - Python 버전:
call "%CONDA_BAT%" run -n sbai python --version
echo  - Jupyter 커널: Python 3.12 (sbai)
echo  - 환경 변수 등록 완료: Conda 및 VS Code
echo.
echo  작업을 시작하려면 'run_sbai_env.bat'을 더블 클릭하세요.
echo ======================================================================
echo.
pause