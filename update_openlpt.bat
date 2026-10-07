@echo off
title OpenLPT Updater
echo ==========================================
echo       OpenLPT Auto-Updater
echo ==========================================
echo.
echo Usage: update_openlpt.bat [ENV_NAME]   (default: OpenLPT)
echo.
cd /d "%~dp0"

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=OpenLPT"

echo.
echo NOTE: If the process pauses below, please type your password (SSH passphrase or Git account password) and press Enter.
echo.
echo [1/4] Pulling latest code from git...
git pull
if %errorlevel% neq 0 (
    echo [Error] Git pull failed.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/4] Updating the dependencies of '%ENV_NAME%' from conda-forge...
:: mamba where there is one, else conda; the packages install_windows.bat installs (see there)
set "CONDA_CMD=conda"
where mamba >nul 2>nul
if not errorlevel 1 set "CONDA_CMD=mamba"
set "CONDA_REQ=%TEMP%\openlpt_conda_requirements.txt"
findstr /V /C:"pip-only" requirements.txt > "%CONDA_REQ%"
>>"%CONDA_REQ%" echo qt6-webengine
>>"%CONDA_REQ%" echo pywin32
>>"%CONDA_REQ%" echo libblas=*=*openblas
call %CONDA_CMD% install -n %ENV_NAME% --override-channels -c conda-forge --file "%CONDA_REQ%" -y
if errorlevel 1 (
    echo [Error] Updating the conda-forge packages failed.
    pause
    exit /b 1
)

echo.
echo [3/4] Activating Conda Environment '%ENV_NAME%'...
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo [Warning] Failed to activate '%ENV_NAME%'. Trying to proceed with current env...
)

:: requirements.txt's lines marked pip-only (pycine)
for /f "usebackq tokens=1" %%p in (`findstr /C:"pip-only" requirements.txt ^| findstr /V /B /C:"#"`) do (
    pip install "%%p"
    if errorlevel 1 (
        echo [Error] pip install %%p failed.
        pause
        exit /b 1
    )
)

:: Set consistent build environment for Windows
set "CMAKE_GENERATOR=NMake Makefiles"
set "CMAKE_BUILD_TYPE=Release"
set "CMAKE_GENERATOR_INSTANCE="
set "CMAKE_GENERATOR_PLATFORM="
set "CMAKE_GENERATOR_TOOLSET="

:: Activate VS environment: found with vswhere, as install_windows.bat does (any VS 2022 edition or the Build Tools)
echo [INFO] Activating Visual Studio environment...
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_PATH="
if not exist "%VSWHERE%" goto :NoVSWhere
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VS_PATH=%%i"
:NoVSWhere
if not defined VS_PATH (
    echo [Error] Visual Studio C++ build tools not found: run install_windows.bat first.
    pause
    exit /b 1
)
call "%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Clean previous build to avoid generator mismatch
if exist build rmdir /s /q build
if exist openlpt.egg-info rmdir /s /q openlpt.egg-info

echo.
echo [4/4] Re-installing OpenLPT package...
pip install . --no-build-isolation
if %errorlevel% neq 0 (
    echo [Error] Pip install failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ==========================================
echo       Update Successful! 
echo ==========================================
echo Please restart OpenLPT manually.
pause
