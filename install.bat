@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================
echo  MMDetection Object ^& Color Vision Studio
echo  Installer
echo ============================================

set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if not exist "%PY311%" (
    echo Python 3.11 was not found. Downloading Python 3.11.9...
    curl.exe -L --retry 3 -o "%TEMP%\python-3.11.9-amd64.exe" "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    if errorlevel 1 goto :fail
    "%TEMP%\python-3.11.9-amd64.exe" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=1 Include_test=0 Include_pip=1
    if errorlevel 1 goto :fail
)

if not exist "%PY311%" (
    echo Python 3.11 installer finished, but python.exe is missing.
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    "%PY311%" -m venv .venv
    if errorlevel 1 goto :fail
)

call ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)"
if errorlevel 1 (
    echo Existing .venv is not Python 3.11. Delete the .venv folder and run install.bat again.
    goto :fail
)

set "PYTHONUNBUFFERED=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
echo Updating pip...
".venv\Scripts\python.exe" -m pip install -U pip setuptools wheel
if errorlevel 1 goto :fail

nvidia-smi >nul 2>&1
if errorlevel 1 goto :cpu_torch
echo NVIDIA GPU detected. Installing PyTorch for CUDA 12.1.
".venv\Scripts\python.exe" -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto :fail
set "MMCV_URL=https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html"
goto :after_torch
:cpu_torch
echo NVIDIA driver was not found. Installing the CPU build of PyTorch.
".venv\Scripts\python.exe" -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail
set "MMCV_URL=https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html"
:after_torch

echo Pinning NumPy 1.26. Torch 2.1 cannot run with NumPy 2.
".venv\Scripts\python.exe" -m pip install "numpy==1.26.4"
if errorlevel 1 goto :fail

echo Installing MMCV 2.1.0...
".venv\Scripts\python.exe" -m pip install "mmcv==2.1.0" -f %MMCV_URL%
if errorlevel 1 goto :fail

echo Installing MMEngine and MMDetection 3.3.0...
".venv\Scripts\python.exe" -m pip install "mmengine>=0.10.3,<1.0.0" "mmdet==3.3.0"
if errorlevel 1 goto :fail

echo Installing application packages...
".venv\Scripts\python.exe" -m pip install "PySide6>=6.6,<6.10" "opencv-python==4.10.0.84" "scipy>=1.11,<1.14" "pillow>=9,<11"
if errorlevel 1 goto :fail

echo Restoring NumPy 1.26 in case a dependency upgraded it...
".venv\Scripts\python.exe" -m pip install "numpy==1.26.4"
if errorlevel 1 goto :fail

for %%D in (models checkpoints screenshots recordings objects datasets training custom_models custom_models\configs logs assets) do (
    if not exist "%%D" mkdir "%%D"
)

echo Running import, DetInferencer and CUDA smoke test...
".venv\Scripts\python.exe" scripts\smoke_test.py --download-defaults
if errorlevel 1 goto :fail

echo.
echo Installation finished.
echo Start the application with start.bat
pause
exit /b 0

:fail
echo.
echo Installation failed. See the messages above.
pause
exit /b 1
