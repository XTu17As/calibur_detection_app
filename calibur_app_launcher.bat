@echo off
setlocal enabledelayedexpansion

TITLE Calibur App - System Check & Monitor

:: --- LOG FILE SETUP ---
set "LOGFILE=%~dp0calibur_install.log"
echo ======================================== > "%LOGFILE%"
echo Starting Calibur App - %date% %time% >> "%LOGFILE%"
echo ======================================== >> "%LOGFILE%"

set "RESTART_REQUIRED=0"

echo ========================================
echo        CALIBUR APP - AUTO-INSTALLER
echo ========================================
echo.

:: --- 1. CHECK/INSTALL PYTHON ---
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [MISSING] Python not found. Installing...
    curl -L "https://www.python.org/ftp/python/3.10.8/python-3.10.8-amd64.exe" -o "%TEMP%\python_installer.exe"
    start /wait "" "%TEMP%\python_installer.exe" /quiet InstallAllUsers=1 PrependPath=1
    del "%TEMP%\python_installer.exe"
    set "RESTART_REQUIRED=1"
)

:: --- 2. CHECK/INSTALL NODE.JS ---
node -v >nul 2>&1
if %errorlevel% neq 0 (
    echo [MISSING] Node.js not found. Installing...
    curl -L "https://nodejs.org/dist/v22.12.0/node-v22.12.0-x64.msi" -o "%TEMP%\node_installer.msi"
    start /wait msiexec /i "%TEMP%\node_installer.msi" /qn
    del "%TEMP%\node_installer.msi"
    set "RESTART_REQUIRED=1"
)

:: --- 3. REFRESH PATH IF NEEDED ---
if "!RESTART_REQUIRED!"=="1" (
    echo [INFO] Refreshing environment variables...
    for /f "skip=2 tokens=3*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "syspath=%%A %%B"
    for /f "skip=2 tokens=3*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "userpath=%%A %%B"
    set "PATH=!userpath!;!syspath!"
)

:: --- 4. BACKEND SETUP (WITH IMPROVED DEPENDENCY CHECK) ---
echo [INFO] Preparing Backend...
cd backend
if not exist "venv" (
    echo [INFO] Creating Virtual Environment...
    python -m venv venv
)
call venv\Scripts\activate

:: IMPROVED CHECK: Verify each critical package individually
echo [INFO] Checking installed packages...
set "NEEDS_INSTALL=0"

:: Check critical packages using pip show
pip show fastapi >nul 2>&1
if %errorlevel% neq 0 set "NEEDS_INSTALL=1"

pip show torch >nul 2>&1
if %errorlevel% neq 0 set "NEEDS_INSTALL=1"

pip show opencv-python >nul 2>&1
if %errorlevel% neq 0 set "NEEDS_INSTALL=1"

pip show google-api-python-client >nul 2>&1
if %errorlevel% neq 0 set "NEEDS_INSTALL=1"

:: Additional check for CUDA-enabled PyTorch
if "!NEEDS_INSTALL!"=="0" (
    echo [INFO] Verifying CUDA support...
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [WARNING] CUDA not available or torch not CUDA-enabled. Reinstalling...
        set "NEEDS_INSTALL=1"
    )
)

:: Install only if needed
if "!NEEDS_INSTALL!"=="1" (
    echo [INFO] Missing or incompatible dependencies detected. Installing...
    echo [LOG] Installing CUDA 11.8 Torch stack...
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118 --quiet
    echo [LOG] Installing requirements.txt...
    pip install -r requirements.txt --quiet
    echo [OK] Dependencies installed.
) else (
    echo [OK] All dependencies verified and CUDA ready.
)

echo [INFO] Starting Backend Server...
start "CALIBUR_BACKEND_PROC" cmd /c "title CALIBUR_BACKEND_PROC && python main.py"
cd ..

:: --- 5. FRONTEND SETUP ---
echo [INFO] Preparing Frontend...
if not exist "node_modules" (
    echo [INFO] Node modules missing. Installing...
    call npm install --quiet
)
start "CALIBUR_FRONTEND_PROC" cmd /c "title CALIBUR_FRONTEND_PROC && npm run dev"

:: --- 6. LAUNCH ---
echo [INFO] Waiting for services to initialize...
timeout /t 5 >nul
start http://localhost:5173

:: --- 7. MONITORING LOOP ---
cls
echo ============================================================
echo               CALIBUR APP IS NOW RUNNING
echo ============================================================
echo  [STATUS] Backend:  RUNNING (CUDA 11.8)
echo  [STATUS] Frontend: RUNNING
echo.
echo  - To close the app, close the Backend or Frontend windows.
echo  - This window will auto-exit when the app stops.
echo ============================================================

:MONITOR
timeout /t 3 >nul
tasklist /FI "WINDOWTITLE eq CALIBUR_BACKEND_PROC" | find "cmd.exe" >nul
if %errorlevel% neq 0 goto SHUTDOWN
tasklist /FI "WINDOWTITLE eq CALIBUR_FRONTEND_PROC" | find "cmd.exe" >nul
if %errorlevel% neq 0 goto SHUTDOWN
goto MONITOR

:SHUTDOWN
echo.
echo [DETECTION] App component closed. Cleaning up...
taskkill /FI "WINDOWTITLE eq CALIBUR_BACKEND_PROC" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq CALIBUR_FRONTEND_PROC" /T /F >nul 2>&1
echo [INFO] Shutdown complete.
timeout /t 2 >nul
exit
