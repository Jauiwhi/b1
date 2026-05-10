@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ================================================
echo          AI Document Processing System
echo ================================================
echo.

:check_python
python --version >nul 2>&1
if %errorlevel% neq 0 goto :python_not_found

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set pyver=%%v
echo [INFO] Python detected: %pyver%

for /f "tokens=1,2 delims=." %%a in ("%pyver%") do (
    set major=%%a
    set minor=%%b
)

if %major% LSS 3 goto :python_old
if %major% EQU 3 if %minor% LSS 8 goto :python_old
goto :check_deps

:python_not_found
echo [ERROR] Python not found!
echo Please download from: https://www.python.org/downloads/
echo.
start https://www.python.org/downloads/
pause
exit /b 1

:python_old
echo [ERROR] Python version too old. Need 3.8+
echo Please download latest: https://www.python.org/downloads/
echo.
start https://www.python.org/downloads/
pause
exit /b 1

:check_deps
echo.
echo [CHECK] Verifying dependencies...
python check_deps.py
if %errorlevel% equ 0 goto :check_model
echo [WARNING] Some dependencies missing!

:install_deps
echo.
echo Select install source:
echo   1 - Tsinghua mirror (Recommended)
echo   2 - Alibaba mirror
echo   3 - Default PyPI
echo   0 - Exit
echo.
set /p choice="Choice [1-3, 0]: "

if "%choice%"=="1" goto :install_1
if "%choice%"=="2" goto :install_2
if "%choice%"=="3" goto :install_3
if "%choice%"=="0" exit /b 0
goto :install_deps

:install_1
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
goto :check_result

:install_2
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
goto :check_result

:install_3
pip install -r requirements.txt
goto :check_result

:check_result
if %errorlevel% neq 0 (
    echo [ERROR] Install failed. Check network and try again.
    pause
    exit /b 1
)
goto :check_deps

:check_model
echo.
echo [CHECK] Testing model connection...
python -c "from models.factory import ModelFactory; m=ModelFactory(); r=m.check_model_connection(); print('Model OK' if r['connected'] else 'Model Failed')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Model connection test failed. Will retry during processing.
)

:show_menu
echo ================================================
echo           MAIN MENU
echo ================================================
echo.
echo   [1] Problem 1 - Dataset1 Classification
echo   [2] Problem 2 - Dataset2/3 Processing
echo   [3] Problem 3 - Priority Ranking
echo   [4] Run All (1-2-3)
echo   [5] Edit config.yaml
echo.
echo   [0] Exit
echo.
set /p choice="Select [0-5]: "

if "%choice%"=="1" goto :run_p1
if "%choice%"=="2" goto :run_p2
if "%choice%"=="3" goto :run_p3
if "%choice%"=="4" goto :run_all
if "%choice%"=="5" goto :edit_config
if "%choice%"=="0" goto :exit_prog
echo Invalid choice!
goto :show_menu

:run_p1
echo.
echo [RUN] Problem 1...
python main.py --problem 1
echo.
pause
goto :show_menu

:run_p2
echo.
echo [RUN] Problem 2...
python main.py --problem 2
echo.
pause
goto :show_menu

:run_p3
echo.
echo [RUN] Problem 3...
python main.py --problem 3
echo.
pause
goto :show_menu

:run_all
echo.
echo [RUN] All Problems...
python main.py --problem all
echo.
pause
goto :show_menu

:edit_config
start notepad config.yaml
echo [INFO] Config opened. Save changes after editing.
echo.
pause
goto :show_menu

:exit_prog
echo.
echo ================================================
echo              Goodbye!
echo ================================================
endlocal
exit /b 0
