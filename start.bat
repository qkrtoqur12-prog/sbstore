@echo off
cd /d "%~dp0backend"
if errorlevel 1 (
    echo ERROR: Cannot find backend folder: %~dp0backend
    pause
    exit /b 1
)

echo [OK] Backend folder: %CD%

if not exist "venv\Scripts\uvicorn.exe" (
    echo Installing packages...
    C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe -m venv venv
    venv\Scripts\pip.exe install -r requirements.txt
    if errorlevel 1 ( echo ERROR: install failed & pause & exit /b 1 )
)

echo.
echo ============================================
echo  Server: http://localhost:8010
echo  Press Ctrl+C to stop
echo ============================================
echo.

start "" http://localhost:8010
venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8010

echo.
echo [STOPPED] Server has exited.
pause
