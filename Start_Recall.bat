@echo off
setlocal
cd /d "%~dp0apps\desktop" || exit /b 1
where npm.cmd >nul 2>&1 || (
    echo npm.cmd was not found. Install Node.js or add it to PATH.
    exit /b 1
)
echo Starting Recall development app...
call npm.cmd run dev
exit /b %errorlevel%
