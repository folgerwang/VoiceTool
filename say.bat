@echo off
rem 配音台 launcher. First run installs everything (setup.ps1), then runs say.py.
rem   say --list
rem   say -v storyteller "要念的话"
rem   say --web
setlocal
set "HERE=%~dp0"
if not exist "%HERE%venv\Scripts\python.exe" (
  echo [say] no venv yet - running setup.ps1 ...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%setup.ps1"
  if errorlevel 1 exit /b 1
)
"%HERE%venv\Scripts\python.exe" "%HERE%say.py" %*
