@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PIP_DISABLE_PIP_VERSION_CHECK=1
if not exist input mkdir input
if not exist output mkdir output

rem ---------- 1) find Python 3.10+ (install it automatically if missing) ----------
set "PY="
call :findpy
if not defined PY (
  echo Python 3.10+ was not found. Trying to install it automatically...
  where winget >nul 2>nul && winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
  call :findpy
)
if not defined PY (
  echo.
  echo Could not install Python automatically.
  echo Install Python 3.10 or newer from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH"^) and double-click START.bat again.
  pause & exit /b 1
)

rem ---------- 2) one-time setup: private environment + libraries ----------
if exist ".venv\Scripts\python.exe" ".venv\Scripts\python.exe" -c "import numpy, PIL" >nul 2>nul && goto run
echo First run: downloading the required libraries ^(one time only, needs internet^)...
if exist .venv rmdir /s /q .venv
%PY% -m venv .venv || goto setup_failed
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul
set TRIES=0
:pip_try
".venv\Scripts\python.exe" -m pip install -r scripts\requirements.txt && goto pip_ok
set /a TRIES+=1
if %TRIES% lss 3 (
  echo Download failed, retrying...
  timeout /t 3 >nul
  goto pip_try
)
goto setup_failed
:pip_ok
rem Optional: native .7z support. If it fails, the built-in Windows tar is used instead.
".venv\Scripts\python.exe" -m pip install -r scripts\requirements-optional.txt >nul 2>nul

:run
rem Make sure the optional 7z library is present (cheap check, installs only once).
".venv\Scripts\python.exe" -c "import py7zr" >nul 2>nul || ".venv\Scripts\python.exe" -m pip install -r scripts\requirements-optional.txt >nul 2>nul
echo.
echo Put face files in the "input" folder, then press Enter.
pause >nul
".venv\Scripts\python.exe" scripts\batch.py %*
echo.
echo Results are in the "output" folder.
pause
exit /b 0

:setup_failed
echo.
echo Setup failed. Check your internet connection and double-click START.bat again.
pause
exit /b 1

:findpy
for %%C in ("py -3" "python" "python3") do (
  if not defined PY (
    %%~C -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul && set "PY=%%~C"
  )
)
if not defined PY for %%D in ("%LocalAppData%\Programs\Python\Python312\python.exe" "%LocalAppData%\Programs\Python\Python313\python.exe" "%LocalAppData%\Programs\Python\Python311\python.exe" "%ProgramFiles%\Python312\python.exe") do (
  if exist "%%~D" if not defined PY set PY="%%~D"
)
exit /b 0
