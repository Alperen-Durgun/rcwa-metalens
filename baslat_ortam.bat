@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo [Kurulum] Sanal ortam olusturuluyor...
  python -m venv .venv
  call ".venv\Scripts\activate.bat"
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call ".venv\Scripts\activate.bat"
)
echo.
echo RCWA calisma ortami hazir. Klasor aciliyor...
start "" explorer "%~dp0"
cmd /k
