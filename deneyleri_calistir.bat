@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  python -m venv .venv
  call ".venv\Scripts\activate.bat"
  pip install -r requirements.txt
) else ( call ".venv\Scripts\activate.bat" )
echo [Otomasyon] Bekleyen RCWA deneyleri calistiriliyor...
python run_experiment.py --all
echo.
echo Bitti. Raporlar klasorunu ve Deney Panosunu kontrol et.
pause
