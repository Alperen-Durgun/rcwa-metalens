@echo off
chcp 65001 >nul
REM Creates desktop shortcuts for the .bat launchers in this folder.
REM Run it once after cloning the project on a new computer.
set "HERE=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$h = $env:HERE; $d = [Environment]::GetFolderPath('Desktop'); $W = New-Object -ComObject WScript.Shell;" ^
  "@(@('RCWA Environment.lnk','setup_env.bat'),@('RCWA Tests.lnk','run_tests.bat'),@('RCWA Run Experiments.lnk','run_experiments.bat')) | ForEach-Object {" ^
  "$s = $W.CreateShortcut((Join-Path $d $_[0])); $s.TargetPath = (Join-Path $h $_[1]); $s.WorkingDirectory = $h; $s.Save() }"
echo Desktop shortcuts created.
pause
