@echo off
chcp 65001 >nul
REM This script creates desktop shortcuts for the .bat launchers in this folder.
REM Run it once after opening the project on a new computer.
set "HERE=%~dp0"
powershell -NoProfile -Command ^
  "$h=%HERE%; $d=[Environment]::GetFolderPath(Desktop); $W=New-Object -ComObject WScript.Shell;" ^
  "@(@(RCWA Environment.lnk,baslat_ortam.bat),@(RCWA Tests.lnk,testleri_calistir.bat),@(RCWA Run Experiments.lnk,deneyleri_calistir.bat)) | ForEach-Object {" ^
  "$s=$W.CreateShortcut((Join-Path $d $_[0])); $s.TargetPath=(Join-Path $h $_[1]); $s.WorkingDirectory=$h; $s.Save() }"
echo Desktop shortcuts created.
pause
