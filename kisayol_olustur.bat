@echo off
chcp 65001 >nul
REM Bu betik, bu klasordeki .bat launcher lari icin masaustunde kisayol olusturur.
REM Baska bir bilgisayarda vault u actiktan sonra bir kez calistir.
set "HERE=%~dp0"
powershell -NoProfile -Command ^
  "$h=%HERE%; $d=[Environment]::GetFolderPath(Desktop); $W=New-Object -ComObject WScript.Shell;" ^
  "@(@(RCWA Calisma Ortami.lnk,baslat_ortam.bat),@(RCWA Testleri.lnk,testleri_calistir.bat),@(RCWA Deneyleri Calistir.lnk,deneyleri_calistir.bat)) | ForEach-Object {" ^
  "$s=$W.CreateShortcut((Join-Path $d $_[0])); $s.TargetPath=(Join-Path $h $_[1]); $s.WorkingDirectory=$h; $s.Save() }"
echo Masaustu kisayollari olusturuldu.
pause
