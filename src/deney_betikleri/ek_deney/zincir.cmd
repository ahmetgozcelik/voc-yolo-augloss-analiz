@echo off
rem Ek deney tam kuyruk zinciri: H05-pos/neg -> YOLO26n -> YOLO11s (ON-KAYIT-ek-deney.md, Butce sirasi).
rem Claude Code kabugundan bagimsiz baslatilir (WMI Win32_Process.Create): scripts\ek_deney\zincir_baslat.ps1
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set PY=<YEREL_YOL>
set K=<REPO>\.claude\skills\konudan-makale\scripts\kuyruk.py
cd /d "<WS>"
set L=runs\kuyruklar\ek-zincir.log
echo ZINCIR-BASLADI %date% %time% >> %L%
for %%Q in (ek-h05pos ek-yolo26n ek-yolo11s) do (
  echo KUYRUK %%Q basladi !date! !time! >> %L%
  "%PY%" "%K%" --kok . kos %%Q >> %L% 2>&1
  echo KUYRUK %%Q bitti rc=!errorlevel! !date! !time! >> %L%
)
echo ZINCIR-BITTI %date% %time% >> %L%
endlocal
