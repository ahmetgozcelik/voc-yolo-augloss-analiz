@echo off
rem Hakem tur 3: B3(c) KITTI val last.pt tanısı -> H05c kuyruğu (ON-KAYIT-H05c.md). WMI ile bağımsız başlatılır.
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set PY=<YEREL_YOL>
set VPY=<WS>\env\.venv\Scripts\python.exe
set K=<REPO>\.claude\skills\konudan-makale\scripts\kuyruk.py
cd /d "<WS>"
set L=runs\kuyruklar\ek-zincir.log
echo ZINCIR-H05C-BASLADI %date% %time% >> %L%
"%VPY%" scripts\ek_deney\kitti_val_last.py > sonuclar\ek_deney\val-last-tani.log 2>&1
echo VAL-LAST bitti rc=!errorlevel! !date! !time! >> %L%
"%PY%" "%K%" --kok . kos ek-h05c >> %L% 2>&1
echo KUYRUK ek-h05c bitti rc=!errorlevel! !date! !time! >> %L%
echo ZINCIR-BITTI %date% %time% >> %L%
endlocal
