#!/usr/bin/env bash
# runs/BILDIRIM.log'a yeni bir terminal olay satırı (BİTTİ / ÖLDÜ / ASILI / KUYRUK-BİTTİ) düşene kadar bekler, satırı basıp çıkar.
# Ayrıca zincir sürecinin canlı olduğunu denetler (ek-zincir.log'da ZINCIR-BITTI yoksa ve 'kuyruk.py' süreci yoksa ALARM).
# Kullanım: bash scripts/ek_deney/olay_bekle.sh   (WS kökünden; Claude Code'da run_in_background ile)
cd "$(dirname "$0")/../.."
L=runs/BILDIRIM.log
n0=$(wc -l < "$L")
while true; do
  n=$(wc -l < "$L")
  if [ "$n" -gt "$n0" ]; then
    yeni=$(tail -n +$((n0 + 1)) "$L" | grep -E "BİTTİ|ÖLDÜ|ASILI|KUYRUK-BİTTİ" )
    if [ -n "$yeni" ]; then echo "$yeni" | sed -E 's/ (val_|sure=).*//'; exit 0; fi  # körlük (ON-KAYIT v5b Ö1): metrik basılmaz
    n0=$n
  fi
  if ! grep -q "ZINCIR-BITTI" runs/kuyruklar/ek-zincir.log 2>/dev/null; then
    if ! powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*kuyruk.py*kos*' }) { exit 0 } else { exit 1 }" >/dev/null 2>&1; then
      echo "ALARM: zincir süreci yok ve ZINCIR-BITTI yazılmamış"; exit 2
    fi
  fi
  sleep 60
done
