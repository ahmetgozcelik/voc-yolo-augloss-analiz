# zincir.cmd'yi Claude Code kabuğunun süreç ağacı DIŞINDA başlatır (WMI → WmiPrvSE alt süreci).
# Gerekçe: Claude Code düşük RAM'de kendi arka plan kabuğunu ve alt süreçlerini sonlandırabiliyor (arıza tablosu 2026-09-29, 2026-10-18).
$cmd = Join-Path $PSScriptRoot "zincir.cmd"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c `"$cmd`"" }
"ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId)"
