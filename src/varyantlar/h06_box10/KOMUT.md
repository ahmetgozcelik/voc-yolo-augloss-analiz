Tam koşum (seed S, 100 ep):
`WS/env/.venv/Scripts/python.exe WS/varyantlar/h06_box10/kos.py --seed S --epochs 100 --fraction 1.0 --imgsz 640 --batch 16 --workers 4 --override box=10.0 --run-dir {RUN_DIR} --out {RUN_DIR}/metrics.json`
(kuyruk.py ile; `{RUN_DIR}` kosum.py tarafından çözülür.) Doğrudan baseline koşucusu: `python WS/scripts/rakip_kos.py --seed S --override box=10.0 ...`
