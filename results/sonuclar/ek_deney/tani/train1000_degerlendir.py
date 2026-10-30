from ultralytics import YOLO
if __name__ == "__main__":
    R="runs/20261019-124541_ek_yolo26n_baseline_s0/train/weights/"
    for w in ("best.pt","last.pt"):
        m=YOLO(R+w); r=m.val(data=r"<YEREL_YOL>", split="val", imgsz=640, batch=16, device=0, plots=False, verbose=False, project=r"<YEREL_YOL>", name=w, exist_ok=True, workers=0)
        print("SONUC", w, "train1000 mAP50-95", round(r.box.map,4), "mAP50", round(r.box.map50,4), "sinif", [round(x,3) for x in r.box.maps], flush=True)
