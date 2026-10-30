"""VOC XML -> YOLO dönüşümü. Ultralytics paketindeki cfg/datasets/VOC.yaml 'download' betiğinin
convert_label fonksiyonu KOPYALANMADAN, yaml'dan okunup exec edilir; dönüşüm döngüsü betikteki ile
birebir aynıdır. Tek fark: indirme adımı (download(urls,...)) atlanır, çünkü resmî tar'lar
curl ile data/VOC/raw'a indirilip MD5 doğrulanarak data/VOC/images altına açıldı.
Kullanım (WS kökünden): env/.venv/Scripts/python.exe scripts/veri/voc_donustur.py
"""
from pathlib import Path
import ultralytics
from ultralytics.utils import YAML, TQDM

WS = Path(__file__).resolve().parents[2]
ycfg = Path(ultralytics.__file__).parent / "cfg/datasets/VOC.yaml"
yaml = YAML.load(ycfg)
yaml["path"] = str(WS / "data/VOC")
src = yaml["download"]
conv_src = src.split("# Download")[0]           # import'lar + convert_label (verbatim)
ns = {"yaml": yaml}
exec(conv_src, ns)
convert_label = ns["convert_label"]

# --- aşağısı VOC.yaml '# Convert' bloğu ile birebir ---
dir = Path(yaml["path"])
path = dir / "images/VOCdevkit"
for year, image_set in ("2012", "train"), ("2012", "val"), ("2007", "train"), ("2007", "val"), ("2007", "test"):
    imgs_path = dir / "images" / f"{image_set}{year}"
    lbs_path = dir / "labels" / f"{image_set}{year}"
    imgs_path.mkdir(exist_ok=True, parents=True)
    lbs_path.mkdir(exist_ok=True, parents=True)
    with open(path / f"VOC{year}/ImageSets/Main/{image_set}.txt") as f:
        image_ids = f.read().strip().split()
    for id in TQDM(image_ids, desc=f"{image_set}{year}"):
        f = path / f"VOC{year}/JPEGImages/{id}.jpg"
        lb_path = (lbs_path / f.name).with_suffix(".txt")
        f.rename(imgs_path / f.name)
        convert_label(path, lb_path, year, id)
print("ultralytics", ultralytics.__version__, "VOC.yaml:", ycfg)
