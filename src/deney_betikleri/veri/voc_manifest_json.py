"""data_manifest.json'ı üretilmiş dosyalardan derler + manifest için sınıf tablosunu (markdown) basar.
Kullanım: env/.venv/Scripts/python.exe scripts/veri/voc_manifest_json.py"""
import csv, json
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "data"
oz = json.loads((D / "_split_denetim_ozet.json").read_text(encoding="utf-8"))
lk = json.loads((D / "sizinti_raporu.json").read_text(encoding="utf-8"))
meta = json.loads((D / "splits/split_meta.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader(open(D / "sinif_dagilimi.csv", encoding="utf-8")))
by = {}
for r in rows:
    by.setdefault(r["class"], {})[r["split"]] = r
m = {
  "dataset": "PASCAL VOC 2007+2012 (detection, Main)", "topic": "2.1 / cv-1",
  "archives": {
    "VOCtrainval_06-Nov-2007.tar": {"bytes": 460032000, "md5": "c52e279531787c972589f7e41ab4ae64", "md5_official_torchvision": "c52e279531787c972589f7e41ab4ae64", "sha256": "7d8cd951101b0957ddfd7a530bdc8a94f06121cfc1e511bb5937e973020c7508"},
    "VOCtest_06-Nov-2007.tar": {"bytes": 451020800, "md5": "b6e924de25625d8de591ea690078ad9f", "md5_official_torchvision": "b6e924de25625d8de591ea690078ad9f", "sha256": "6836888e2e01dca84577a849d339fa4f73e1e4f135d312430c4856b5609b4892"},
    "VOCtrainval_11-May-2012.tar": {"bytes": 1999639040, "md5": "6cd6e144f989b92b3379bac3b3de84fd", "md5_official_torchvision": "6cd6e144f989b92b3379bac3b3de84fd", "sha256": "e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb"}},
  "raw_counts": {"VOC2007_JPEG": 9963, "VOC2007_XML": 9963, "VOC2012_JPEG": 17125, "VOC2012_XML": 17125,
                 "VOC2012_not_in_Main_trainval_left_in_devkit": 5585},
  "objects_from_xml": {"VOC2007_trainval": {"all": 15662, "difficult": 3054, "non_difficult": 12608},
                       "VOC2007_test": {"all": 14976, "difficult": 2944, "non_difficult": 12032},
                       "VOC2012_trainval": {"all": 31561, "difficult": 4111, "non_difficult": 27450}},
  "converted": {"images_jpg": 21503, "label_txt": 21503, "label_lines_total": 52090, "label_vs_xml_mismatch": oz["label_mismatch_count"]},
  "disk_GB": {"raw_tars": 2.911, "images_dir_incl_devkit": 2.863, "labels": 0.003, "data_VOC_total_du_h": "5.6G"},
  "split": meta, "stats": oz["stats"],
  "leakage": {k: v for k, v in lk.items() if not k.endswith("detail") and k != "ncc_top15"},
  "loader_test": {"ultralytics": "8.4.165", "train": {"images": 15551, "objects": 37651}, "val": {"images": 1000, "objects": 2407}, "nc": 20},
  "per_class": {c: {s: {"images": int(v["images"]), "objects": int(v["objects"]), "difficult_dropped": int(v["difficult_objects_dropped"]), "small": int(v["small_lt32sq"]), "medium": int(v["medium_32_96"]), "large": int(v["large_ge96sq"])} for s, v in d.items()} for c, d in by.items()},
}
(D / "data_manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
print("| id | sınıf | train img | train obj | val img | val obj | test img | test obj | train difficult | train small/med/large |")
print("|---|---|---|---|---|---|---|---|---|---|")
for c, d in by.items():
    t, v, s = d["train"], d["val"], d["test"]
    print(f"| {t['class_id']} | {c} | {t['images']} | {t['objects']} | {v['images']} | {v['objects']} | {s['images']} | {s['objects']} | {t['difficult_objects_dropped']} | {t['small_lt32sq']}/{t['medium_32_96']}/{t['large_ge96sq']} |")
