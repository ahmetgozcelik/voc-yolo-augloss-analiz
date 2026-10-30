import csv, json, re, sys, glob, os
from collections import Counter
D = sys.argv[1]  # data dir
VOC = os.path.join(D, "VOC/images/VOCdevkit/VOC2007")
rx = re.compile(r"staticflickr\.com/\d+/(\d+)_")
# VOC2007: flickrid per image, split membership
test_ids = set(open(os.path.join(VOC, "ImageSets/Main/test.txt")).read().split())
voc = {}
for f in glob.glob(os.path.join(VOC, "Annotations/*.xml")):
    s = open(f, encoding="utf-8").read()
    m = re.search(r"<flickrid>(\d+)</flickrid>", s)
    if m:
        stem = os.path.basename(f)[:-4]
        voc[m.group(1)] = "test" if stem in test_ids else "trainval"
# COCO
coco = {}
for sp in ("train2017", "val2017"):
    for im in json.load(open(os.path.join(D, f"cakisma/coco/annotations/captions_{sp}.json")))["images"]:
        m = rx.search(im.get("flickr_url", ""))
        if m: coco[m.group(1)] = sp
# Open Images
oi = {}
for sp, fn in (("train", "train-images-boxable-with-rotation.csv"), ("validation", "validation-images-with-rotation.csv"), ("test", "test-images-with-rotation.csv")):
    with open(os.path.join(D, "cakisma/oi", fn), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            u = r["OriginalLandingURL"] or ""
            m = re.search(r"flickr\.com/photos/[^/]+/(\d+)", u) or rx.search(r["OriginalURL"] or "")
            if m: oi[m.group(1)] = (sp, r["ImageID"])
out = {"n": {"voc07": len(voc), "coco": len(coco), "oi": len(oi)}}
def kes(a, b, fa, fb):
    k = set(a) & set(b)
    return {"toplam": len(k), "kirilim": Counter(f"{fa(a[i])}|{fb(b[i])}" for i in k)}
out["voc07_oi"] = kes(voc, oi, lambda v: v, lambda v: v[0])
out["voc07_coco"] = kes(voc, coco, lambda v: v, lambda v: v)
out["coco_oi"] = kes(coco, oi, lambda v: v, lambda v: v[0])
json.dump({"oi_voc07_cakisan_imageid": sorted(oi[i][1] for i in set(voc) & set(oi)),
           "oi_coco_cakisan_imageid": sorted(oi[i][1] for i in set(coco) & set(oi))},
          open(os.path.join(D, "cakisma/cakisan_oi_imageid.json"), "w"))
print(json.dumps(out, indent=1, default=dict))
