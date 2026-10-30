# -*- coding: utf-8 -*-
"""H04 CPU testi: RandomPerspective._compute_affine_matrix'un ornekledigi olcek s, scale=0.25 icin [0.75,1.25] ve
scale=0.5 (baseline) icin [0.5,1.5] araliginda mi; kos.py'nin rp_scale_bul yardimcisi Compose icinde scale'i buluyor mu.
Kullanim: CUDA_VISIBLE_DEVICES=-1 python test_scale.py"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ortak"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ultralytics.data.augment import Compose, RandomPerspective  # noqa: E402
import kos  # noqa: E402

img = np.zeros((640, 640, 3), np.uint8)
random.seed(0)
for sc in (0.25, 0.5):
    rp = RandomPerspective(scale=sc, degrees=0.0, translate=0.1, shear=0.0, perspective=0.0)
    s = np.array([rp._compute_affine_matrix(img, (640, 640))[1] for _ in range(20000)])
    print(f"scale={sc}: ornek s min={s.min():.4f} max={s.max():.4f} ort={s.mean():.4f} std={s.std():.4f}")
    assert s.min() >= 1 - sc - 1e-6 and s.max() <= 1 + sc + 1e-6
    assert s.max() - s.min() > 1.9 * sc
    assert kos.rp_scale_bul(Compose([Compose([rp]), rp])) == sc
print("TUM H04 CPU TESTLERI GECTI")
