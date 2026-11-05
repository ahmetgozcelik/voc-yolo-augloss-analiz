# A multi-seed evaluation of augmentation and loss interventions in real-time YOLO fine-tuning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23797460.svg)](https://doi.org/10.5281/zenodo.23797460)

This repository contains the code, pre-registration documents, seed-level results, split lists, run logs and Online
Resource 1 (Electronic Supplementary Material) of the following article.

> Ahmed Gürkan Özçelik, Ufuk Asıl. *A Multi-Seed Evaluation of Augmentation and Loss Interventions in Fine-Tuning
> Real-Time YOLO Detectors: A Study on PASCAL VOC and KITTI* (Turkish title: *Gerçek Zamanlı YOLO Modellerinin İnce
> Ayarında Artırma ve Kayıp Müdahalelerinin Çok Tohumlu Değerlendirmesi: PASCAL VOC ve KITTI Üzerine Bir İnceleme*).
> Ostim Technical University. Zenodo archive (all versions): https://doi.org/10.5281/zenodo.23797460

## Summary

Seven training-time interventions in the PASCAL VOC 07+12 fine-tuning of COCO-pretrained YOLO11n (multi-scale training,
scale amplitude, Inner-CIoU, box gain, Varifocal classification loss, box-based copy-paste and its dose) and training-free
inference options (NMS threshold, Soft-NMS, test resolution, TTA, WBF), and the same cards on KITTI with YOLO11s and
YOLO26n, were evaluated with a three-seed protocol with locked test sets, tied to decision rules written in advance. On
VOC, none of the seven interventions passed the acceptance threshold on validation (0/7); on KITTI, one of ten
improvement tests passed (1/10), and this acceptance was sensitive to checkpoint selection. The article shows that
small-gain claims could not be separated from noise under a typical three-seed fine-tuning budget and documents four
noise sources: seed variance, validation sampling, single-seed screening and checkpoint selection. The results hold for
untuned single-point interventions against an untuned default recipe. Details and all caveats are in the article
(`paper/main.pdf`).

## Folders

| Folder | Contents |
|---|---|
| `paper/` | Article source (`main.tex`, in-article appendix `appendix.tex`), Online Resource 1 (`esm.tex`, `esm_A.tex`–`esm_D.tex`), `tables/`, `figures/` (+ `figures/data/`), `references.bib`, compiled `main.pdf` and `esm.pdf`; Turkish version archived in `paper/tr/` |
| `preregistration/` | Hypothesis cards and VOC decision rules (`voc/`), K5 test plan, plausibility anchor; KITTI and H05 rule documents (`kitti_ve_h05/`), queue expectation files (`beklenti/`), SHA256 record |
| `src/` | Intervention patches and unit tests (`varyantlar/`), training/evaluation/analysis scripts (`deney_betikleri/`), table generators and number checks (`tablo_ureticileri/`), figure generators (`sekil_ureticileri/`) |
| `results/` | Per-run metric files (`runs/<run>/metrics.json`), analysis outputs (`sonuclar/`), table and figure inputs (`evidence.json`) |
| `splits/` | VOC train/validation/test lists (`voc/`), KITTI scene-group split, test groups, `DontCare` regions and SHA256 digest (`kitti/`) |
| `logs/` | Experiment log (`deney-gunlugu.md`), run configurations (`runs/<run>/config.json`), run queues (`kuyruk/`) |
| `env/` | `requirements.txt`, installation and environment information, Ultralytics version pin |

Folder and file names inside the repository are in Turkish, as in the original working environment.

## Table and figure → file mapping

Article (`paper/main.pdf`):

| Item | Source file | Generator / input |
|---|---|---|
| Table 1 (hypothesis cards) | `paper/main.tex` | `preregistration/voc/H01.md`–`H08.md` |
| Table 2 (level of evidence) | `paper/main.tex` | article text (claim audit codes) |
| Table 3 (baseline plausibility) | `paper/tables/T1_replikasyon.tex` | `src/tablo_ureticileri/build_tables.py` ← `results/evidence.json` |
| Table 4 (main VOC table) | `paper/tables/T_voc_ana.tex` | `src/tablo_ureticileri/build_tables.py` |
| Table 5 (KITTI test) | `paper/tables/T7_kitti_test.tex` | `src/tablo_ureticileri/build_ek_tables.py` |
| Figure 1 (protocol) | `paper/figures/f0_protokol.tex` | hand-written TikZ |
| Figure 2 (Δ and intervals) | `paper/figures/f1_nokta_aralik.tex` | `src/sekil_ureticileri/f1_nokta_aralik.py` |
| Figure 3 (accuracy–latency) | `paper/figures/f3_pareto.tex` | `src/sekil_ureticileri/f3_pareto.py` |

Online Resource 1 (`paper/esm.pdf`): Table A1 `T6_deney_kaydi.tex`; Figure A1 `f4_butce.tex`; Table B1 `T_ekb_mutlak.tex`;
Table B2 `T_ekb_test_ikincil.tex`; Table B3 `T4_h07_gecikme.tex`; Figure B1 `f2_hata_turu.tex`; Figure B2 `f5_doz.tex`;
Figure B3 `f6_nitel.tex`; Tables C1–C5 `T8_kitti_ikincil.tex`, `T9_kitti_epoch.tex`, `T10_kitti_test_ek.tex`,
`T11_kitti_sinif.tex`, `T12_kitti_saglamlik.tex`; Figure C1 `f7_best_last.tex`.

## Reproduction

1. Environment: Python 3.11, `pip install -r env/requirements.txt` (Ultralytics 8.4.165, commit `004db34`; details in
   `env/ultralytics-kurulum.md`). Runs were done on a single NVIDIA RTX 5050 (8 GB) under Windows 10.
2. Data: PASCAL VOC 2007/2012 and KITTI 2D object detection must be downloaded from their official sites (this
   repository contains no images). Splits are in `splits/`; VOC conversion is done with `src/deney_betikleri/veri/`, KITTI
   preparation with `src/deney_betikleri/ek_deney/kitti_hazirla.py`.
3. Training and evaluation: the full command and parameters of each run are in `logs/runs/<run>/config.json` and
   `logs/kuyruk/*.jsonl`; interventions run through `src/varyantlar/<card>/kos.py`. One VOC run takes ≈3 GPU hours.
4. Tables and figures: `src/tablo_ureticileri/build_tables.py` and `build_ek_tables.py` generate the tables under
   `paper/tables/` from `results/evidence.json`; `src/sekil_ureticileri/*.py` generate the figure data. The article is
   compiled in `paper/` with `latexmk -pdf main.tex; latexmk -pdf esm.tex; latexmk -pdf main.tex`.

Known limitations: the scripts were written for the original workspace layout; local absolute paths are replaced in this
repository by the placeholders `<WS>` (experiment workspace), `<MW>` (manuscript workspace), `<REPO>` and `<YEREL_YOL>`
(local path) and must be filled with local paths before re-running. The image tiles of the qualitative panel (Figure B3)
derive from VOC test images and are not in the repository; `esm.tex` does not compile without this figure, which is
included in `paper/esm.pdf`.

## Not included

Dataset images (VOC, KITTI), model weights (`.pt`), raw prediction dumps (`*preds*.json`, `.npz`), large raw run folders
and caches. These are not needed to recompute the numbers in the article: the numbers are derived from
`results/runs/*/metrics.json`, `results/sonuclar/` and `results/evidence.json`.

## Pre-registration and timestamps

Hypothesis cards and decision rules were written before the runs; the only exception is that the KEEP rule (TUT in the
original records) froze in round 2, after the H01–H03 results. This cannot be verified independently: file timestamps are
not cryptographic, the cards were not uploaded to a public pre-registration repository (OSF, Zenodo), version control
started on 2026-10-16 with a single bulk commit, and the KITTI rule document is recorded only by its SHA256 digest in the
experiment log (`preregistration/SHA256-KAYDI.md`). Six VOC card files (H01, H02, H04, H05, H06, H08) were rewritten in
bulk on 2026-10-03, after the screening runs and the opening of the locked test; their existence before the runs is
supported only by indirect records (queue expectation and ranking files, the experiment log, and the timestamps of code
and counter files). All nine deviations from the rules written in advance are listed in the Limitations section of the
article.

## Citation

See `CITATION.cff`.

## License

Code (`src/` and scripts): GNU AGPL-3.0 (`LICENSE`); the intervention patches build on AGPL-3.0-licensed Ultralytics code.
Text, tables, figures, hypothesis cards and other documents: CC BY 4.0 (`LICENSE-docs`). The datasets are subject to their
own licenses (KITTI: CC BY-NC-SA 3.0).

## Use of AI tools

Use of AI tools: The authors used AI-based tools at various stages of this work (e.g., code development, analysis and
language editing). All outputs were reviewed and verified by the authors, who take full responsibility for the content
of the article.
