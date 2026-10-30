# -*- coding: utf-8 -*-
"""
H08: H03 `BoxCopyPaste`'in YUKSEK DOZLU hali. h03_boxcp/boxcp.py aynen ice aktarilir (dokunulmaz);
tek fark yapici varsayilani: max_sqrt_alan 48 -> 96 px. p ise `--override copy_paste=1.0` ile gelir
(Ultralytics CopyPaste(p=hyp.copy_paste)). Diger her sey H03 ile birebir ayni.

Sinif BU MODULDE tanimli (Windows spawn pickle: isciler `boxcp_doz` modulunu sys.path'ten ice aktarir).
"""
import sys
from pathlib import Path

_AD = Path(__file__).resolve().parent
_H03 = _AD.parent / "h03_boxcp"
if str(_H03) not in sys.path:
    sys.path.insert(0, str(_H03))
import boxcp  # noqa: E402  (H03 modulu; degistirilmez)

MAX_SQRT_ALAN = 96.0


class BoxCopyPasteDoz(boxcp.BoxCopyPaste):
    """H03 BoxCopyPaste; yalniz max_sqrt_alan varsayilani 96 px."""

    def __init__(self, dataset=None, pre_transform=None, p: float = 1.0, mode: str = "flip",
                 max_sqrt_alan: float = MAX_SQRT_ALAN, **kw) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p, mode=mode,
                         max_sqrt_alan=max_sqrt_alan, **kw)


sayac_topla = boxcp.sayac_topla


def uygula():
    """v8_transforms'un ad olarak cozdugu yerlesik CopyPaste'i BoxCopyPasteDoz ile degistirir."""
    import ultralytics.data.augment as augment
    augment.CopyPaste = BoxCopyPasteDoz
    return augment.CopyPaste
