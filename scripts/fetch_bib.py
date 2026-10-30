"""Build paper/references.bib by FETCHING every entry (never from memory).

DOIs -> doi.org content negotiation (BibTeX); arXiv ids -> arxiv.org/bibtex.
Each entry is re-keyed to our cite key and its fetched title is printed next
to the title we EXPECT, so a wrong identifier is caught by eye and by a fuzzy
check. Entries that fail are written to paper/references_FAILED.txt and must
be resolved by a human before they are cited.

    python scripts/fetch_bib.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import requests

# key: (kind, identifier, expected title fragment)
REFS = {
    "akyon2022sahi": ("doi", "10.1109/ICIP46576.2022.9897990", "slicing aided hyper inference"),
    "bejnordi2022salisa": ("arxiv", "2204.02397", "salisa"),
    "berthet2020perturbed": ("arxiv", "2002.08676", "perturbed optimizers"),
    "carion2020detr": ("arxiv", "2005.12872", "end-to-end object detection with transformers"),
    "cheng2023sodsurvey": ("doi", "10.1109/TPAMI.2023.3290594", "small object detection"),
    "deng2021glsan": ("doi", "10.1109/TIP.2020.3045636", "global-local self-adaptive"),
    "du2023ceasc": ("doi", "10.1109/CVPR52729.2023.01291", "adaptive sparse convolutional"),
    "du2018uavdt": ("arxiv", "1804.00518", "unmanned aerial vehicle benchmark"),
    "figurnov2017sact": ("doi", "10.1109/CVPR.2017.194", "spatially adaptive computation time"),
    "gao2018dynamiczoom": ("doi", "10.1109/CVPR.2018.00724", "dynamic zoom-in"),
    "han2022dynamicsurvey": ("doi", "10.1109/TPAMI.2021.3117837", "dynamic neural networks"),
    "huang2022ufpmp": ("doi", "10.1609/aaai.v36i1.19986", "ufpmp-det"),
    "jaderberg2015stn": ("arxiv", "1506.02025", "spatial transformer"),
    "jang2017gumbel": ("arxiv", "1611.01144", "gumbel-softmax"),
    "koyun2022focusdetect": ("doi", "10.1016/j.image.2022.116675", "focus-and-detect"),
    "kuhse2025anytimeyolo": ("arxiv", "2503.17497", "anytime"),
    "li2020dmnet": ("doi", "10.1109/CVPRW50498.2020.00103", "density map guided"),
    "lin2014coco": ("arxiv", "1405.0312", "microsoft coco"),
    "lin2017fpn": ("arxiv", "1612.03144", "feature pyramid networks"),
    "lin2017focal": ("doi", "10.1109/ICCV.2017.324", "focal loss"),
    "lin2023dynamicdet": ("doi", "10.1109/CVPR52729.2023.00608", "dynamicdet"),
    "liu2024yolc": ("doi", "10.1109/TITS.2024.3386928", "yolc"),
    "liu2025esod": ("doi", "10.1109/TIP.2024.3501853", "esod"),
    "najibi2019autofocus": ("doi", "10.1109/ICCV.2019.00984", "autofocus"),
    "perez2018film": ("arxiv", "1709.07871", "film"),
    "rao2021dynamicvit": ("arxiv", "2106.02034", "dynamicvit"),
    "recasens2018zoom": ("doi", "10.1007/978-3-030-01240-3_4", "learning to zoom"),
    "redmon2016yolo": ("arxiv", "1506.02640", "you only look once"),
    "ren2018sbnet": ("doi", "10.1109/CVPR.2018.00908", "sbnet"),
    "ren2015fasterrcnn": ("arxiv", "1506.01497", "faster r-cnn"),
    "singh2018snip": ("doi", "10.1109/CVPR.2018.00377", "scale invariance"),
    "singh2018sniper": ("arxiv", "1805.09300", "sniper"),
    "thavamani2021fovea": ("doi", "10.1109/ICCV48922.2021.01525", "fovea"),
    "thavamani2023lzu": ("doi", "10.1109/CVPR52729.2023.00492", "zoom and unzoom"),
    "wang2023yolov7": ("arxiv", "2207.02696", "yolov7"),
    "wang2020glance": ("arxiv", "2010.05300", "glance and focus"),
    "xie2020ottopk": ("arxiv", "2002.06504", "differentiable top-k"),
    "xu2021adazoom": ("arxiv", "2106.10409", "adazoom"),
    "yang2022querydet": ("doi", "10.1109/CVPR52688.2022.01330", "querydet"),
    "yang2019clusdet": ("doi", "10.1109/ICCV.2019.00840", "clustered object detection"),
    "yu2019slimmable": ("arxiv", "1812.08928", "slimmable"),
    "yu2020tinyperson": ("doi", "10.1109/WACV45572.2020.9093394", "tiny person"),
    "zhu2022visdrone": ("doi", "10.1109/TPAMI.2021.3119563", "detection and tracking meet drones"),
    "zhu2021deformabledetr": ("arxiv", "2010.04159", "deformable detr"),
    # added for the method/implementation sections
    "tian2019fcos": ("doi", "10.1109/ICCV.2019.00972", "fcos"),
    "he2016resnet": ("doi", "10.1109/CVPR.2016.90", "deep residual learning"),
    "rezatofighi2019giou": ("doi", "10.1109/CVPR.2019.00075", "generalized intersection over union"),
    "cordonnier2021patch": ("doi", "10.1109/CVPR46437.2021.00238", "patch selection"),
    "loshchilov2019adamw": ("arxiv", "1711.05101", "decoupled weight decay"),
}


# combining mark (by code point) -> LaTeX accent command
_ACCENT = {0x300: "`", 0x301: "'", 0x302: "^", 0x303: "~", 0x308: '"', 0x30C: "v",
           0x327: "c", 0x307: ".", 0x30A: "r", 0x306: "u", 0x304: "="}
_SPECIAL = {0x131: r"{\i}", 0xF8: r"{\o}", 0xD8: r"{\O}", 0xDF: r"{\ss}", 0x142: r"{\l}", 0x141: r"{\L}",
            0x2013: "--", 0x2014: "---", 0x2019: "'", 0x2018: "`", 0x201C: "``", 0x201D: "''",
            0x2010: "-", 0xA0: " "}


def protect_title(bib: str) -> str:
    """Double-brace the title so IEEEtran.bst keeps acronyms (YOLC, COCO, FCOS)."""
    return re.sub(r"(title\s*=\s*)\{(?!\{)(.*?)\}(\s*,)",
                  lambda m: m.group(1) + "{{" + m.group(2) + "}}" + m.group(3), bib, count=1, flags=re.S)


def sanitize(bib: str) -> str:
    """BibTeX-safe text: month fields are dropped (publishers emit non-standard
    macros such as `june`) and non-ASCII characters become LaTeX accent
    commands, because 8-bit BibTeX splits multi-byte characters when it
    abbreviates first names."""
    import unicodedata
    bib = re.sub(r"^[ \t]*month[ \t]*=[^\n]*\n", "", bib, flags=re.I | re.M)
    bib = re.sub(r",?[ \t]*month[ \t]*=[ \t]*[A-Za-z]+[ \t]*(?=[,}])", "", bib, flags=re.I)
    out = []
    for ch in bib:
        cp = ord(ch)
        if cp < 128:
            out.append(ch)
        elif cp in _SPECIAL:
            out.append(_SPECIAL[cp])
        else:
            d = unicodedata.normalize("NFD", ch)
            if len(d) == 2 and ord(d[1]) in _ACCENT and ord(d[0]) < 128:
                base = r"\i" if (d[0] == "i" and ord(d[1]) != 0x307) else d[0]
                out.append("{\\" + _ACCENT[ord(d[1])] + "{" + base + "}}")
            else:
                out.append("?")
                print(f"  WARNING: unmapped character U+{cp:04X}")
    return protect_title("".join(out))


def fetch(kind: str, ident: str) -> str:
    if kind == "doi":
        r = requests.get(f"https://doi.org/{ident}", headers={"Accept": "application/x-bibtex"}, timeout=30)
    else:
        r = requests.get(f"https://arxiv.org/bibtex/{ident}", timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text.strip()


def main() -> None:
    out, failed = [], []
    for key, (kind, ident, expect) in REFS.items():
        try:
            bib = fetch(kind, ident)
            m = re.search(r"title\s*=\s*[{\"](.+?)[}\"],?\s*(\n|$)", bib, flags=re.I | re.S)
            title = re.sub(r"[{}\s]+", " ", m.group(1)).strip() if m else "?"
            ok = expect.lower() in title.lower().replace("‐", "-")
            bib = re.sub(r"^(@\w+\s*\{)[^,]*,", lambda mm: f"{mm.group(1)}{key},", bib, count=1)
            print(f"{'OK ' if ok else 'CHECK'} {key:<24} <- {title[:90]}")
            if not ok:
                failed.append(f"{key}\t{kind}:{ident}\ttitle mismatch: got '{title}', expected fragment '{expect}'")
            else:
                out.append(f"% source: {kind}:{ident}\n{sanitize(bib)}\n")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {key:<24} {kind}:{ident}  {type(e).__name__}: {e}")
            failed.append(f"{key}\t{kind}:{ident}\t{type(e).__name__}: {e}")
        time.sleep(0.6)
    paper = Path("paper")
    paper.mkdir(exist_ok=True)
    (paper / "references.bib").write_text("\n".join(out), encoding="utf-8")
    (paper / "references_FAILED.txt").write_text("\n".join(failed), encoding="utf-8")
    print(f"\nwrote {len(out)} verified entries; {len(failed)} need attention")
    sys.exit(0)


if __name__ == "__main__":
    main()
