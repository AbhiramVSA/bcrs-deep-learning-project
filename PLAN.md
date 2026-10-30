# BCRS Master Research Plan

Source of truth for methodology, metrics, and the experiment matrix. Derived
from the approved proposal (`initial-resources/main.pdf`). Every experiment in
the paper must trace back to a row in the run matrix below; every number in the
paper must be generated from a JSON in `results/`.

**One-sentence contribution.** A single trained detector whose spatial region
selection is conditioned on an explicit inference-time compute budget traces an
accuracy-computation curve that dominates fixed-cost selective baselines on
high-resolution small-object benchmarks.

**Falsifiable central claim.** For budgets B in a mid-range interval, BCRS at
budget B achieves higher AP than (a) uniform-resolution inference at matched
FLOPs and (b) SAHI-style exhaustive slicing at matched FLOPs, using one
parameter set across all budgets. The oracle-selection bound quantifies how
much of any shortfall is intrinsic to budgeted tile selection vs. our scorer.

---

## 1. Evaluation metrics: hierarchy and rationale

The question "is recall more important than F1?" has a concrete answer for
this problem, and the study is designed around it.

**Primary (headline) metrics — what the paper's claims are stated in:**

| Metric | Why |
|---|---|
| COCO-style AP@[0.50:0.95], maxDets=500 | Threshold-free (integrates precision over all recall levels), the standard in this literature, directly comparable to every published VisDrone baseline. maxDets=500 because VisDrone scenes contain hundreds of instances; the COCO default of 100 clips dense scenes. |
| AP50 | Small objects suffer disproportionately from strict IoU thresholds (a 2-px localization error on a 16-px box destroys IoU); AP50 separates "found the object" from "localized it tightly". |
| AP_s / AP_m / AP_l (size-stratified) | The method exists for small objects; AP_s is where the win must show up. AP_l guards against regressions on large objects (the coarse pass handles them). |
| GFLOPs (analytic) + measured latency (median, p95, A100 batch-1) | The cost axis of every curve. FLOPs compose additively over tiles (budget definition); latency validates that FLOPs savings are real on hardware. |

**The headline artifact is not a single number**: it is the AP-vs-GFLOPs curve
traced by sweeping B over one trained model, summarized by (i) area under the
curve over a fixed budget interval, (ii) AP at three named operating points
(tight / medium / loose), (iii) the gap to the oracle-selection curve.

**Why not F1 as primary.** F1 is defined at one confidence threshold and one
operating point; it hides the precision-recall trade-off, is not reported by
the baselines we compare against, and its value depends on an arbitrary
threshold choice. It appears only in the deployment discussion (an operator
picking a threshold on the deployed curve), never in claims.

**Why recall is the binding diagnostic.** The failure mode of downsampling is
*missed* small objects — false negatives, not false positives. An object the
coarse pass misses is recoverable only if its tile is selected. Therefore the
mechanism-level diagnostics are recall-shaped:

- **AR@500 by size** (average recall): tracks the ceiling detection can reach.
- **Selector recall@k**: fraction of positive tiles (tiles containing coarse-
  missed GT) that appear in the top-k selection. This is THE metric for the
  scoring head — it measures exactly what the selector exists to do, and it is
  aligned with end-task AP gain by construction of the self-derived targets.
- **Coarse-pass miss rate** per size bucket: calibrates how much recall is at
  stake at each downsample factor (also generates the "why tiles" motivation
  figure).
- **Oracle-normalized regret**: (AP_oracle(k) − AP_learned(k)) / (AP_oracle(k)
  − AP_random(k)) — isolates scorer quality from the tile abstraction.

Precision-side failure (duplicate/false positives from fusion) is monitored by
AP itself plus a fusion ablation; it is second-order here.

**Statistical rigor.** Headline configs (BCRS, uniform baselines, SAHI): 3
seeds, report mean ± std; curves plotted with per-seed traces in the appendix.
Ablations: 1 seed is acceptable if effect size clearly exceeds seed noise
(quantify seed noise once on the base config in P1). All AP numbers come from
pycocotools — never hand-rolled — via a single shared `evaluate()` entry point.

---

## 2. Architecture and implementation decisions

**Framework: plain PyTorch 2.x + torchvision components, custom training loop.**
Not MMDetection (registry/config indirection fights a nonstandard two-pass
model; mmcv version pinning is fragile on new CUDA), not Ultralytics (AGPL,
trainer abstractions baked in). BCRS needs: two passes sharing weights, a FiLM-
conditioned scorer between them, a custom top-k relaxation in the graph, and
per-tile loss bookkeeping — easiest and cleanest as our own ~2k-line package
built on tested torchvision parts (ResNet+FPN backbone, box ops, NMS). This is
also the strongest position for code release with the paper.

**Base detector.** FCOS-style anchor-free head (focal loss + GIoU + centerness,
GroupNorm in head) on ResNet-50-FPN, P3–P7. A ResNet-18 variant is the fast-
iteration configuration for development and ablations; headline numbers on
ResNet-50 if time permits, else ResNet-18 consistently everywhere (both are
defensible; consistency is what matters). ImageNet-pretrained backbone.

**Geometry (proposal defaults, corrected 2026-09-08).** Coarse downsample s=4.
Tile g=512, overlap w=32 px. The proposal's "≈12 tiles, each ≈0.087 C_full"
assumed 2000×1500 frames; **VisDrone val frames average 0.96 Mpx** (mostly
1360×765 and 1400×1050), so with g=512 the grid has ≈6 tiles on average and
each tile costs ≈0.27 C_full — three tiles already cost a whole native pass,
and the budget dial has only ~6 positions. Measured in the first sweep
(`results/p2a_oracle_sweep`). Consequence: tile size must be chosen relative
to frame area (g≈256–320 restores ~12–20 tiles at ≈0.07–0.1 C_full each), and
the tile-size ablation (iv) moves from stretch to committed. Budget knob:
k ∈ {0..N}; k(B) = floor((B − C_coarse − C_fuse)/c_tile).

**Scorer g_phi.** Four 3×3 convs (GN+ReLU) on the coarsest FPN level, tile-
aligned average pooling to one logit per tile. Budget B (normalized to [0,1]
over the operating range) → 2-layer MLP → FiLM (scale, shift) on scorer
features.

**Selection.** Inference: hard top-k(B). Training: perturbed maximization
(Berthet et al. 2020) — Gumbel noise at temperature tau, small number of Monte
Carlo samples for the soft indicator; straight-through: hard forward, relaxed
backward. Ablation alternates: Gumbel-softmax top-k, optimal-transport top-k,
REINFORCE baseline. We implement perturbed top-k ourselves (~100 lines, unit-
tested against finite-difference gradients);
tuero/perturbations-differential-pytorch is the cross-check reference.

**Fine pass.** Selected tiles cropped at native resolution, batched, processed
by the SAME detector weights (weight sharing = the coarse/fine passes see
different scales as augmentation). Boxes mapped back to frame coordinates.

**Fusion.** Class-aware NMS over (fine detections in selected tiles) ∪ (coarse
detections); suppress fine boxes truncated at non-image tile borders; down-
weight coarse boxes inside selected tiles (the fine pass is authoritative
there).

**Self-derived selection targets.** After each coarse pass: tile target u_i=1
iff the tile contains a GT box unmatched by coarse detections (IoU ≥ 0.5,
correct class, score above threshold). Loss = L_det_coarse + L_det_fine +
lambda1·BCE(sigma(s), u) + lambda2·margin ranking loss. Budgets sampled
k ~ U{k_min..k_max} each iteration (slimmable-network style). Sensitivity to
the IoU/score thresholds defining "unmatched" is a reported experiment
(promised in proposal Section 6).

**Baselines that fall out of our own codebase (no external repro needed):**
- Uniform resolution s ∈ {1,2,4}: our detector, single pass.
- SAHI-style exhaustive slicing: our fine pass on ALL tiles + same fusion.
- Oracle selection: rank tiles by true unmatched-GT count.
- Random-k and objectness-ranked selection: scorer ablations.
- Fixed-k trained / no-FiLM truncation: budget-conditioning ablations.

**External baselines** (ClusDet, DMNet, UFPMP-Det, QueryDet, CEASC, ESOD,
DynamicDet): compared at their published operating points, plotted as points
on our curve figure, with backbone-parity caveats stated. Reproducing them is
explicitly outside the committed core (matches proposal Sections 5.2 and 6).
Exception: **ESOD is set up locally** (official code + released VisDrone
weights, `docs/ESOD_SETUP.md`) so its operating point can be measured under
OUR evaluation protocol rather than copied from the paper.

**Training recipe lessons from ESOD** (their `hyp.visdrone.yaml`; full table in
`docs/ESOD_SETUP.md`). Their YOLOv5m reaches ~0.27–0.30 val AP vs our 0.105,
and the gap decomposes into: COCO-pretrained detector vs ImageNet backbone
only; 50 epochs with mosaic + mixup + scale jitter ±50% vs our 20 epochs with
horizontal flip only; weight EMA; batch 8 at 1536 px. Consequence for P1: if the
frozen-BN control does not close the s1 ≤ s2 gap, the recipe upgrades go in
this order, one variable at a time: (1) scale jitter [0.5, 1.5] + mosaic-style
multi-image crops, (2) 50-epoch schedule + EMA, (3) fcos_r50. Each is a config
flag so the whole uniform curve can be re-run consistently (~1–1.5 h per run).

---

## 3. Datasets

| Dataset | Role | Size | Acquisition |
|---|---|---|---|
| VisDrone-DET 2019 | Primary (committed) | 6,471 train / 548 val / 1,610 test-dev; ~1.8 GB | Scriptable: Ultralytics asset mirror (verified live) — `scripts/download_visdrone.sh` |
| UAVDT-M | Transfer (stretch) | ~40k frames, vehicles; several GB | Zenodo record 14575517 (scriptable) |
| TinyPerson | Transfer (stretch) | 1,610 imgs / 72,651 boxes | ucas-vg TinyBenchmark (Google Drive) |

Protocol: train on VisDrone train, all model selection on VisDrone val,
test-dev touched only for final tables. Cross-dataset transfer (train VisDrone
→ test UAVDT) probes whether the selector learns content vs. dataset layout.

Preprocessing: one-time conversion of VisDrone annotation txt → COCO JSON
(`scripts/prepare_visdrone.py`); "ignored regions" (class 0) masked out of both
training loss and evaluation per VisDrone protocol; classes 1–10 kept; the
`others` class handled per standard VisDrone-DET convention. Converted JSONs
are deterministic artifacts regenerated by script on the server (not committed;
the script + a checksum are).

---

## 4. Experiment phases and run matrix

Phase gates: each phase has an acceptance check before the next starts.

**P0 — Harness (local RTX 3060 + tiny subsets).**
Dataloader, COCO eval wrapper, FCOS training loop.
Gate: overfit a 16-image subset to ~0 loss; eval pipeline reproduces
pycocotools reference numbers on a canned prediction file.

**P1 — Base detector reproduction (A100).**
Uniform-resolution runs s ∈ {1,2,4} on VisDrone; standard schedule
(~15–30 epochs). Gate: the s=1 model lands in the published FCOS/RetinaNet-
class VisDrone band (~28–33 AP on val). This validates the harness AND yields
the trivial trade-off curve BCRS must beat, plus shared pretrained detector
weights for P2.

**P2 — BCRS core (A100).**
Order deliberately de-risked:
 a. Two-pass + ORACLE selection, detector frozen from P1: validates tiling,
    fine pass, fusion, and measures the achievable ceiling per k.
 b. + learned scorer, fixed k, detector frozen: isolates scorer learning.
 c. Full BCRS: joint training, budget sampling, FiLM conditioning.
Gate: (a) the oracle curve sits clearly above the uniform curve (if not, the
tile abstraction itself is broken — stop and diagnose); (c) learned selection
within stated regret of oracle at mid budgets.

**P3 — Budget sweep + headline curves (A100).**
Sweep B over k=0..N on val for BCRS (single model), uniform, SAHI;
3 seeds for headline configs; final numbers on test-dev.

**P4 — Ablations (committed: i–iii; stretch: iv–vi).**
 (i) FiLM budget conditioning vs truncated budget-independent ranking
 (ii) selection target: self-derived vs raw objectness vs density-style
 (iii) gradient estimator: perturbed vs Gumbel-softmax vs OT vs REINFORCE
 (iv) tile size g / overlap w      [stretch]
 (v) weight sharing vs separate fine head   [stretch]
 (vi) budget-sampled vs fixed-k training evaluated off-budget  [stretch]

**P5 — Transfer + latency (stretch).**
UAVDT, TinyPerson; measured latency tables; embedded board only if hardware
appears.

**P6 — Paper.**
NeurIPS-style LaTeX (template from the ml-paper-writing skill), tables and
figures generated by `scripts/make_tables.py` and `scripts/make_curves.py`
straight from `results/` JSONs. Citation workflow: programmatic verification
only.

---

## 5. A100 execution and optimization

**Throughput checklist (applied from P1 onward):**
- bf16 autocast + TF32 matmuls (A100-native), channels_last memory format
- torch.compile on the detector forward (fallback: eager if compile chokes on
  the two-pass graph; compile coarse and fine submodules separately)
- fused AdamW; gradient accumulation only if VRAM-bound
- DataLoader: workers ≈ CPU cores, pinned memory, persistent workers; decoded-
  image cache (LMDB/npy) if JPEG decode becomes the bottleneck (VisDrone frames
  are large); fine-pass tiles cropped on GPU from the already-loaded frame
- batched fine pass across tiles AND across images (flatten [B·k] tiles)
- per-run wall-clock target: ≤ 6 h; checkpoint+resume every epoch;
  run dirs keyed by config name + git commit
- one experiment at a time on the single A100; queue tracked in
  `scripts/queue.md`

**Run bookkeeping.** Every run: `configs/<name>.yaml` (committed) → on server
`python -m bcrs.train --config configs/<name>.yaml` → writes
`results/<name>/metrics.json`, `curve.json`, `train_log.txt` → committed and
pushed from the server by `scripts/push_results.sh`. Checkpoints stay on the
server (gitignored). I read pushed JSONs locally and decide the next runs.

**Division of labor.** All code is developed and smoke-tested locally (RTX
3060, 6 GB: batch 1–2, ResNet-18, ~800 px inputs is enough to validate shapes,
losses decreasing, eval plumbing). The A100 runs only pulled, tagged code.

---

## 6. Rigor / anti-hallucination protocol

1. **No hand-written numbers.** Paper tables/figures are compiled from
   `results/*/metrics.json` by scripts. A claims ledger (`paper/claims.md`)
   maps every claim sentence → result file.
2. **One eval path.** pycocotools behind a single `bcrs.eval` entry point;
   identical settings (maxDets=500, area thresholds) for every method.
3. **Ordering invariants asserted in code:** oracle ≥ learned ≥ random at
   every k; AP(k=N) ≥ AP(k=0) on val; uniform s=1 ≥ s=2 ≥ s=4. Violations
   fail the run loudly instead of producing a quietly wrong table.
4. **Citations:** never from memory; Semantic Scholar/CrossRef verification
   per the skill workflow; placeholders explicitly flagged for human review.
5. **Seeds & determinism:** fixed seeds logged per run; cudnn benchmark on
   (speed) but seed variance quantified once in P1.
6. **Test-dev discipline:** test-dev evaluated only for final tables (twice
   total: once at P3 freeze, once for camera-ready).

---

## 7. Open items owned by the user

1. Paste A100 server specs: `nvidia-smi; nproc; free -h; df -h .;
   python3 --version` (need: VRAM 40 vs 80 GB, CPU cores, disk headroom).
2. Confirm the server can pull the private GitHub repo (deploy key or PAT),
   and can push results back (or paste JSONs manually if push is blocked).
3. ~15 GB free disk on the server for VisDrone + caches (+~10 GB per stretch
   dataset later).
4. Optional: Weights & Biases account for live dashboards (not required; the
   JSON+git loop is the default).
