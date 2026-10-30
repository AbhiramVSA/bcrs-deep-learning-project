# Plan: fp32 box decoder, new evaluation, new training, latency, control, decisions

Date: 2026-10-05. Script: `scripts/pipeline_fp32.sh`. This plan covers the five remaining items for the paper.

## 1. The defect

In bf16 autocast, torchvision `BoxLinearCoder.decode` changes the anchors to the dtype of the regression output. Thus all decoded box coordinates are bf16.

| Coordinate range (px) | bf16 spacing (px) |
|---|---|
| 256 to 512 | 2 |
| 512 to 1024 | 4 |
| 1024 to 2048 | 8 |

A 512 px tile keeps all coordinates below 512. A whole VisDrone frame has coordinates up to 2000. Thus a whole frame gets boxes that are 2 to 4 times less accurate than a tile. The defect also changes the detection scores to bf16, which causes ties in the score order.

The same decode supplies the GIoU loss in `FCOSHead.compute_loss`. Thus the defect also changed the training of the native-resolution detector. The effect on tiles and on the s=4 detector is small, because their coordinates stay below 512.

The correction is in `src/bcrs/models/detector.py`:

- `Float32BoxCoder` encodes and decodes in fp32 (inference and GIoU loss).
- `FCOSFp32Post` does the post-processing (scores, decode, clip, NMS) in fp32, outside autocast.
- `build_detector(..., legacy_bf16_decode=True)` gives the old behavior. Use it only to reproduce old numbers.

`tests/test_box_decode_dtype.py` shows the defect in legacy mode and its absence in the default mode.

## 2. What stays valid and what changes

These findings compare BCRS with BCRS in the same pipeline. Thus they do not depend on the defect:

- the ablation null results,
- learned selection against random and oracle selection,
- the seed spread,
- the tile-cost model,
- the batch-1 latency order.

These comparisons are not valid until we have fp32 numbers:

- BCRS against whole-image inference,
- exhaustive tiling against whole-image inference (the "unexplained" +3.8 AP),
- the uniform-resolution curve, mainly the s=1 point.

## 3. The plan for each item

### Item 1: fp32 decoder and a new evaluation of all results (stage A, eval only, approximately 6 h)

The code change is complete. Stage A evaluates each v2 checkpoint again with the fp32 decoder. Stage A trains nothing.

| Step | Output in `results/` | Old result for comparison |
|---|---|---|
| A1 uniform s4, s2, s1 | `fp32eval_uniform/s{4,2,1}.json` | `p1_uniform_s*_v2` |
| A1 s1 in legacy mode | `fp32eval_uniform/s1_legacy_bf16.json` | `cap_study/whole_s1_cap1000.json` |
| A2 frozen sweep, g=512 | `p3_sweep_v2_t512_fp32` | `p3_sweep_v2_t512` |
| A3 joint sweep | `p3_sweep_joint_v2_fp32` | `p3_sweep_joint_v2` |
| A4 seed sweeps (now with whole image) | `p3_sweep_joint_v2_seed{1,2}_fp32` | `p3_sweep_joint_v2_seed{1,2}` |
| A5 ablation sweeps | `abl_*_sweep_fp32`, `abl_film_fixed_budget_fp32` | same names without `_fp32` |
| A6 frozen sweep, g=320 | `p3_sweep_v2_t320_fp32` | `p3_sweep_v2_t320` |
| A7 comparison | `fp32eval_compare/report.md`, `compare.json` | |

The legacy-mode result in A1 must agree with the committed 15.88 AP. If it agrees, the old and new numbers differ only because of the decoder. `scripts/compare_fp32.py` does this check and writes it in the report.

### Item 2: train the main models again (stage B, approximately 17 h)

We train again in all cases (`RETRAIN=always`, the default). The reason: stage A sees only the evaluation half of the defect. The GIoU loss half is in the weights, and only new training removes it. A small change in stage A does not prove that the v2 weights are correct.

`compare_fp32.py` also writes a decision value (`retrain_suggested`, threshold 0.3 AP). With `RETRAIN=auto`, the pipeline follows that value. With `RETRAIN=never`, the pipeline trains no v3 model.

Recipe v3 is recipe v2 with the fp32 decoder. All other settings are the same. The configs are `configs/*_v3.yaml`.

| Step | Run | Approximate time |
|---|---|---|
| B1 | `p1_uniform_s4_v3` | 0.6 h |
| B2 | `p1_uniform_s2_v3` | 1.3 h |
| B3 | `p1_uniform_s1_v3` | 4 h |
| B4 | `eval_ckpt.py` on B1 to B3 -> `v3eval_uniform/` | 0.3 h |
| B5 | `p2b_scorer_s4v3` | 0.8 h |
| B6 | sweep `p3_sweep_v3_t512` (frozen detectors) | 1 h |
| B7 | `p2c_bcrs_joint_v3` | 5 h |
| B8 | sweep `p3_sweep_joint_v3` | 1 h |
| B9 | `p1_uniform_s1_v3_cont12` + eval (item 4) | 2 h |
| B10 | sweep `p3_sweep_v3_t320` | 1 h |

Stage C (seeds 1 and 2, approximately 12 h) and stage E (three ablations and the fixed-budget sweep, approximately 13 h) follow. Then all tables of the paper come from one generation of models.

### Item 3: latency (stage D, approximately 1.5 h)

We do two things. The paper reports the batch-1 result as a limitation. Stage D also measures in two settings where computation sets the time:

| Measurement | Output in `results/latency_v3/` | Script |
|---|---|---|
| GPU, batch 1 | `gpu_b1.json` | `measure_latency.py` |
| GPU, 1 and 8 frames for each call | `gpu_throughput_b{1,8}.json` | `measure_throughput.py` |
| CPU, fp32, 4 and 16 threads | `cpu_t4.json`, `cpu_t16.json` | `measure_latency.py --device cpu` |

On the laptop CPU, the time agrees with the cost model: one 512 px tile takes 0.27 of the time of a full frame, and the cost model gives 0.273. On the laptop GPU with 4 frames for each call, the forward-pass times have the same order as the cost model. But small inputs use more time than the model gives. The A100 at batch 1 does not show the cost order, because kernel launches and post-processing set the time.

### Item 4: matched-epochs control (step B9)

The first try stopped 25 s after start. A second job (seed 2) ran on the GPU at the same time, because `pipeline_cap.sh` did not wait correctly. Its wait command found the word "DONE" in its own message line. The possible cause is that the GPU had no more free memory. A local run with the same settings (init checkpoint, frozen BN, EMA, scale jitter, channels_last) trained and evaluated correctly. Thus the code is not the cause.

`pipeline_fp32.sh` prevents this error:

- It runs one job at a time and holds a lock file.
- Before each step, it waits until the GPU has enough free memory (20 GB for training, 8 GB for evaluation).
- If a step stops with a CUDA out-of-memory error, it tries the step again, a maximum of 3 times.
- Its end line `=== PIPELINE_FP32_FINISHED` is unique.

The v3 control is step B9. With `RETRAIN=never`, the pipeline runs the v2 control. That control uses `legacy_bf16_decode: true`, so its training agrees with the v2 joint model.

### Item 5: author decisions

`scripts/apply_decisions.py` writes the decisions into `paper/main.tex` and records them in `paper/DECISIONS.md`. Section 6 gives the options.

## 4. Brainstorm: alternatives and the reasons for each choice

| Item | Alternative | Decision and reason |
|---|---|---|
| 1 | Evaluate the full network in fp32 | Rejected. The defect is only in the decode. Full fp32 changes the network numbers too and doubles the time. |
| 1 | fp32 decode only, bf16 scores | Rejected. bf16 scores cause ties in the score order, and COCO AP uses that order. |
| 1 | No legacy mode | Rejected. Without it, we cannot show that the decoder alone causes the change. |
| 2 | Retrain only when stage A shows a large change | Kept as `RETRAIN=auto`, not the default. Stage A cannot see the training half of the defect. |
| 2 | Fine-tune the v2 weights with the fp32 loss | Rejected. The training length and conditions then differ between models. |
| 2 | Retrain only the main chain | Possible. If time is short, stop after stage D. Then label the ablation and seed tables as v2 models with fp32 evaluation. |
| 3 | Limitation text only | Not sufficient. A reviewer will ask if the cost model agrees with any real time. |
| 3 | Jetson measurement | Optional. We have no Jetson. `measure_latency.py` runs on a Jetson without changes if PyTorch is installed. |
| 3 | Faster BCRS code (batched post-processing, TensorRT) | Rejected for this paper. It changes the evaluated pipeline. Give it as future work. |
| 4 | Run the v2 control again | Only with `RETRAIN=never`. With v3, the control must start from the v3 native detector. |

## 5. Rules for the paper after the results

1. Use `python scripts/make_tables.py --gen auto`. It takes the newest generation (v3, then fp32, then v2) and writes the generation in each caption. The fp32 generation never uses bf16 numbers as a substitute.
2. A claim that BCRS is better than dense inference must hold against three references: the v3 s=1 detector, the matched-epochs control, and whole-image inference with the joint detector. The difference must be larger than the range of the three seeds.
3. If exhaustive tiling still gains AP over whole-image inference after the correction, the paper must give the cause. If the cause is not known, the paper must say so.
4. Ablation rows count as different from the full model only if the difference is larger than the seed range.
5. If the main claim does not hold, the paper becomes a measurement paper: the bf16 decode defect, the tile cost on real frames, and the gap between operation count and latency.

## 6. Decisions for the author

### Affiliation

Only you can give this. Give the exact text: department, institution, city, country. If you have no institution, use "Independent Researcher, City, Country".

### Title

| Option | Text |
|---|---|
| A (recommended) | Budget-Constrained Region Selection for Efficient Small-Object Detection in High-Resolution Imagery |
| B | Budget-Conditioned Region Selection for Efficient Small-Object Detection in High-Resolution Imagery (current) |

Option A keeps the acronym BCRS. The ablation (i) shows that the budget input of the scorer has no measurable effect. Thus "Conditioned" describes a part that does not contribute. Stage E tests ablation (i) again on v3. If the v3 result shows an effect, option B is correct.

### Venue and page limit

Facts from the venue pages on 2026-10-05. The deadlines of ICASSP 2027 and WACV 2027 are past.

| Venue | Page limit | Cost | Change to the 9-page draft |
|---|---|---|---|
| IEEE Access | No limit | APC USD 2,160 | None. Move to the IEEE Access template. |
| IEEE GRSL | 5 pages total | Hybrid journal. Fees not checked. | Cut 4 pages. Use the `[journal]` option. |
| IEEE SPL | 4 pages + 1 page of references | Hybrid journal. Fees not checked. | Cut 4 pages. |
| IEEE ICIP 2027 | 4 pages + 1 page of references (rule of earlier years) | Registration | Cut 4 pages. The 2027 call is not published. |

Recommendation: decide the venue after stage A (approximately 6 h after the start). The fp32 numbers show if the main claim stays.

- If the claim stays and you can pay the APC, use IEEE Access. No cut is necessary.
- If the claim stays and you cannot pay the APC, use IEEE GRSL. Its readers work on aerial imagery.
- If the claim does not stay, use IEEE Access for the measurement paper.

Apply the decisions with one command. Then compile the PDF and do the command again with `--venue` to see the page count:

```bash
python scripts/apply_decisions.py --expansion "Budget-Constrained Region Selection" \
    --affiliation "Department, Institution, City, Country" --venue access
```

## 7. Commands

On the A100 server:

```bash
cd /path/to/bcrs-deep-learning-project
source .venv/bin/activate
git pull
DRY=1 bash scripts/pipeline_fp32.sh | grep -E "STEP|RUN|SKIP|DECISION"
nohup bash scripts/pipeline_fp32.sh > runs/pipeline_fp32.log 2>&1 &
```

Progress:

```bash
grep -E "^=== |FAILED|WARN|DECISION|WAIT" runs/pipeline_fp32.log | tail -25
tail -c 400 runs/logs_fp32/$(ls -t runs/logs_fp32 | head -1)
```

To run only some stages, set `STAGES`. Example: `STAGES="A" nohup bash scripts/pipeline_fp32.sh > runs/pipeline_fp32_A.log 2>&1 &`. A second start skips the steps that have results. Set `REDO=1` to do them again.

## 8. Risks

- Other users load the shared GPU. All times in this plan can be 1.5 to 2 times longer.
- A git pull during the run changes the Python code for the next steps. Each result file records its commit. Do not push changes to `src/` while stage B runs.
- The CPU timings are noisy if other jobs use the CPU. Each CPU result records the load average before and after.
