# bf16 vs fp32 box decode: same v2 checkpoints

Units: AP points (AP x 100). Old: the committed v2 numbers (bf16 decode). New: fp32 decode.

## Uniform baselines

| Input | Old source | AP old | AP new | dAP | AP_s old | AP_s new | dAP_s |
|---|---|---|---|---|---|---|---|
| s=4 | training-time eval (best epoch) | 5.78 | 5.97 | 0.19 | 0.75 | 0.73 | -0.02 |
| s=2 | training-time eval (best epoch) | 13.00 | 14.20 | 1.20 | 4.40 | 4.78 | 0.38 |
| s=1 | training-time eval (best epoch) | 15.80 | 19.09 | 3.29 | 6.54 | 10.34 | 3.80 |
| s=1, eval_ckpt.py both modes | eval_ckpt.py --legacy-bf16-decode | 15.88 | 19.09 | 3.21 | 6.59 | 10.34 | 3.75 |

Legacy mode reproduces the committed whole-image s=1 number (results/cap_study): True.

## Gaps that the claims use

| Comparison | Old | New | Change |
|---|---|---|---|
| BCRS learned k=1 minus whole image (p3_sweep_joint_v2_fp32) | -0.50 | -8.58 | -8.08 |
| BCRS learned k=3 minus whole image (p3_sweep_joint_v2_fp32) | 5.45 | -2.66 | -8.11 |
| BCRS learned k=1 minus whole image (p3_sweep_joint_v2_seed1_fp32) | n/a | -8.31 | n/a |
| BCRS learned k=3 minus whole image (p3_sweep_joint_v2_seed1_fp32) | n/a | -2.68 | n/a |
| BCRS learned k=1 minus whole image (p3_sweep_joint_v2_seed2_fp32) | n/a | -8.49 | n/a |
| BCRS learned k=3 minus whole image (p3_sweep_joint_v2_seed2_fp32) | n/a | -2.75 | n/a |
| exhaustive tiling (k=15) minus whole image, frozen detectors | 3.82 | 0.53 | -3.29 |

## Sweeps

| Sweep | Ranking | k | AP old | AP new | dAP | AP_s old | AP_s new | dAP_s |
|---|---|---|---|---|---|---|---|---|
| frozen detectors, g=512 | whole | 0 | 15.77 | 18.92 | 3.15 | 6.53 | 10.18 | 3.65 |
| frozen detectors, g=512 | learned | 0 | 5.83 | 5.94 | 0.11 | 0.78 | 0.74 | -0.04 |
| frozen detectors, g=512 | learned | 1 | 12.15 | 12.20 | 0.05 | 7.34 | 7.34 | 0.00 |
| frozen detectors, g=512 | learned | 2 | 14.66 | 14.65 | -0.01 | 8.65 | 8.62 | -0.03 |
| frozen detectors, g=512 | learned | 3 | 16.98 | 16.93 | -0.05 | 10.37 | 10.29 | -0.08 |
| frozen detectors, g=512 | learned | 4 | 17.73 | 17.68 | -0.05 | 10.54 | 10.50 | -0.04 |
| frozen detectors, g=512 | learned | 6 | 18.79 | 18.69 | -0.10 | 10.61 | 10.56 | -0.05 |
| frozen detectors, g=512 | learned | 8 | 19.55 | 19.42 | -0.13 | 10.86 | 10.80 | -0.06 |
| frozen detectors, g=512 | learned | 15 | 19.57 | 19.45 | -0.12 | 10.82 | 10.77 | -0.05 |
| frozen detectors, g=512 | oracle | 0 | 5.83 | 5.94 | 0.11 | 0.78 | 0.74 | -0.04 |
| frozen detectors, g=512 | oracle | 1 | 12.99 | 13.05 | 0.06 | 8.57 | 8.55 | -0.02 |
| frozen detectors, g=512 | oracle | 2 | 16.26 | 16.27 | 0.01 | 10.27 | 10.23 | -0.04 |
| frozen detectors, g=512 | oracle | 3 | 17.88 | 17.85 | -0.03 | 10.84 | 10.81 | -0.03 |
| frozen detectors, g=512 | oracle | 4 | 18.81 | 18.76 | -0.05 | 10.93 | 10.98 | 0.05 |
| frozen detectors, g=512 | oracle | 6 | 19.54 | 19.42 | -0.12 | 10.88 | 10.85 | -0.03 |
| frozen detectors, g=512 | oracle | 8 | 19.63 | 19.48 | -0.15 | 10.86 | 10.79 | -0.07 |
| frozen detectors, g=512 | oracle | 15 | 19.59 | 19.45 | -0.14 | 10.82 | 10.77 | -0.05 |
| frozen detectors, g=512 | random | 0 | 5.83 | 5.94 | 0.11 | 0.78 | 0.74 | -0.04 |
| frozen detectors, g=512 | random | 1 | 9.56 | 9.64 | 0.08 | 3.90 | 3.89 | -0.01 |
| frozen detectors, g=512 | random | 2 | 11.88 | 11.89 | 0.01 | 5.77 | 5.75 | -0.02 |
| frozen detectors, g=512 | random | 3 | 13.95 | 13.95 | 0.00 | 7.13 | 7.15 | 0.02 |
| frozen detectors, g=512 | random | 4 | 15.23 | 15.19 | -0.04 | 8.01 | 8.05 | 0.04 |
| frozen detectors, g=512 | random | 6 | 17.90 | 17.80 | -0.10 | 9.55 | 9.55 | 0.00 |
| frozen detectors, g=512 | random | 8 | 19.42 | 19.30 | -0.12 | 10.74 | 10.68 | -0.06 |
| frozen detectors, g=512 | random | 15 | 19.58 | 19.45 | -0.13 | 10.82 | 10.77 | -0.05 |
| frozen detectors, g=320 | oracle | 0 | 5.83 | 5.94 | 0.11 | 0.78 | 0.74 | -0.04 |
| frozen detectors, g=320 | oracle | 1 | 9.81 | 9.87 | 0.06 | 5.89 | 5.83 | -0.06 |
| frozen detectors, g=320 | oracle | 2 | 12.28 | 12.17 | -0.11 | 7.82 | 7.76 | -0.06 |
| frozen detectors, g=320 | oracle | 3 | 14.04 | 13.97 | -0.07 | 8.97 | 8.92 | -0.05 |
| frozen detectors, g=320 | oracle | 4 | 15.29 | 15.17 | -0.12 | 9.76 | 9.65 | -0.11 |
| frozen detectors, g=320 | oracle | 6 | 16.85 | 16.74 | -0.11 | 10.39 | 10.30 | -0.09 |
| frozen detectors, g=320 | oracle | 8 | 17.66 | 17.56 | -0.10 | 10.60 | 10.51 | -0.09 |
| frozen detectors, g=320 | oracle | 32 | 18.65 | 18.52 | -0.13 | 10.78 | 10.65 | -0.13 |
| frozen detectors, g=320 | random | 0 | 5.83 | 5.94 | 0.11 | 0.78 | 0.74 | -0.04 |
| frozen detectors, g=320 | random | 1 | 7.09 | 7.16 | 0.07 | 1.96 | 2.01 | 0.05 |
| frozen detectors, g=320 | random | 2 | 8.09 | 8.15 | 0.06 | 2.70 | 2.76 | 0.06 |
| frozen detectors, g=320 | random | 3 | 9.11 | 9.14 | 0.03 | 3.66 | 3.71 | 0.05 |
| frozen detectors, g=320 | random | 4 | 10.23 | 10.25 | 0.02 | 4.57 | 4.58 | 0.01 |
| frozen detectors, g=320 | random | 6 | 12.20 | 12.20 | 0.00 | 5.98 | 6.00 | 0.02 |
| frozen detectors, g=320 | random | 8 | 13.63 | 13.59 | -0.04 | 7.12 | 7.12 | 0.00 |
| frozen detectors, g=320 | random | 32 | 18.65 | 18.52 | -0.13 | 10.77 | 10.65 | -0.12 |
| joint, seed 0 | whole | 0 | 14.15 | 22.42 | 8.27 | 6.76 | 13.67 | 6.91 |
| joint, seed 0 | learned | 0 | 5.03 | 5.11 | 0.08 | 0.52 | 0.55 | 0.03 |
| joint, seed 0 | learned | 1 | 13.65 | 13.84 | 0.19 | 9.39 | 9.62 | 0.23 |
| joint, seed 0 | learned | 2 | 17.29 | 17.44 | 0.15 | 11.49 | 11.72 | 0.23 |
| joint, seed 0 | learned | 3 | 19.60 | 19.76 | 0.16 | 13.01 | 13.17 | 0.16 |
| joint, seed 0 | learned | 4 | 20.52 | 20.68 | 0.16 | 13.38 | 13.54 | 0.16 |
| joint, seed 0 | learned | 6 | 21.84 | 21.95 | 0.11 | 13.76 | 13.89 | 0.13 |
| joint, seed 0 | learned | 8 | 22.40 | 22.52 | 0.12 | 13.97 | 14.11 | 0.14 |
| joint, seed 0 | learned | 15 | 22.42 | 22.54 | 0.12 | 13.99 | 14.13 | 0.14 |
| joint, seed 0 | oracle | 0 | 5.03 | 5.11 | 0.08 | 0.52 | 0.55 | 0.03 |
| joint, seed 0 | oracle | 1 | 14.30 | 14.47 | 0.17 | 10.25 | 10.39 | 0.14 |
| joint, seed 0 | oracle | 2 | 17.97 | 18.09 | 0.12 | 12.50 | 12.63 | 0.13 |
| joint, seed 0 | oracle | 3 | 20.11 | 20.27 | 0.16 | 13.31 | 13.48 | 0.17 |
| joint, seed 0 | oracle | 4 | 21.20 | 21.40 | 0.20 | 13.77 | 13.92 | 0.15 |
| joint, seed 0 | oracle | 6 | 22.18 | 22.32 | 0.14 | 13.97 | 14.11 | 0.14 |
| joint, seed 0 | oracle | 8 | 22.42 | 22.54 | 0.12 | 13.98 | 14.13 | 0.15 |
| joint, seed 0 | oracle | 15 | 22.42 | 22.54 | 0.12 | 13.99 | 14.13 | 0.14 |
| joint, seed 0 | random | 0 | 5.03 | 5.11 | 0.08 | 0.52 | 0.55 | 0.03 |
| joint, seed 0 | random | 1 | 9.74 | 9.88 | 0.14 | 4.69 | 4.80 | 0.11 |
| joint, seed 0 | random | 2 | 12.81 | 12.94 | 0.13 | 7.22 | 7.30 | 0.08 |
| joint, seed 0 | random | 3 | 15.47 | 15.57 | 0.10 | 8.89 | 8.98 | 0.09 |
| joint, seed 0 | random | 4 | 17.20 | 17.33 | 0.13 | 10.15 | 10.28 | 0.13 |
| joint, seed 0 | random | 6 | 20.32 | 20.48 | 0.16 | 12.29 | 12.43 | 0.14 |
| joint, seed 0 | random | 8 | 22.21 | 22.33 | 0.12 | 13.86 | 14.01 | 0.15 |
| joint, seed 0 | random | 15 | 22.43 | 22.54 | 0.11 | 13.98 | 14.12 | 0.14 |
| joint, seed 1 | learned | 0 | 5.11 | 5.16 | 0.05 | 0.57 | 0.56 | -0.01 |
| joint, seed 1 | learned | 1 | 14.01 | 14.14 | 0.13 | 9.83 | 9.94 | 0.11 |
| joint, seed 1 | learned | 2 | 17.35 | 17.49 | 0.14 | 11.78 | 11.88 | 0.10 |
| joint, seed 1 | learned | 3 | 19.65 | 19.77 | 0.12 | 13.21 | 13.32 | 0.11 |
| joint, seed 1 | learned | 4 | 20.81 | 20.86 | 0.05 | 13.68 | 13.79 | 0.11 |
| joint, seed 1 | learned | 6 | 22.01 | 22.11 | 0.10 | 14.04 | 14.14 | 0.10 |
| joint, seed 1 | learned | 8 | 22.50 | 22.59 | 0.09 | 14.20 | 14.30 | 0.10 |
| joint, seed 1 | learned | 15 | 22.54 | 22.63 | 0.09 | 14.21 | 14.31 | 0.10 |
| joint, seed 1 | oracle | 0 | 5.11 | 5.16 | 0.05 | 0.57 | 0.56 | -0.01 |
| joint, seed 1 | oracle | 1 | 14.46 | 14.60 | 0.14 | 10.56 | 10.61 | 0.05 |
| joint, seed 1 | oracle | 2 | 18.14 | 18.25 | 0.11 | 12.76 | 12.81 | 0.05 |
| joint, seed 1 | oracle | 3 | 20.20 | 20.28 | 0.08 | 13.62 | 13.68 | 0.06 |
| joint, seed 1 | oracle | 4 | 21.35 | 21.42 | 0.07 | 14.05 | 14.12 | 0.07 |
| joint, seed 1 | oracle | 6 | 22.33 | 22.40 | 0.07 | 14.23 | 14.30 | 0.07 |
| joint, seed 1 | oracle | 8 | 22.55 | 22.64 | 0.09 | 14.21 | 14.31 | 0.10 |
| joint, seed 1 | oracle | 15 | 22.54 | 22.63 | 0.09 | 14.22 | 14.31 | 0.09 |
| joint, seed 1 | random | 0 | 5.11 | 5.16 | 0.05 | 0.57 | 0.56 | -0.01 |
| joint, seed 1 | random | 1 | 9.86 | 9.91 | 0.05 | 4.86 | 4.95 | 0.09 |
| joint, seed 1 | random | 2 | 12.88 | 12.90 | 0.02 | 7.25 | 7.26 | 0.01 |
| joint, seed 1 | random | 3 | 15.49 | 15.56 | 0.07 | 8.95 | 8.96 | 0.01 |
| joint, seed 1 | random | 4 | 17.23 | 17.34 | 0.11 | 10.35 | 10.41 | 0.06 |
| joint, seed 1 | random | 6 | 20.45 | 20.56 | 0.11 | 12.63 | 12.70 | 0.07 |
| joint, seed 1 | random | 8 | 22.33 | 22.43 | 0.10 | 14.08 | 14.19 | 0.11 |
| joint, seed 1 | random | 15 | 22.53 | 22.63 | 0.10 | 14.21 | 14.31 | 0.10 |
| joint, seed 2 | learned | 0 | 5.11 | 5.20 | 0.09 | 0.49 | 0.50 | 0.01 |
| joint, seed 2 | learned | 1 | 13.70 | 13.84 | 0.14 | 9.42 | 9.55 | 0.13 |
| joint, seed 2 | learned | 2 | 17.26 | 17.38 | 0.12 | 11.48 | 11.59 | 0.11 |
| joint, seed 2 | learned | 3 | 19.47 | 19.58 | 0.11 | 12.87 | 12.92 | 0.05 |
| joint, seed 2 | learned | 4 | 20.46 | 20.56 | 0.10 | 13.22 | 13.26 | 0.04 |
| joint, seed 2 | learned | 6 | 21.86 | 21.94 | 0.08 | 13.74 | 13.79 | 0.05 |
| joint, seed 2 | learned | 8 | 22.36 | 22.43 | 0.07 | 13.92 | 13.97 | 0.05 |
| joint, seed 2 | learned | 15 | 22.38 | 22.45 | 0.07 | 13.94 | 13.98 | 0.04 |
| joint, seed 2 | oracle | 0 | 5.11 | 5.20 | 0.09 | 0.49 | 0.50 | 0.01 |
| joint, seed 2 | oracle | 1 | 14.33 | 14.48 | 0.15 | 10.28 | 10.38 | 0.10 |
| joint, seed 2 | oracle | 2 | 17.86 | 17.94 | 0.08 | 12.39 | 12.49 | 0.10 |
| joint, seed 2 | oracle | 3 | 19.97 | 20.07 | 0.10 | 13.36 | 13.43 | 0.07 |
| joint, seed 2 | oracle | 4 | 21.22 | 21.32 | 0.10 | 13.75 | 13.83 | 0.08 |
| joint, seed 2 | oracle | 6 | 22.09 | 22.19 | 0.10 | 13.91 | 13.98 | 0.07 |
| joint, seed 2 | oracle | 8 | 22.36 | 22.45 | 0.09 | 13.90 | 13.98 | 0.08 |
| joint, seed 2 | oracle | 15 | 22.38 | 22.46 | 0.08 | 13.91 | 13.98 | 0.07 |
| joint, seed 2 | random | 0 | 5.11 | 5.20 | 0.09 | 0.49 | 0.50 | 0.01 |
| joint, seed 2 | random | 1 | 9.81 | 9.90 | 0.09 | 4.73 | 4.76 | 0.03 |
| joint, seed 2 | random | 2 | 12.97 | 13.01 | 0.04 | 7.16 | 7.17 | 0.01 |
| joint, seed 2 | random | 3 | 15.51 | 15.58 | 0.07 | 8.82 | 8.89 | 0.07 |
| joint, seed 2 | random | 4 | 17.25 | 17.32 | 0.07 | 10.14 | 10.20 | 0.06 |
| joint, seed 2 | random | 6 | 20.28 | 20.36 | 0.08 | 12.28 | 12.33 | 0.05 |
| joint, seed 2 | random | 8 | 22.17 | 22.26 | 0.09 | 13.78 | 13.86 | 0.08 |
| joint, seed 2 | random | 15 | 22.38 | 22.46 | 0.08 | 13.92 | 13.98 | 0.06 |
| abl (i) budget input fixed | learned | 0 | 5.03 | 5.11 | 0.08 | 0.52 | 0.55 | 0.03 |
| abl (i) budget input fixed | learned | 1 | 13.93 | 14.14 | 0.21 | 9.64 | 9.87 | 0.23 |
| abl (i) budget input fixed | learned | 2 | 17.33 | 17.46 | 0.13 | 11.44 | 11.65 | 0.21 |
| abl (i) budget input fixed | learned | 3 | 19.60 | 19.76 | 0.16 | 13.01 | 13.17 | 0.16 |
| abl (i) budget input fixed | learned | 4 | 20.57 | 20.74 | 0.17 | 13.41 | 13.57 | 0.16 |
| abl (i) budget input fixed | learned | 6 | 21.92 | 22.05 | 0.13 | 13.78 | 13.92 | 0.14 |
| abl (i) budget input fixed | learned | 8 | 22.39 | 22.51 | 0.12 | 13.98 | 14.12 | 0.14 |
| abl (i) budget input fixed | learned | 15 | 22.42 | 22.54 | 0.12 | 13.98 | 14.13 | 0.15 |
| abl (ii) objectness targets | learned | 0 | 5.02 | 5.07 | 0.05 | 0.57 | 0.56 | -0.01 |
| abl (ii) objectness targets | learned | 1 | 14.03 | 14.16 | 0.13 | 9.66 | 9.85 | 0.19 |
| abl (ii) objectness targets | learned | 2 | 17.19 | 17.31 | 0.12 | 11.56 | 11.74 | 0.18 |
| abl (ii) objectness targets | learned | 3 | 19.71 | 19.80 | 0.09 | 13.17 | 13.33 | 0.16 |
| abl (ii) objectness targets | learned | 4 | 20.80 | 20.91 | 0.11 | 13.71 | 13.85 | 0.14 |
| abl (ii) objectness targets | learned | 6 | 22.16 | 22.26 | 0.10 | 14.09 | 14.20 | 0.11 |
| abl (ii) objectness targets | learned | 8 | 22.56 | 22.66 | 0.10 | 14.16 | 14.26 | 0.10 |
| abl (ii) objectness targets | learned | 15 | 22.59 | 22.69 | 0.10 | 14.17 | 14.27 | 0.10 |
| abl (iii) Gumbel top-k | learned | 0 | 5.09 | 5.23 | 0.14 | 0.59 | 0.62 | 0.03 |
| abl (iii) Gumbel top-k | learned | 1 | 14.13 | 14.29 | 0.16 | 9.93 | 10.14 | 0.21 |
| abl (iii) Gumbel top-k | learned | 2 | 17.56 | 17.69 | 0.13 | 11.65 | 11.79 | 0.14 |
| abl (iii) Gumbel top-k | learned | 3 | 19.77 | 19.86 | 0.09 | 13.11 | 13.24 | 0.13 |
| abl (iii) Gumbel top-k | learned | 4 | 20.74 | 20.82 | 0.08 | 13.49 | 13.64 | 0.15 |
| abl (iii) Gumbel top-k | learned | 6 | 22.05 | 22.15 | 0.10 | 13.92 | 14.09 | 0.17 |
| abl (iii) Gumbel top-k | learned | 8 | 22.36 | 22.46 | 0.10 | 13.98 | 14.12 | 0.14 |
| abl (iii) Gumbel top-k | learned | 15 | 22.35 | 22.46 | 0.11 | 13.99 | 14.13 | 0.14 |
| abl (iii) no selection grad | learned | 0 | 5.15 | 5.23 | 0.08 | 0.55 | 0.63 | 0.08 |
| abl (iii) no selection grad | learned | 1 | 14.11 | 14.30 | 0.19 | 10.07 | 10.28 | 0.21 |
| abl (iii) no selection grad | learned | 2 | 17.56 | 17.73 | 0.17 | 11.92 | 12.08 | 0.16 |
| abl (iii) no selection grad | learned | 3 | 19.73 | 19.86 | 0.13 | 13.20 | 13.39 | 0.19 |
| abl (iii) no selection grad | learned | 4 | 20.93 | 21.06 | 0.13 | 13.69 | 13.83 | 0.14 |
| abl (iii) no selection grad | learned | 6 | 22.13 | 22.26 | 0.13 | 14.09 | 14.22 | 0.13 |
| abl (iii) no selection grad | learned | 8 | 22.52 | 22.64 | 0.12 | 14.16 | 14.28 | 0.12 |
| abl (iii) no selection grad | learned | 15 | 22.54 | 22.67 | 0.13 | 14.17 | 14.29 | 0.12 |

## Decision on retraining

Threshold: 0.3 AP points.

- whole-image native AP (eval_ckpt, same checkpoint): change 3.21
- BCRS learned k=3 minus whole image (p3_sweep_joint_v2_fp32): change -8.11
- exhaustive tiling (k=15) minus whole image, frozen detectors: change -3.29

Retraining suggested: True (whole-image native AP (eval_ckpt, same checkpoint): +3.21; BCRS learned k=3 minus whole image (p3_sweep_joint_v2_fp32): -8.11; exhaustive tiling (k=15) minus whole image, frozen detectors: -3.29).

The evaluation sees only half of the defect. The GIoU training loss also used bf16 boxes, mostly at native resolution, so a small change here does not show that the trained weights are correct.
