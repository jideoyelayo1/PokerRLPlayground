Evaluation: 40,000 duplicate deals (160,000 hands) per cell, deals from seed 7000, 4 seats, stacks 200 (100 big blinds), blinds 1/2. bb/100 [95% CI over deals]. Commit 1b0446f.

| run | parameters | kept checkpoint (hands) | vs random | vs call | vs tight | score (worst lower bound) | rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp-s-seed3 | 36,871 | 320,000 | 359 [332, 386] | 341 [323, 359] | 16 [11, 21] | 11 | 1 |
| gru-seed3 | 108,807 | 150,000 | 809 [778, 841] | 293 [272, 314] | 3 [-5, 11] | -5 | 2 |
| mlp-s-seed2 | 36,871 | 280,000 | 338 [311, 365] | 203 [189, 218] | -2 [-9, 5] | -9 | 3 |
| mlp-s-seed1 | 36,871 | 400,000 | 378 [352, 404] | 99 [87, 111] | -14 [-19, -9] | -19 | 4 |
| gru-seed1 | 108,807 | 75,000 | 958 [924, 992] | 87 [70, 104] | -25 [-34, -17] | -34 | 5 |
| mlp-l-seed3 | 473,607 | 120,000 | 654 [620, 688] | 125 [109, 140] | -39 [-48, -30] | -48 | 6 |
| mlp-l-seed2 | 473,607 | 200,000 | 351 [325, 378] | -30 [-39, -21] | -42 [-50, -34] | -50 | 7 |
| mlp-l-seed1 | 473,607 | 120,000 | 497 [469, 525] | 223 [202, 244] | -55 [-64, -46] | -64 | 8 |
| gru-seed2 | 108,807 | 90,000 | 916 [880, 951] | 259 [236, 281] | -58 [-68, -48] | -68 | 9 |
| random-bot (control) | | | -0 [-40, 40] | -128 [-167, -88] | -669 [-692, -646] | | |

Across seeds, per architecture: mean of the per-seed bb/100 [95% t-interval over seeds].

| candidate | seeds | vs random | vs call | vs tight |
|---|---:|---:|---:|---:|
| mlp-s | 3 | 358 [309, 408] | 214 [-88, 516] | 0 [-37, 37] |
| mlp-l | 3 | 501 [125, 876] | 106 [-211, 422] | -45 [-66, -25] |
| gru | 3 | 894 [704, 1,084] | 213 [-61, 487] | -27 [-103, 50] |

Training (bench/train_demo.py; wall time with the other runs in parallel; decisions per hand of self-play over the last tenth of the run).

| run | hands | wall | CPU | hands/s | decisions/hand | kept checkpoint | its validation score |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru-seed1 | 150,000 | 287 min | 2.56 h | 9 | 27.7 | 75,000 | -68 |
| gru-seed2 | 150,000 | 262 min | 2.15 h | 10 | 16.1 | 90,000 | -77 |
| gru-seed3 | 150,000 | 283 min | 2.50 h | 9 | 20.2 | 150,000 | -74 |
| mlp-l-seed1 | 400,000 | 177 min | 1.31 h | 38 | 14.6 | 120,000 | -88 |
| mlp-l-seed2 | 400,000 | 179 min | 1.33 h | 37 | 13.6 | 200,000 | -93 |
| mlp-l-seed3 | 400,000 | 175 min | 1.31 h | 38 | 13.2 | 120,000 | -106 |
| mlp-s-seed1 | 400,000 | 96 min | 0.58 h | 69 | 18.1 | 400,000 | -38 |
| mlp-s-seed2 | 400,000 | 97 min | 0.59 h | 68 | 14.0 | 280,000 | -33 |
| mlp-s-seed3 | 400,000 | 99 min | 0.60 h | 67 | 16.4 | 320,000 | 4 |

Seats: mlp-s-seed3 in all three (clearly best).
Beats each baseline (lower end of the 95% interval above 0): mlp-s-seed3: random yes, call yes, tight yes.
