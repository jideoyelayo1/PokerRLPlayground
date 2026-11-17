| agent | vs | seed 1: bb/100 [95% CI] | seed 2: bb/100 [95% CI] | seed 3: bb/100 [95% CI] | mean over seeds [95% t-CI] |
|---|---|---|---|---|---|
| MLP | random | -8,698 [-21,762, 4,367] | 12,598 [-814, 26,010] | -12,168 [-23,647, -689] | -2,756 [-36,070, 30,558] |
| MLP | call | -59 [-794, 675] | 32 [-36, 101] | 0 [0, 0] | -9 [-125, 107] |
| MLP | tight | -9,838 [-14,039, -5,636] | -7,153 [-11,272, -3,034] | -66,684 [-79,131, -54,238] | -27,892 [-111,421, 55,637] |
| LSTM | random | 20,761 [7,858, 33,665] | 165 [-8,370, 8,700] | 5,081 [-7,443, 17,604] | 8,669 [-18,055, 35,393] |
| LSTM | call | 0 [0, 0] | -38 [-38, -38] | -4 [-72, 63] | -14 [-65, 37] |
| LSTM | tight | -15,990 [-22,629, -9,351] | -18 [-19, -17] | -13,035 [-19,026, -7,045] | -9,681 [-30,793, 11,430] |
| GRU | random | 2,440 [-9,594, 14,474] | 20,423 [7,160, 33,686] | 29,995 [16,863, 43,127] | 17,619 [-17,136, 52,374] |
| GRU | call | 0 [0, 0] | 144 [-597, 885] | -4 [-72, 63] | 46 [-163, 256] |
| GRU | tight | -69,993 [-81,983, -58,003] | -19,387 [-25,774, -13,000] | 2,389 [-4,823, 9,602] | -28,997 [-121,255, 63,261] |
| Transformer | random | 24,115 [11,408, 36,821] | 10,423 [-1,940, 22,787] | 29,678 [16,363, 42,993] | 21,405 [-3,212, 46,023] |
| Transformer | call | 0 [0, 0] | 32 [-36, 101] | -4 [-72, 63] | 9 [-40, 59] |
| Transformer | tight | -2,357 [-9,856, 5,142] | -6,184 [-11,068, -1,300] | -1,500 [-8,206, 5,207] | -3,347 [-9,543, 2,849] |
| random-bot | random | 5,326 [-7,314, 17,966] | 568 [-11,874, 13,010] | -7,017 [-19,262, 5,228] | -374 [-15,841, 15,092] |
| random-bot | call | -1,088 [-13,573, 11,397] | -1,876 [-14,648, 10,895] | 11,556 [-1,219, 24,332] | 2,864 [-15,863, 21,591] |
| random-bot | tight | -33,631 [-40,572, -26,689] | -28,819 [-36,034, -21,603] | -31,363 [-38,636, -24,090] | -31,271 [-37,252, -25,290] |

seed 1: trained 2000 hands in 12.0 min wall (0.48 CPU-hours, 324 tournament resets, 15 aborted hands); evaluation 3.4 min, 1000 deals x 4 seats per cell

seed 2: trained 2000 hands in 10.8 min wall (0.44 CPU-hours, 462 tournament resets, 2 aborted hands); evaluation 3.1 min, 1000 deals x 4 seats per cell

seed 3: trained 2000 hands in 15.6 min wall (0.68 CPU-hours, 792 tournament resets, 41 aborted hands); evaluation 4.4 min, 1000 deals x 4 seats per cell

| seed | seat (model) | first actions in the last 25% of training hands |
|---|---|---|
| 1 | all | 1528 of 4680 logged rows are folds (32.6%); the old logging dropped every one of them |
| 1 | 0 (MLP) | allin 100%, raise_pot 0% (n=484) |
| 1 | 1 (LSTM) | call 75%, raise_pot 19%, raise_bb 3%, allin 1%, fold 1% (n=67) |
| 1 | 2 (GRU) | allin 91%, raise_bb 3%, fold 2%, raise_pot 2%, raise_half 2% (n=58) |
| 1 | 3 (Transformer) | raise_bb 54%, fold 43%, raise_half 1%, call 1%, raise_pot 0%, allin 0% (n=493) |
| 2 | all | 1366 of 4941 logged rows are folds (27.6%); the old logging dropped every one of them |
| 2 | 0 (MLP) | not dealt in (busted) (n=0) |
| 2 | 1 (LSTM) | not dealt in (busted) (n=0) |
| 2 | 2 (GRU) | allin 100% (n=500) |
| 2 | 3 (Transformer) | fold 50%, raise_bb 45%, raise_pot 5% (n=500) |
| 3 | all | 519 of 5586 logged rows are folds (9.3%); the old logging dropped every one of them |
| 3 | 0 (MLP) | allin 98%, call 2%, check 0% (n=488) |
| 3 | 1 (LSTM) | allin 97%, raise_bb 2%, raise_half 0%, call 0% (n=484) |
| 3 | 2 (GRU) | call 95%, raise_bb 2%, fold 2%, raise_pot 1%, allin 0%, raise_half 0% (n=488) |
| 3 | 3 (Transformer) | raise_bb 80%, call 16%, allin 2%, raise_pot 1%, raise_half 0% (n=480) |
