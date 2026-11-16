| agent | vs | seed 1: bb/100 [95% CI] | seed 2: bb/100 [95% CI] | seed 3: bb/100 [95% CI] | mean over seeds [95% t-CI] |
|---|---|---|---|---|---|
| MLP | random | -1,853 [-14,902, 11,196] | -11,170 [-22,858, 518] | 351 [-11,466, 12,169] | -4,224 [-19,417, 10,969] |
| MLP | call | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| MLP | tight | -2,013 [-3,601, -425] | -69,495 [-81,534, -57,456] | -66,685 [-79,132, -54,238] | -46,064 [-140,905, 48,776] |
| LSTM | random | 9,928 [-2,645, 22,502] | 19,527 [7,287, 31,768] | 8,628 [-2,897, 20,154] | 12,695 [-2,094, 27,483] |
| LSTM | call | -2 [-5, 1] | 0 [0, 0] | 14 [-27, 55] | 4 [-17, 25] |
| LSTM | tight | -21,866 [-28,069, -15,664] | -17,086 [-23,444, -10,728] | -14,668 [-20,704, -8,631] | -17,873 [-26,974, -8,772] |
| GRU | random | 8,540 [-3,160, 20,241] | 25,510 [12,632, 38,389] | 8,878 [-1,101, 18,858] | 14,310 [-9,792, 38,412] |
| GRU | call | 0 [0, 0] | -25 [-66, 16] | -38 [-38, -38] | -21 [-68, 27] |
| GRU | tight | -69,994 [-81,984, -58,004] | -17,211 [-23,567, -10,856] | -38 [-38, -38] | -29,081 [-119,653, 61,491] |
| Transformer | random | -8,576 [-16,209, -942] | -9,236 [-21,223, 2,752] | 17,811 [4,345, 31,277] | -0 [-38,329, 38,329] |
| Transformer | call | 0 [0, 0] | 0 [0, 0] | 14 [-27, 55] | 5 [-15, 24] |
| Transformer | tight | -11,482 [-16,359, -6,604] | -69,495 [-81,534, -57,456] | -11,408 [-17,998, -4,819] | -30,795 [-114,058, 52,468] |
| random-bot | random | -792 [-13,326, 11,742] | 10,861 [-1,926, 23,649] | -618 [-13,143, 11,908] | 3,150 [-13,441, 19,741] |
| random-bot | call | -1,096 [-13,584, 11,391] | -1,847 [-14,618, 10,923] | 11,577 [-1,199, 24,353] | 2,878 [-15,862, 21,618] |
| random-bot | tight | -34,148 [-41,180, -27,115] | -29,638 [-36,785, -22,491] | -30,717 [-38,068, -23,367] | -31,501 [-37,351, -25,651] |

seed 1: trained 2000 hands in 9.5 min wall (0.34 CPU-hours, 21 tournament resets, 0 aborted hands); evaluation 3.7 min, 1000 deals x 4 seats per cell

seed 2: trained 2000 hands in 11.7 min wall (0.46 CPU-hours, 1263 tournament resets, 8 aborted hands); evaluation 4.2 min, 1000 deals x 4 seats per cell

seed 3: trained 2000 hands in 10.0 min wall (0.38 CPU-hours, 190 tournament resets, 3 aborted hands); evaluation 3.3 min, 1000 deals x 4 seats per cell

| seed | seat (model) | first actions in the last 25% of training hands |
|---|---|---|
| 1 | all | 1959 of 4060 logged rows are folds (48.3%); the old logging dropped every one of them |
| 1 | 0 (MLP) | not dealt in (busted) (n=0) |
| 1 | 1 (LSTM) | allin 100% (n=500) |
| 1 | 2 (GRU) | not dealt in (busted) (n=0) |
| 1 | 3 (Transformer) | fold 50%, raise_bb 36%, raise_pot 14% (n=500) |
| 2 | all | 567 of 6591 logged rows are folds (8.6%); the old logging dropped every one of them |
| 2 | 0 (MLP) | allin 61%, call 28%, fold 6%, raise_pot 3%, raise_bb 1%, raise_half 1%, check 0% (n=487) |
| 2 | 1 (LSTM) | allin 95%, raise_bb 4%, call 1%, raise_pot 0%, unknown 0%, raise_half 0% (n=477) |
| 2 | 2 (GRU) | raise_bb 50%, allin 47%, call 3% (n=443) |
| 2 | 3 (Transformer) | allin 76%, call 24%, fold 0% (n=446) |
| 3 | all | 1734 of 4439 logged rows are folds (39.1%); the old logging dropped every one of them |
| 3 | 0 (MLP) | allin 98%, raise_pot 1%, raise_bb 0%, fold 0%, raise_half 0%, call 0% (n=486) |
| 3 | 1 (LSTM) | allin 53%, fold 18%, unknown 12%, raise_pot 9%, call 6%, raise_bb 3% (n=34) |
| 3 | 2 (GRU) | raise_pot 39%, allin 22%, fold 22%, call 6%, check 6%, raise_bb 6% (n=18) |
| 3 | 3 (Transformer) | fold 49%, raise_bb 48%, unknown 1%, raise_half 1%, allin 1%, raise_pot 1%, call 0% (n=498) |
