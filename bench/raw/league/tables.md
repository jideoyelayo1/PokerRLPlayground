Kept checkpoint (best validation score) of each run against the suite: bb/100 [95% CI over deals], 4,000 duplicate deals per cell, seed 34000. Last column: the mean over the four unseen bots (tag, lag, potodds, bluff), per deal.

| run | kept at hands | val score | random | call | tight | tag | lag | potodds | bluff | unseen mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latest-seed1 | 60,000 | -47 | 1,001 [888, 1,113] | 48 [22, 75] | 3 [-21, 27] | -272 [-315, -230] | -273 [-334, -213] | 29 [21, 36] | 796 [690, 901] | 70 [36, 103] |
| latest-seed2 | 40,000 | -8 | 883 [773, 994] | 248 [193, 304] | -7 [-27, 13] | -208 [-243, -173] | -219 [-272, -167] | 18 [10, 26] | 761 [651, 871] | 88 [55, 121] |
| snap-seed1 | 40,000 | -45 | 870 [764, 976] | 78 [37, 118] | -10 [-32, 13] | -314 [-357, -272] | -408 [-471, -344] | 8 [-4, 20] | 699 [592, 805] | -4 [-40, 32] |
| snap-seed2 | 40,000 | -21 | 655 [561, 748] | 113 [72, 154] | 5 [-14, 24] | -276 [-312, -239] | -297 [-350, -244] | 15 [7, 24] | 607 [506, 708] | 13 [-20, 45] |
| snapbot-seed1 | 60,000 | -50 | 943 [830, 1,056] | 250 [190, 311] | -8 [-36, 21] | -175 [-213, -138] | -184 [-245, -124] | 12 [-0, 23] | 754 [647, 861] | 101 [68, 135] |
| snapbot-seed2 | 30,000 | -80 | 974 [859, 1,089] | 113 [62, 164] | -45 [-80, -10] | -382 [-432, -333] | -514 [-584, -445] | -3 [-15, 10] | 575 [472, 678] | -81 [-118, -44] |

The last checkpoint (end of training) instead of the kept one, same cells.

| run | hands | random | call | tight | tag | lag | potodds | bluff | unseen mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latest-seed1 | 60,000 | 1,001 [888, 1,113] | 48 [22, 75] | 3 [-21, 27] | -272 [-315, -230] | -273 [-334, -213] | 29 [21, 36] | 796 [690, 901] | 70 [36, 103] |
| latest-seed2 | 60,000 | 710 [620, 800] | 433 [360, 506] | -21 [-55, 13] | -245 [-287, -202] | -190 [-248, -133] | 2 [-12, 17] | 850 [742, 958] | 104 [69, 139] |
| snap-seed1 | 60,000 | 793 [689, 897] | 179 [121, 236] | -36 [-62, -11] | -320 [-362, -278] | -285 [-340, -230] | -11 [-27, 5] | 786 [681, 890] | 43 [7, 78] |
| snap-seed2 | 60,000 | 668 [570, 767] | 255 [198, 312] | -62 [-91, -33] | -261 [-299, -223] | -220 [-272, -168] | -37 [-57, -16] | 482 [390, 574] | -9 [-39, 21] |
| snapbot-seed1 | 60,000 | 943 [830, 1,056] | 250 [190, 311] | -8 [-36, 21] | -175 [-213, -138] | -184 [-245, -124] | 12 [-0, 23] | 754 [647, 861] | 101 [68, 135] |
| snapbot-seed2 | 60,000 | 840 [732, 949] | 48 [12, 85] | -51 [-91, -11] | -214 [-257, -171] | -291 [-353, -229] | -32 [-50, -13] | 535 [433, 637] | -0 [-33, 33] |

Paired difference from latest-only self-play at the same seed, kept checkpoint: mean over deals of (variant - latest) on the same deals, bb/100 [95% CI over deals]. The interval covers the deals, not the retraining: it does not include how much a run changes with its seed.

| seed | variant | random | call | tight | tag | lag | potodds | bluff | unseen mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | snap | -131 [-267, 5] | 29 [-17, 76] | -13 [-36, 11] | -42 [-73, -11] | -134 [-180, -89] | -21 [-33, -9] | -97 [-208, 14] | -74 [-108, -39] |
| 1 | snapbot | -58 [-201, 86] | 202 [138, 266] | -11 [-38, 17] | 97 [65, 129] | 89 [38, 140] | -17 [-30, -5] | -42 [-158, 74] | 32 [-5, 68] |
| 2 | snap | -229 [-353, -104] | -136 [-198, -74] | 12 [-9, 32] | -67 [-98, -36] | -78 [-124, -32] | -3 [-13, 7] | -154 [-260, -48] | -75 [-109, -42] |
| 2 | snapbot | 91 [-53, 234] | -135 [-205, -66] | -38 [-70, -6] | -174 [-212, -136] | -295 [-350, -240] | -21 [-33, -8] | -186 [-302, -71] | -169 [-208, -130] |

Paired difference from latest-only self-play at the same seed, last checkpoint: mean over deals of (variant - latest) on the same deals, bb/100 [95% CI over deals]. The interval covers the deals, not the retraining: it does not include how much a run changes with its seed.

| seed | variant | random | call | tight | tag | lag | potodds | bluff | unseen mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | snap | -208 [-341, -74] | 130 [68, 192] | -39 [-69, -10] | -48 [-85, -10] | -11 [-63, 40] | -40 [-56, -23] | -10 [-120, 101] | -27 [-64, 9] |
| 1 | snapbot | -58 [-201, 86] | 202 [138, 266] | -11 [-38, 17] | 97 [65, 129] | 89 [38, 140] | -17 [-30, -5] | -42 [-158, 74] | 32 [-5, 68] |
| 2 | snap | -42 [-158, 75] | -178 [-260, -96] | -42 [-78, -5] | -16 [-53, 21] | -30 [-80, 21] | -39 [-62, -16] | -368 [-476, -260] | -113 [-149, -77] |
| 2 | snapbot | 130 [2, 258] | -384 [-461, -308] | -30 [-65, 5] | 31 [-7, 68] | -101 [-156, -46] | -34 [-54, -13] | -315 [-433, -197] | -105 [-144, -66] |

Mean over the seeds of each run's own result (kept checkpoint), no interval: two seeds do not support one.

| variant | random | call | tight | tag | lag | potodds | bluff | unseen mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| latest | 942 | 148 | -2 | -240 | -246 | 23 | 778 | 79 |
| snap | 762 | 95 | -2 | -295 | -352 | 12 | 653 | 4 |
| snapbot | 959 | 182 | -26 | -279 | -349 | 4 | 664 | 10 |

Cross-play, kept checkpoints: the run on the left in one seat against three copies of the run on the top, bb/100 [95% CI over deals], 4,000 duplicate deals, seed 34000. Positive: the row run wins.

| run \ copies | latest-seed1 | latest-seed2 | snap-seed1 | snap-seed2 | snapbot-seed1 | snapbot-seed2 |
|---|---:|---:|---:|---:|---:|---:|
| latest-seed1 | - | -154 [-237, -71] | 240 [154, 327] | 162 [75, 250] | 237 [141, 332] | 375 [276, 473] |
| latest-seed2 | 88 [9, 166] | - | 304 [209, 399] | 267 [186, 347] | 404 [307, 501] | 455 [357, 553] |
| snap-seed1 | -204 [-285, -123] | -234 [-324, -145] | - | 5 [-89, 99] | 124 [27, 222] | 184 [85, 283] |
| snap-seed2 | -201 [-285, -117] | -306 [-388, -224] | -0 [-97, 96] | - | 65 [-33, 162] | 150 [48, 251] |
| snapbot-seed1 | -165 [-252, -77] | -265 [-361, -168] | -161 [-257, -65] | -44 [-141, 54] | - | 31 [-75, 136] |
| snapbot-seed2 | -355 [-448, -263] | -486 [-586, -386] | -209 [-309, -109] | -206 [-307, -104] | -125 [-227, -22] | - |

Cross-play by variant: mean bb/100 of the row variant's runs against the column variant's runs (every pairing of different runs; for the same variant, the two runs of different seeds), with the lowest and highest pairing.

| variant \ copies | latest | snap | snapbot |
|---|---:|---:|---:|
| latest | -33 (-154 to 88) | 243 (162 to 304) | 368 (237 to 455) |
| snap | -236 (-306 to -201) | 2 (-0 to 5) | 131 (65 to 184) |
| snapbot | -318 (-486 to -165) | -155 (-209 to -44) | -47 (-125 to 31) |
