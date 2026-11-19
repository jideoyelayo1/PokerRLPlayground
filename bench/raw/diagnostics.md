Diagnostics for the agent's seat in every duplicate-deal cell of bench/eval_suite.py (definitions: `Stats` there; thresholds: `FLAGS`). Hands = 4 x deals. 3-bet: share of its preflop decisions facing one raise; AF = (raises + all-ins) / calls; free folds = folds with check legal (count, share of decisions); all-in = share of decisions.

| part | agent | opponent | stack bb | hands | bb/100 | VPIP | PFR | 3-bet | fold to bet | showdown | AF | free folds | all-in | illegal | flags |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cross | gru-seed1 | mlp-s-seed3 | 100 | 40,000 | -298 | 63% | 48% | 45% | 5% | 41% | 17.44 | 3,535 (0.9%) | 2% | 0 | free_folds |
| cross | gru-seed2 | mlp-s-seed3 | 100 | 40,000 | -225 | 58% | 55% | 56% | 8% | 35% | 14.19 | 3,632 (1.3%) | 2% | 0 | free_folds |
| cross | gru-seed3 | mlp-s-seed3 | 100 | 40,000 | -129 | 56% | 50% | 50% | 9% | 32% | 13.36 | 4,310 (1.6%) | 2% | 0 | free_folds |
| cross | mlp-l-seed1 | mlp-s-seed3 | 100 | 40,000 | -610 | 87% | 49% | 34% | 2% | 53% | 10.19 | 5,185 (1.0%) | 1% | 0 | free_folds |
| cross | mlp-l-seed2 | mlp-s-seed3 | 100 | 40,000 | -478 | 84% | 46% | 34% | 3% | 42% | 7.71 | 7,262 (1.7%) | 1% | 0 | free_folds |
| cross | mlp-l-seed3 | mlp-s-seed3 | 100 | 40,000 | -468 | 76% | 49% | 37% | 3% | 44% | 11.22 | 6,466 (1.5%) | 1% | 0 | free_folds |
| cross | mlp-s-seed1 | mlp-s-seed3 | 100 | 40,000 | -41 | 80% | 41% | 29% | 10% | 32% | 3.26 | 6,854 (2.9%) | 1% | 0 | free_folds |
| cross | mlp-s-seed2 | mlp-s-seed3 | 100 | 40,000 | -31 | 69% | 31% | 23% | 9% | 37% | 3.17 | 4,694 (1.9%) | 1% | 0 | free_folds |
| formats | gru-seed3 | call | 50 | 16,000 | 273 | 61% | 59% | - | 37% | 35% | 86.40 | 5,954 (15.5%) | 6% | 0 | free_folds |
| formats | gru-seed3 | call | 100 | 16,000 | 290 | 56% | 52% | - | 43% | 26% | 39.79 | 6,596 (18.3%) | 2% | 0 | free_folds |
| formats | gru-seed3 | call | 200 | 16,000 | 66 | 33% | 31% | - | 68% | 15% | 42.22 | 5,391 (19.7%) | 0% | 0 | free_folds |
| formats | gru-seed3 | call | 400 | 16,000 | -35 | 0% | 0% | - | 100% | 0% | nan | 3,988 (24.8%) | 0% | 0 | free_folds, vpip_low, fold_to_bet |
| formats | gru-seed3 | random | 50 | 16,000 | 514 | 55% | 44% | 40% | 37% | 32% | 2.02 | 170 (0.6%) | 7% | 0 | free_folds |
| formats | gru-seed3 | random | 100 | 16,000 | 782 | 46% | 36% | 33% | 45% | 21% | 1.90 | 226 (0.8%) | 5% | 0 | free_folds |
| formats | gru-seed3 | random | 200 | 16,000 | 676 | 22% | 18% | 17% | 67% | 8% | 1.84 | 229 (1.0%) | 3% | 0 | free_folds |
| formats | gru-seed3 | random | 400 | 16,000 | 76 | 0% | 0% | 0% | 100% | 0% | 0.48 | 187 (1.2%) | 0% | 0 | free_folds, vpip_low, fold_to_bet |
| formats | gru-seed3 | tight | 50 | 16,000 | -11 | 58% | 46% | 50% | 27% | 18% | 6.54 | 1,811 (6.3%) | 2% | 0 | free_folds |
| formats | gru-seed3 | tight | 100 | 16,000 | -13 | 47% | 37% | 42% | 41% | 12% | 6.04 | 2,037 (8.1%) | 1% | 0 | free_folds |
| formats | gru-seed3 | tight | 200 | 16,000 | 4 | 15% | 13% | 14% | 80% | 3% | 7.45 | 2,017 (11.8%) | 0% | 0 | free_folds |
| formats | gru-seed3 | tight | 400 | 16,000 | -19 | 0% | 0% | 0% | 100% | 0% | nan | 1,504 (10.7%) | 0% | 0 | free_folds, vpip_low, fold_to_bet |
| formats | mlp-s-seed2 | call | 50 | 16,000 | 106 | 62% | 43% | - | 28% | 30% | 5.80 | 7,930 (20.0%) | 1% | 0 | free_folds |
| formats | mlp-s-seed2 | call | 100 | 16,000 | 191 | 60% | 33% | - | 29% | 35% | 3.75 | 6,904 (16.5%) | 1% | 0 | free_folds |
| formats | mlp-s-seed2 | call | 200 | 16,000 | 95 | 88% | 87% | - | 11% | 60% | 415.48 | 5,072 (9.5%) | 0% | 0 | free_folds |
| formats | mlp-s-seed2 | call | 400 | 16,000 | 98 | 100% | 100% | - | 0% | 95% | nan | 862 (1.4%) | 0% | 0 | free_folds, vpip_high |
| formats | mlp-s-seed2 | random | 50 | 16,000 | 294 | 63% | 35% | 27% | 33% | 34% | 0.87 | 414 (1.3%) | 8% | 0 | free_folds |
| formats | mlp-s-seed2 | random | 100 | 16,000 | 375 | 58% | 23% | 19% | 43% | 17% | 0.55 | 268 (0.8%) | 3% | 0 | free_folds |
| formats | mlp-s-seed2 | random | 200 | 16,000 | -220 | 67% | 63% | 55% | 46% | 7% | 2.76 | 88 (0.3%) | 1% | 0 | free_folds |
| formats | mlp-s-seed2 | random | 400 | 16,000 | -1,149 | 76% | 75% | 63% | 42% | 4% | 10.93 | 13 (0.0%) | 1% | 0 | free_folds |
| formats | mlp-s-seed2 | tight | 50 | 16,000 | -13 | 63% | 35% | 16% | 22% | 26% | 2.26 | 2,562 (7.9%) | 2% | 0 | free_folds |
| formats | mlp-s-seed2 | tight | 100 | 16,000 | -13 | 65% | 24% | 14% | 20% | 34% | 1.54 | 1,978 (5.4%) | 1% | 0 | free_folds |
| formats | mlp-s-seed2 | tight | 200 | 16,000 | -79 | 75% | 69% | 85% | 16% | 21% | 9.37 | 1,107 (3.6%) | 0% | 0 | free_folds |
| formats | mlp-s-seed2 | tight | 400 | 16,000 | -626 | 88% | 88% | 100% | 7% | 23% | 33.49 | 262 (0.8%) | 0% | 0 | free_folds |
| formats | mlp-s-seed3 | call | 50 | 16,000 | 165 | 58% | 25% | - | 30% | 32% | 1.86 | 7,210 (18.4%) | 2% | 0 | free_folds |
| formats | mlp-s-seed3 | call | 100 | 16,000 | 326 | 62% | 31% | - | 27% | 46% | 2.78 | 5,419 (11.8%) | 1% | 0 | free_folds |
| formats | mlp-s-seed3 | call | 200 | 16,000 | 305 | 51% | 28% | - | 38% | 58% | 3.96 | 2,100 (4.4%) | 1% | 0 | free_folds |
| formats | mlp-s-seed3 | call | 400 | 16,000 | 314 | 33% | 17% | - | 61% | 34% | 3.81 | 3,300 (9.4%) | 1% | 0 | free_folds |
| formats | mlp-s-seed3 | random | 50 | 16,000 | 244 | 60% | 23% | 15% | 40% | 24% | 0.53 | 432 (1.4%) | 7% | 0 | free_folds |
| formats | mlp-s-seed3 | random | 100 | 16,000 | 437 | 60% | 24% | 16% | 41% | 17% | 0.54 | 312 (0.9%) | 4% | 0 | free_folds |
| formats | mlp-s-seed3 | random | 200 | 16,000 | 534 | 51% | 23% | 17% | 47% | 13% | 0.67 | 98 (0.3%) | 5% | 0 | free_folds |
| formats | mlp-s-seed3 | random | 400 | 16,000 | 688 | 39% | 22% | 21% | 53% | 18% | 0.87 | 185 (0.7%) | 10% | 0 | free_folds |
| formats | mlp-s-seed3 | tight | 50 | 16,000 | -1 | 65% | 23% | 6% | 20% | 32% | 1.22 | 2,943 (8.2%) | 1% | 0 | free_folds |
| formats | mlp-s-seed3 | tight | 100 | 16,000 | 24 | 68% | 30% | 6% | 17% | 32% | 1.78 | 2,015 (5.6%) | 0% | 0 | free_folds |
| formats | mlp-s-seed3 | tight | 200 | 16,000 | 17 | 59% | 26% | 5% | 27% | 30% | 2.21 | 941 (2.8%) | 0% | 0 | free_folds |
| formats | mlp-s-seed3 | tight | 400 | 16,000 | -31 | 42% | 14% | 5% | 47% | 18% | 1.79 | 1,776 (6.3%) | 0% | 0 | free_folds |
| formats | random-bot | call | 50 | 16,000 | -162 | 78% | 64% | - | 18% | 59% | 11.27 | 4,449 (11.8%) | 18% | 0 | free_folds |
| formats | random-bot | call | 100 | 16,000 | -188 | 78% | 65% | - | 18% | 59% | 11.47 | 4,372 (11.5%) | 17% | 0 | free_folds |
| formats | random-bot | call | 200 | 16,000 | -441 | 78% | 65% | - | 18% | 60% | 11.55 | 4,336 (11.4%) | 17% | 0 | free_folds |
| formats | random-bot | call | 400 | 16,000 | -209 | 78% | 65% | - | 18% | 60% | 11.49 | 4,311 (11.4%) | 17% | 0 | free_folds |
| formats | random-bot | random | 50 | 16,000 | 9 | 77% | 60% | 52% | 25% | 55% | 2.19 | 235 (0.9%) | 25% | 0 | free_folds |
| formats | random-bot | random | 100 | 16,000 | 45 | 77% | 61% | 52% | 25% | 55% | 2.26 | 225 (0.9%) | 24% | 0 | free_folds |
| formats | random-bot | random | 200 | 16,000 | -19 | 77% | 60% | 52% | 25% | 54% | 2.30 | 192 (0.7%) | 24% | 0 | free_folds |
| formats | random-bot | random | 400 | 16,000 | 169 | 77% | 60% | 52% | 24% | 55% | 2.25 | 193 (0.7%) | 24% | 0 | free_folds |
| formats | random-bot | tight | 50 | 16,000 | -340 | 70% | 55% | 58% | 19% | 24% | 5.65 | 1,867 (7.6%) | 19% | 0 | free_folds |
| formats | random-bot | tight | 100 | 16,000 | -661 | 69% | 54% | 60% | 20% | 24% | 5.54 | 1,793 (7.2%) | 19% | 0 | free_folds |
| formats | random-bot | tight | 200 | 16,000 | -1,302 | 69% | 54% | 61% | 20% | 24% | 5.66 | 1,750 (7.0%) | 19% | 0 | free_folds |
| formats | random-bot | tight | 400 | 16,000 | -2,763 | 69% | 54% | 59% | 20% | 24% | 5.56 | 1,744 (7.0%) | 19% | 0 | free_folds |
| suite | gru-seed1 | bluff | 100 | 40,000 | 687 | 62% | 48% | 47% | 21% | 37% | 2.51 | 4,683 (3.5%) | 5% | 0 | free_folds |
| suite | gru-seed1 | call | 100 | 40,000 | 109 | 62% | 49% | - | 38% | 35% | 13.42 | 14,505 (14.3%) | 2% | 0 | free_folds |
| suite | gru-seed1 | lag | 100 | 40,000 | -332 | 58% | 42% | 52% | 21% | 22% | 1.85 | 2,186 (2.1%) | 3% | 0 | free_folds |
| suite | gru-seed1 | potodds | 100 | 40,000 | -4 | 54% | 31% | - | 35% | 13% | 3.91 | 5,069 (8.1%) | 0% | 0 | free_folds |
| suite | gru-seed1 | random | 100 | 40,000 | 986 | 53% | 37% | 34% | 38% | 27% | 1.67 | 369 (0.5%) | 10% | 0 | free_folds |
| suite | gru-seed1 | tag | 100 | 40,000 | -294 | 54% | 29% | 31% | 28% | 12% | 2.17 | 1,838 (2.8%) | 2% | 0 | free_folds |
| suite | gru-seed1 | tight | 100 | 40,000 | -25 | 55% | 33% | 31% | 32% | 17% | 3.92 | 4,476 (6.2%) | 1% | 0 | free_folds |
| suite | gru-seed2 | bluff | 100 | 40,000 | 662 | 61% | 56% | 58% | 26% | 34% | 2.62 | 4,330 (3.6%) | 5% | 0 | free_folds |
| suite | gru-seed2 | call | 100 | 40,000 | 309 | 61% | 58% | - | 39% | 34% | 48.59 | 14,567 (15.3%) | 3% | 0 | free_folds |
| suite | gru-seed2 | lag | 100 | 40,000 | -190 | 50% | 47% | 66% | 29% | 17% | 3.18 | 2,280 (2.6%) | 3% | 0 | free_folds |
| suite | gru-seed2 | potodds | 100 | 40,000 | -30 | 43% | 38% | - | 50% | 10% | 16.26 | 3,539 (6.5%) | 2% | 0 | free_folds |
| suite | gru-seed2 | random | 100 | 40,000 | 907 | 52% | 38% | 33% | 40% | 28% | 1.40 | 304 (0.4%) | 3% | 0 | free_folds |
| suite | gru-seed2 | tag | 100 | 40,000 | -166 | 41% | 36% | 41% | 42% | 8% | 4.58 | 725 (1.4%) | 2% | 0 | free_folds |
| suite | gru-seed2 | tight | 100 | 40,000 | -56 | 45% | 39% | 40% | 44% | 12% | 8.94 | 3,603 (5.9%) | 2% | 0 | free_folds |
| suite | gru-seed3 | bluff | 100 | 40,000 | 822 | 56% | 52% | 54% | 28% | 27% | 3.53 | 5,021 (4.3%) | 4% | 0 | free_folds |
| suite | gru-seed3 | call | 100 | 40,000 | 326 | 56% | 53% | - | 43% | 26% | 39.34 | 16,681 (18.5%) | 2% | 0 | free_folds |
| suite | gru-seed3 | lag | 100 | 40,000 | -139 | 50% | 43% | 49% | 29% | 16% | 2.85 | 3,017 (3.4%) | 2% | 0 | free_folds |
| suite | gru-seed3 | potodds | 100 | 40,000 | 3 | 46% | 34% | - | 45% | 9% | 6.16 | 5,449 (10.1%) | 0% | 0 | free_folds |
| suite | gru-seed3 | random | 100 | 40,000 | 823 | 47% | 36% | 33% | 45% | 22% | 1.82 | 468 (0.7%) | 5% | 0 | free_folds |
| suite | gru-seed3 | tag | 100 | 40,000 | -146 | 45% | 34% | 42% | 38% | 8% | 3.19 | 1,953 (3.5%) | 1% | 0 | free_folds |
| suite | gru-seed3 | tight | 100 | 40,000 | -3 | 47% | 37% | 41% | 40% | 12% | 5.81 | 5,084 (8.1%) | 0% | 0 | free_folds |
| suite | mlp-l-seed1 | bluff | 100 | 40,000 | 549 | 86% | 44% | 30% | 18% | 31% | 1.22 | 8,493 (5.6%) | 3% | 0 | free_folds |
| suite | mlp-l-seed1 | call | 100 | 40,000 | 249 | 84% | 52% | - | 11% | 47% | 5.91 | 17,807 (15.1%) | 2% | 0 | free_folds |
| suite | mlp-l-seed1 | lag | 100 | 40,000 | -470 | 84% | 55% | 39% | 10% | 31% | 1.25 | 2,059 (1.6%) | 2% | 0 | free_folds |
| suite | mlp-l-seed1 | potodds | 100 | 40,000 | -29 | 76% | 47% | - | 9% | 22% | 4.66 | 4,464 (5.9%) | 1% | 0 | free_folds |
| suite | mlp-l-seed1 | random | 100 | 40,000 | 476 | 68% | 34% | 24% | 35% | 20% | 0.93 | 575 (0.6%) | 5% | 0 | free_folds |
| suite | mlp-l-seed1 | tag | 100 | 40,000 | -359 | 75% | 41% | 28% | 9% | 17% | 1.85 | 1,751 (2.2%) | 2% | 0 | free_folds |
| suite | mlp-l-seed1 | tight | 100 | 40,000 | -62 | 79% | 47% | 25% | 9% | 26% | 3.78 | 3,914 (4.6%) | 1% | 0 | free_folds |
| suite | mlp-l-seed2 | bluff | 100 | 40,000 | 329 | 83% | 44% | 32% | 18% | 27% | 1.33 | 9,576 (6.2%) | 1% | 0 | free_folds |
| suite | mlp-l-seed2 | call | 100 | 40,000 | -30 | 81% | 50% | - | 14% | 36% | 5.31 | 21,498 (18.9%) | 0% | 0 | free_folds |
| suite | mlp-l-seed2 | lag | 100 | 40,000 | -337 | 81% | 52% | 42% | 13% | 26% | 1.10 | 3,255 (2.6%) | 2% | 0 | free_folds |
| suite | mlp-l-seed2 | potodds | 100 | 40,000 | -11 | 73% | 44% | - | 13% | 17% | 4.02 | 7,508 (10.7%) | 0% | 0 | free_folds |
| suite | mlp-l-seed2 | random | 100 | 40,000 | 300 | 66% | 30% | 23% | 38% | 16% | 0.81 | 966 (1.1%) | 4% | 0 | free_folds |
| suite | mlp-l-seed2 | tag | 100 | 40,000 | -267 | 72% | 38% | 26% | 13% | 13% | 1.64 | 3,466 (4.7%) | 1% | 0 | free_folds |
| suite | mlp-l-seed2 | tight | 100 | 40,000 | -33 | 76% | 44% | 25% | 12% | 19% | 3.30 | 6,894 (8.6%) | 1% | 0 | free_folds |
| suite | mlp-l-seed3 | bluff | 100 | 40,000 | 664 | 71% | 43% | 33% | 16% | 37% | 1.78 | 8,134 (5.4%) | 3% | 0 | free_folds |
| suite | mlp-l-seed3 | call | 100 | 40,000 | 123 | 64% | 43% | - | 22% | 39% | 7.64 | 17,716 (16.1%) | 1% | 0 | free_folds |
| suite | mlp-l-seed3 | lag | 100 | 40,000 | -392 | 74% | 55% | 43% | 14% | 24% | 1.74 | 2,927 (2.5%) | 2% | 0 | free_folds |
| suite | mlp-l-seed3 | potodds | 100 | 40,000 | -4 | 70% | 51% | - | 13% | 16% | 6.09 | 6,419 (9.3%) | 1% | 0 | free_folds |
| suite | mlp-l-seed3 | random | 100 | 40,000 | 652 | 64% | 40% | 28% | 33% | 29% | 1.18 | 824 (1.0%) | 7% | 0 | free_folds |
| suite | mlp-l-seed3 | tag | 100 | 40,000 | -295 | 71% | 46% | 22% | 13% | 13% | 2.49 | 2,743 (3.9%) | 2% | 0 | free_folds |
| suite | mlp-l-seed3 | tight | 100 | 40,000 | -41 | 71% | 49% | 20% | 14% | 20% | 4.42 | 5,947 (7.7%) | 1% | 0 | free_folds |
| suite | mlp-s-seed1 | bluff | 100 | 40,000 | 538 | 75% | 39% | 29% | 27% | 17% | 0.88 | 10,278 (8.7%) | 2% | 0 | free_folds |
| suite | mlp-s-seed1 | call | 100 | 40,000 | 95 | 72% | 42% | - | 20% | 27% | 3.39 | 23,026 (22.0%) | 0% | 0 | free_folds |
| suite | mlp-s-seed1 | lag | 100 | 40,000 | -148 | 78% | 42% | 30% | 21% | 18% | 0.65 | 2,560 (2.1%) | 1% | 0 | free_folds |
| suite | mlp-s-seed1 | potodds | 100 | 40,000 | 2 | 72% | 36% | - | 12% | 24% | 2.15 | 9,078 (10.9%) | 0% | 0 | free_folds |
| suite | mlp-s-seed1 | random | 100 | 40,000 | 400 | 62% | 28% | 19% | 43% | 16% | 0.66 | 898 (1.1%) | 4% | 0 | free_folds |
| suite | mlp-s-seed1 | tag | 100 | 40,000 | -129 | 72% | 29% | 12% | 17% | 17% | 0.89 | 4,871 (5.9%) | 1% | 0 | free_folds |
| suite | mlp-s-seed1 | tight | 100 | 40,000 | -18 | 74% | 35% | 11% | 13% | 27% | 1.83 | 8,065 (9.0%) | 0% | 0 | free_folds |
| suite | mlp-s-seed2 | bluff | 100 | 40,000 | 296 | 65% | 31% | 24% | 29% | 16% | 0.85 | 8,882 (7.5%) | 1% | 0 | free_folds |
| suite | mlp-s-seed2 | call | 100 | 40,000 | 198 | 60% | 33% | - | 29% | 35% | 3.71 | 17,328 (16.5%) | 1% | 0 | free_folds |
| suite | mlp-s-seed2 | lag | 100 | 40,000 | -144 | 67% | 29% | 18% | 20% | 25% | 0.45 | 2,032 (1.6%) | 1% | 0 | free_folds |
| suite | mlp-s-seed2 | potodds | 100 | 40,000 | -14 | 64% | 23% | - | 18% | 31% | 1.70 | 5,879 (6.9%) | 1% | 0 | free_folds |
| suite | mlp-s-seed2 | random | 100 | 40,000 | 342 | 58% | 23% | 19% | 43% | 17% | 0.53 | 651 (0.8%) | 3% | 0 | free_folds |
| suite | mlp-s-seed2 | tag | 100 | 40,000 | -139 | 66% | 21% | 14% | 20% | 24% | 0.70 | 2,622 (2.9%) | 1% | 0 | free_folds |
| suite | mlp-s-seed2 | tight | 100 | 40,000 | -10 | 65% | 24% | 14% | 19% | 34% | 1.50 | 4,917 (5.3%) | 1% | 0 | free_folds |
| suite | mlp-s-seed3 | bluff | 100 | 40,000 | 629 | 67% | 29% | 21% | 32% | 18% | 0.70 | 6,639 (5.5%) | 2% | 0 | free_folds |
| suite | mlp-s-seed3 | call | 100 | 40,000 | 364 | 61% | 31% | - | 28% | 46% | 2.82 | 13,324 (11.7%) | 1% | 0 | free_folds |
| suite | mlp-s-seed3 | lag | 100 | 40,000 | -175 | 70% | 32% | 14% | 18% | 25% | 0.47 | 1,975 (1.5%) | 1% | 0 | free_folds |
| suite | mlp-s-seed3 | potodds | 100 | 40,000 | 23 | 66% | 31% | - | 16% | 29% | 2.12 | 5,424 (6.5%) | 0% | 0 | free_folds |
| suite | mlp-s-seed3 | random | 100 | 40,000 | 405 | 60% | 23% | 16% | 42% | 16% | 0.53 | 626 (0.8%) | 4% | 0 | free_folds |
| suite | mlp-s-seed3 | tag | 100 | 40,000 | -138 | 69% | 26% | 5% | 18% | 20% | 0.80 | 2,907 (3.4%) | 1% | 0 | free_folds |
| suite | mlp-s-seed3 | tight | 100 | 40,000 | 5 | 68% | 30% | 5% | 17% | 32% | 1.78 | 4,795 (5.3%) | 0% | 0 | free_folds |
| suite | random-bot | bluff | 100 | 40,000 | -627 | 81% | 69% | 65% | 20% | 52% | 3.90 | 4,585 (4.8%) | 20% | 0 | free_folds |
| suite | random-bot | call | 100 | 40,000 | -153 | 78% | 65% | - | 18% | 59% | 11.68 | 11,027 (11.6%) | 17% | 0 | free_folds |
| suite | random-bot | lag | 100 | 40,000 | -890 | 74% | 65% | 66% | 20% | 32% | 3.88 | 2,614 (3.4%) | 19% | 0 | free_folds |
| suite | random-bot | potodds | 100 | 40,000 | -261 | 66% | 51% | - | 20% | 10% | 5.87 | 4,210 (7.7%) | 18% | 0 | free_folds |
| suite | random-bot | random | 100 | 40,000 | 24 | 77% | 60% | 52% | 24% | 55% | 2.22 | 581 (0.9%) | 24% | 0 | free_folds |
| suite | random-bot | tag | 100 | 40,000 | -563 | 66% | 51% | 60% | 20% | 14% | 3.94 | 2,234 (4.2%) | 19% | 0 | free_folds |
| suite | random-bot | tight | 100 | 40,000 | -679 | 69% | 55% | 59% | 20% | 23% | 5.65 | 4,595 (7.4%) | 18% | 0 | free_folds |

Result by position (bb/100; seat relative to the button, 4 seats) and final pot (bb: mean / median / 90th percentile; share of hands in bins <=3, <=6, <=12, <=25, <=50, <=100, <=200, >200).

| part | agent | opponent | stack bb | BTN | SB | BB | UTG | pot mean / median / p90 | pot bins % |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| cross | gru-seed1 | mlp-s-seed3 | 100 | -389 | -203 | -389 | -211 | 87.1 / 38.5 / 215.5 | 4 8 14 15 15 8 13 23 |
| cross | gru-seed2 | mlp-s-seed3 | 100 | -200 | -254 | -339 | -108 | 79.2 / 34.0 / 210.0 | 5 9 15 16 14 8 16 18 |
| cross | gru-seed3 | mlp-s-seed3 | 100 | -193 | -108 | -194 | -21 | 76.3 / 32.0 / 209.0 | 5 9 15 16 14 8 15 18 |
| cross | mlp-l-seed1 | mlp-s-seed3 | 100 | -705 | -627 | -573 | -533 | 97.3 / 55.5 / 217.0 | 2 5 12 14 16 9 18 25 |
| cross | mlp-l-seed2 | mlp-s-seed3 | 100 | -478 | -599 | -488 | -347 | 87.0 / 41.0 / 213.0 | 2 7 14 15 16 9 15 21 |
| cross | mlp-l-seed3 | mlp-s-seed3 | 100 | -481 | -553 | -445 | -394 | 90.9 / 44.0 / 217.0 | 3 7 14 14 15 9 15 24 |
| cross | mlp-s-seed1 | mlp-s-seed3 | 100 | -71 | 3 | -99 | 3 | 62.6 / 25.0 / 203.5 | 3 9 18 20 16 8 13 13 |
| cross | mlp-s-seed2 | mlp-s-seed3 | 100 | -64 | 7 | -99 | 32 | 62.5 / 25.0 / 202.5 | 4 10 18 19 16 8 13 12 |
| formats | gru-seed3 | call | 50 | 269 | 234 | 301 | 287 | 38.8 / 8.0 / 200.0 | 18 21 19 21 4 1 15 0 |
| formats | gru-seed3 | call | 100 | 408 | 136 | 292 | 324 | 32.4 / 8.0 / 26.0 | 22 24 21 22 3 1 0 6 |
| formats | gru-seed3 | call | 200 | 114 | 35 | 34 | 81 | 12.7 / 3.5 / 20.0 | 35 33 14 15 1 1 0 1 |
| formats | gru-seed3 | call | 400 | 1 | -45 | -96 | 0 | 3.4 / 3.5 / 4.0 | 50 50 0 0 0 0 0 0 |
| formats | gru-seed3 | random | 50 | 493 | 474 | 561 | 526 | 108.9 / 104.0 / 154.5 | 2 3 3 3 3 15 72 0 |
| formats | gru-seed3 | random | 100 | 935 | 722 | 799 | 673 | 198.2 / 202.5 / 302.0 | 3 3 3 3 2 2 16 68 |
| formats | gru-seed3 | random | 200 | 873 | 704 | 657 | 471 | 349.9 / 401.0 / 601.0 | 5 5 4 4 2 1 1 78 |
| formats | gru-seed3 | random | 400 | 10 | 210 | 84 | 0 | 644.3 / 801.0 / 1200.5 | 6 6 4 4 2 1 1 76 |
| formats | gru-seed3 | tight | 50 | -2 | -53 | -24 | 35 | 11.2 / 3.5 / 19.0 | 43 31 13 3 2 4 3 0 |
| formats | gru-seed3 | tight | 100 | 7 | -60 | -14 | 16 | 10.8 / 3.0 / 11.5 | 52 27 11 3 1 1 2 1 |
| formats | gru-seed3 | tight | 200 | 51 | -41 | -12 | 18 | 5.8 / 1.5 / 5.5 | 77 16 4 1 0 0 1 1 |
| formats | gru-seed3 | tight | 400 | 0 | -50 | -26 | 0 | 4.6 / 1.5 / 3.5 | 88 10 2 0 0 0 0 0 |
| formats | mlp-s-seed2 | call | 50 | 117 | 106 | 61 | 138 | 13.6 / 8.0 / 20.0 | 13 35 31 17 1 0 3 0 |
| formats | mlp-s-seed2 | call | 100 | 239 | 266 | 76 | 182 | 18.9 / 4.0 / 28.0 | 13 38 23 15 7 2 0 2 |
| formats | mlp-s-seed2 | call | 200 | 150 | 120 | -28 | 138 | 52.5 / 40.0 / 90.0 | 6 6 11 7 28 35 6 1 |
| formats | mlp-s-seed2 | call | 400 | 85 | 81 | -91 | 317 | 183.2 / 180.0 / 270.0 | 0 0 1 0 1 23 27 48 |
| formats | mlp-s-seed2 | random | 50 | 338 | 219 | 292 | 326 | 108.7 / 104.0 / 164.0 | 2 3 3 3 3 15 70 0 |
| formats | mlp-s-seed2 | random | 100 | 526 | 326 | 308 | 339 | 192.9 / 202.0 / 302.0 | 2 2 3 3 3 3 18 65 |
| formats | mlp-s-seed2 | random | 200 | -132 | -184 | -246 | -320 | 364.9 / 402.0 / 602.5 | 1 1 2 2 3 4 3 83 |
| formats | mlp-s-seed2 | random | 400 | -854 | -1,205 | -1,113 | -1,423 | 727.7 / 802.5 / 1202.5 | 0 1 1 1 1 2 4 89 |
| formats | mlp-s-seed2 | tight | 50 | 3 | -45 | -7 | -3 | 8.5 / 3.0 / 11.0 | 51 30 11 2 1 3 2 0 |
| formats | mlp-s-seed2 | tight | 100 | 3 | -52 | 0 | -5 | 9.7 / 3.0 / 11.0 | 53 27 12 4 1 1 2 1 |
| formats | mlp-s-seed2 | tight | 200 | -70 | -93 | -114 | -39 | 15.7 / 4.0 / 39.0 | 26 34 15 10 9 4 1 1 |
| formats | mlp-s-seed2 | tight | 400 | -625 | -586 | -642 | -652 | 42.6 / 6.0 / 121.5 | 12 38 16 2 12 6 8 5 |
| formats | mlp-s-seed3 | call | 50 | 161 | 98 | 137 | 265 | 15.5 / 4.0 / 20.0 | 15 52 19 8 1 1 5 0 |
| formats | mlp-s-seed3 | call | 100 | 279 | 328 | 269 | 426 | 21.8 / 4.0 / 20.0 | 15 43 22 12 3 2 0 3 |
| formats | mlp-s-seed3 | call | 200 | 283 | 214 | 320 | 403 | 25.0 / 4.0 / 20.0 | 23 35 20 17 2 0 0 2 |
| formats | mlp-s-seed3 | call | 400 | 320 | 75 | 346 | 514 | 35.0 / 4.0 / 18.0 | 34 38 11 12 3 0 0 2 |
| formats | mlp-s-seed3 | random | 50 | 195 | 248 | 220 | 315 | 103.7 / 102.5 / 152.5 | 2 3 3 4 3 17 68 0 |
| formats | mlp-s-seed3 | random | 100 | 377 | 358 | 432 | 580 | 193.9 / 202.0 / 302.0 | 2 2 3 3 3 3 18 65 |
| formats | mlp-s-seed3 | random | 200 | 449 | 579 | 238 | 871 | 366.9 / 401.5 / 601.0 | 2 3 3 3 3 2 2 82 |
| formats | mlp-s-seed3 | random | 400 | 709 | 211 | 743 | 1,089 | 730.0 / 801.5 / 1201.0 | 3 4 3 3 2 2 2 80 |
| formats | mlp-s-seed3 | tight | 50 | 18 | -35 | -14 | 26 | 7.3 / 3.0 / 9.0 | 56 26 11 2 0 2 2 0 |
| formats | mlp-s-seed3 | tight | 100 | 33 | 14 | 25 | 24 | 7.0 / 3.0 / 9.5 | 51 29 15 3 0 0 1 1 |
| formats | mlp-s-seed3 | tight | 200 | 18 | 28 | 27 | -5 | 7.4 / 3.0 / 9.5 | 53 27 16 3 0 0 0 1 |
| formats | mlp-s-seed3 | tight | 400 | -4 | 1 | -36 | -84 | 11.1 / 2.5 / 8.5 | 64 20 12 2 0 0 0 1 |
| formats | random-bot | call | 50 | -172 | -196 | -262 | -18 | 102.1 / 72.0 / 200.0 | 9 12 10 8 6 8 47 0 |
| formats | random-bot | call | 100 | -322 | -175 | -404 | 149 | 189.4 / 90.0 / 400.0 | 9 12 10 8 6 6 5 44 |
| formats | random-bot | call | 200 | -604 | -379 | -623 | -157 | 357.7 / 90.0 / 800.0 | 9 12 10 8 6 6 4 45 |
| formats | random-bot | call | 400 | -341 | -587 | -334 | 426 | 688.4 / 90.0 / 1600.0 | 9 12 10 8 6 6 4 45 |
| formats | random-bot | random | 50 | 23 | -22 | -115 | 148 | 122.0 / 125.5 / 200.0 | 2 2 2 2 2 13 78 0 |
| formats | random-bot | random | 100 | 148 | -204 | 109 | 128 | 238.9 / 229.5 / 400.0 | 2 2 2 2 1 2 12 78 |
| formats | random-bot | random | 200 | -95 | -112 | -94 | 227 | 470.8 / 435.5 / 800.0 | 2 2 2 2 1 1 1 89 |
| formats | random-bot | random | 400 | 157 | -117 | -74 | 712 | 931.0 / 834.5 / 1600.0 | 2 2 2 2 1 1 1 89 |
| formats | random-bot | tight | 50 | -308 | -384 | -384 | -284 | 31.0 / 4.5 / 101.0 | 32 23 8 4 2 15 16 0 |
| formats | random-bot | tight | 100 | -653 | -736 | -633 | -621 | 57.6 / 4.5 / 201.0 | 33 23 7 4 2 1 15 16 |
| formats | random-bot | tight | 200 | -1,383 | -1,277 | -1,310 | -1,237 | 110.8 / 4.5 / 401.0 | 33 22 8 4 2 1 1 30 |
| formats | random-bot | tight | 400 | -2,797 | -2,875 | -2,897 | -2,484 | 213.2 / 4.5 / 801.0 | 33 22 8 4 2 1 1 30 |
| suite | gru-seed1 | bluff | 100 | 857 | 654 | 469 | 768 | 195.3 / 187.5 / 400.0 | 1 1 4 7 9 9 22 46 |
| suite | gru-seed1 | call | 100 | 166 | 100 | 27 | 143 | 32.2 / 10.0 / 40.0 | 19 23 15 22 13 4 1 4 |
| suite | gru-seed1 | lag | 100 | -211 | -540 | -398 | -179 | 53.2 / 14.5 / 189.5 | 24 16 8 14 9 5 15 8 |
| suite | gru-seed1 | potodds | 100 | 16 | -44 | 4 | 7 | 4.8 / 3.0 / 9.0 | 56 27 13 2 0 0 0 0 |
| suite | gru-seed1 | random | 100 | 1,033 | 960 | 974 | 978 | 206.7 / 203.0 / 303.0 | 2 3 3 3 2 2 15 70 |
| suite | gru-seed1 | tag | 100 | -267 | -406 | -251 | -250 | 19.5 / 3.5 / 69.0 | 46 31 8 4 1 2 6 2 |
| suite | gru-seed1 | tight | 100 | 22 | -107 | -24 | 9 | 11.1 / 3.5 / 12.0 | 46 28 17 5 1 1 2 1 |
| suite | gru-seed2 | bluff | 100 | 807 | 613 | 544 | 685 | 185.2 / 177.5 / 400.0 | 1 1 4 8 10 10 24 43 |
| suite | gru-seed2 | call | 100 | 338 | 291 | 297 | 309 | 41.4 / 8.0 / 48.0 | 21 20 20 23 7 1 1 7 |
| suite | gru-seed2 | lag | 100 | -64 | -307 | -225 | -163 | 50.1 / 13.0 / 177.0 | 31 13 5 15 8 6 15 7 |
| suite | gru-seed2 | potodds | 100 | -17 | -42 | -37 | -23 | 7.0 / 2.0 / 9.0 | 65 22 8 2 0 0 1 1 |
| suite | gru-seed2 | random | 100 | 1,016 | 930 | 781 | 900 | 206.5 / 203.0 / 305.5 | 3 3 3 3 2 2 16 69 |
| suite | gru-seed2 | tag | 100 | -138 | -193 | -177 | -157 | 15.7 / 3.0 / 20.5 | 55 28 3 3 1 1 5 2 |
| suite | gru-seed2 | tight | 100 | -41 | -67 | -104 | -11 | 14.0 / 3.0 / 12.0 | 54 25 11 3 1 1 3 2 |
| suite | gru-seed3 | bluff | 100 | 1,079 | 727 | 604 | 876 | 175.6 / 168.0 / 400.0 | 1 1 5 9 11 10 24 40 |
| suite | gru-seed3 | call | 100 | 407 | 274 | 347 | 275 | 31.1 / 8.0 / 24.0 | 22 24 22 23 3 1 0 5 |
| suite | gru-seed3 | lag | 100 | -52 | -305 | -184 | -17 | 47.1 / 12.5 / 165.0 | 30 15 5 15 9 6 16 5 |
| suite | gru-seed3 | potodds | 100 | 6 | -18 | -3 | 26 | 4.6 / 2.0 / 8.0 | 63 25 8 3 0 0 0 0 |
| suite | gru-seed3 | random | 100 | 1,057 | 783 | 658 | 794 | 199.3 / 202.5 / 302.0 | 3 3 3 3 2 2 16 68 |
| suite | gru-seed3 | tag | 100 | -144 | -214 | -142 | -84 | 15.0 / 3.0 / 20.5 | 51 32 5 3 1 2 5 1 |
| suite | gru-seed3 | tight | 100 | 0 | -17 | -34 | 38 | 11.0 / 3.0 / 12.0 | 52 27 12 3 1 1 3 1 |
| suite | mlp-l-seed1 | bluff | 100 | 734 | 369 | 503 | 588 | 189.6 / 187.0 / 400.0 | 0 1 3 6 8 10 26 46 |
| suite | mlp-l-seed1 | call | 100 | 335 | 164 | 269 | 229 | 42.6 / 12.0 / 64.0 | 6 19 28 31 5 2 1 7 |
| suite | mlp-l-seed1 | lag | 100 | -334 | -575 | -550 | -420 | 62.3 / 22.5 / 192.0 | 15 13 9 15 11 9 20 8 |
| suite | mlp-l-seed1 | potodds | 100 | -51 | -9 | -22 | -34 | 7.9 / 3.5 / 10.5 | 39 36 19 4 1 0 1 1 |
| suite | mlp-l-seed1 | random | 100 | 521 | 469 | 440 | 473 | 203.5 / 203.0 / 302.5 | 1 2 2 2 2 2 18 70 |
| suite | mlp-l-seed1 | tag | 100 | -294 | -388 | -324 | -430 | 24.1 / 3.5 / 103.0 | 36 33 10 5 2 4 8 3 |
| suite | mlp-l-seed1 | tight | 100 | -46 | -56 | -74 | -73 | 14.7 / 3.5 / 16.5 | 30 35 21 6 2 1 3 2 |
| suite | mlp-l-seed2 | bluff | 100 | 535 | 230 | 222 | 330 | 171.1 / 175.0 / 314.0 | 0 1 4 7 9 11 27 41 |
| suite | mlp-l-seed2 | call | 100 | 22 | -81 | -95 | 34 | 16.4 / 10.0 / 20.0 | 8 24 29 32 5 1 0 1 |
| suite | mlp-l-seed2 | lag | 100 | -144 | -539 | -365 | -299 | 55.1 / 20.0 / 172.0 | 17 12 10 16 12 9 18 6 |
| suite | mlp-l-seed2 | potodds | 100 | -5 | 0 | -12 | -27 | 5.9 / 3.5 / 10.0 | 48 32 16 3 1 0 0 0 |
| suite | mlp-l-seed2 | random | 100 | 441 | 306 | 178 | 276 | 198.3 / 202.5 / 302.0 | 1 2 2 3 2 3 19 68 |
| suite | mlp-l-seed2 | tag | 100 | -240 | -282 | -227 | -318 | 20.4 / 3.5 / 74.0 | 42 29 9 5 3 3 6 2 |
| suite | mlp-l-seed2 | tight | 100 | -7 | -43 | -33 | -50 | 12.2 / 3.5 / 14.0 | 37 32 19 5 2 1 2 1 |
| suite | mlp-l-seed3 | bluff | 100 | 793 | 520 | 630 | 712 | 189.8 / 188.0 / 400.0 | 0 1 4 7 9 9 23 46 |
| suite | mlp-l-seed3 | call | 100 | 129 | 112 | 103 | 149 | 23.9 / 8.0 / 26.0 | 11 27 26 26 5 2 1 3 |
| suite | mlp-l-seed3 | lag | 100 | -263 | -550 | -424 | -333 | 59.2 / 19.5 / 189.0 | 18 16 9 13 10 7 19 8 |
| suite | mlp-l-seed3 | potodds | 100 | 24 | -10 | -6 | -24 | 6.9 / 3.5 / 10.0 | 46 36 13 3 0 0 1 0 |
| suite | mlp-l-seed3 | random | 100 | 830 | 596 | 464 | 719 | 211.3 / 204.0 / 303.5 | 1 2 3 2 2 2 15 72 |
| suite | mlp-l-seed3 | tag | 100 | -231 | -386 | -297 | -266 | 22.0 / 3.5 / 101.0 | 40 34 8 4 2 2 8 2 |
| suite | mlp-l-seed3 | tight | 100 | 4 | -69 | -62 | -40 | 13.7 / 3.5 / 13.5 | 37 35 17 5 1 1 2 2 |
| suite | mlp-s-seed1 | bluff | 100 | 692 | 416 | 560 | 482 | 153.0 / 150.0 / 301.0 | 0 1 5 10 11 12 27 33 |
| suite | mlp-s-seed1 | call | 100 | 144 | 50 | 79 | 105 | 15.7 / 8.0 / 20.0 | 11 37 33 12 3 2 0 1 |
| suite | mlp-s-seed1 | lag | 100 | -51 | -249 | -199 | -92 | 44.4 / 16.5 / 147.5 | 16 15 12 18 13 8 13 4 |
| suite | mlp-s-seed1 | potodds | 100 | -2 | -4 | 7 | 6 | 4.4 / 3.0 / 7.0 | 59 30 9 1 0 0 0 0 |
| suite | mlp-s-seed1 | random | 100 | 500 | 385 | 423 | 294 | 194.4 / 202.5 / 302.0 | 1 2 3 4 3 2 17 67 |
| suite | mlp-s-seed1 | tag | 100 | -98 | -176 | -159 | -84 | 14.2 / 3.5 / 25.0 | 46 28 11 6 3 2 3 1 |
| suite | mlp-s-seed1 | tight | 100 | -16 | -46 | -17 | 6 | 8.3 / 3.5 / 10.0 | 46 32 15 3 1 1 1 1 |
| suite | mlp-s-seed2 | bluff | 100 | 443 | 175 | 189 | 377 | 146.4 / 150.0 / 284.0 | 0 1 6 9 11 12 27 32 |
| suite | mlp-s-seed2 | call | 100 | 249 | 216 | 139 | 186 | 19.8 / 4.0 / 28.0 | 13 38 23 14 7 2 0 2 |
| suite | mlp-s-seed2 | lag | 100 | -76 | -241 | -126 | -132 | 43.6 / 18.0 / 140.0 | 19 12 10 19 14 8 15 2 |
| suite | mlp-s-seed2 | potodds | 100 | 2 | -60 | -2 | 6 | 6.5 / 3.0 / 8.0 | 64 23 8 2 1 0 1 0 |
| suite | mlp-s-seed2 | random | 100 | 556 | 248 | 285 | 279 | 194.8 / 202.0 / 302.0 | 2 2 3 3 3 3 18 66 |
| suite | mlp-s-seed2 | tag | 100 | -131 | -193 | -103 | -129 | 15.7 / 3.5 / 44.0 | 48 24 9 5 4 4 5 1 |
| suite | mlp-s-seed2 | tight | 100 | 2 | -50 | -5 | 14 | 10.2 / 3.0 / 11.0 | 52 26 13 4 1 1 2 1 |
| suite | mlp-s-seed3 | bluff | 100 | 556 | 525 | 643 | 794 | 150.1 / 144.0 / 300.0 | 0 2 6 10 11 12 27 32 |
| suite | mlp-s-seed3 | call | 100 | 310 | 370 | 393 | 384 | 21.9 / 4.0 / 20.0 | 15 43 22 12 3 2 0 3 |
| suite | mlp-s-seed3 | lag | 100 | -62 | -278 | -223 | -136 | 44.0 / 16.5 / 144.0 | 18 14 12 16 14 9 13 4 |
| suite | mlp-s-seed3 | potodds | 100 | 31 | 15 | 20 | 26 | 4.5 / 3.0 / 6.5 | 64 25 9 1 0 0 0 0 |
| suite | mlp-s-seed3 | random | 100 | 436 | 397 | 298 | 490 | 194.7 / 202.0 / 302.0 | 2 2 3 3 3 3 19 66 |
| suite | mlp-s-seed3 | tag | 100 | -88 | -177 | -145 | -141 | 14.8 / 3.5 / 38.5 | 48 25 10 5 4 3 4 1 |
| suite | mlp-s-seed3 | tight | 100 | 17 | -14 | 7 | 11 | 7.4 / 3.0 / 10.0 | 51 29 15 3 1 0 1 1 |
| suite | random-bot | bluff | 100 | -522 | -735 | -786 | -467 | 277.5 / 358.5 / 400.0 | 0 1 2 4 5 5 15 68 |
| suite | random-bot | call | 100 | 15 | -242 | -283 | -103 | 188.2 / 90.0 / 400.0 | 9 12 10 8 6 6 5 44 |
| suite | random-bot | lag | 100 | -778 | -1,076 | -896 | -810 | 100.6 / 49.5 / 271.5 | 18 13 6 8 6 4 18 28 |
| suite | random-bot | potodds | 100 | -272 | -259 | -215 | -297 | 34.2 / 4.0 / 101.5 | 40 26 5 2 1 1 22 3 |
| suite | random-bot | random | 100 | 227 | -112 | 83 | -100 | 240.9 / 248.0 / 400.0 | 1 2 2 2 1 2 12 78 |
| suite | random-bot | tag | 100 | -537 | -655 | -532 | -529 | 46.1 / 4.0 / 200.0 | 36 27 4 3 2 1 19 10 |
| suite | random-bot | tight | 100 | -619 | -725 | -707 | -664 | 56.9 / 4.5 / 201.0 | 33 23 8 3 2 1 15 15 |

Result by starting hand (bb/100): pairs, suited and offsuit classes pooled, and the best and worst class with at least 30 hands (each class has few hands, so single classes are noisy: see `by_hand` in diagnostics.json).

| part | agent | opponent | stack bb | pairs | suited | offsuit | best class | worst class |
|---|---|---|---:|---:|---:|---:|---|---|
| cross | gru-seed1 | mlp-s-seed3 | 100 | 900 | -221 | -421 | AA 3,813 | QTs -1,546 |
| cross | gru-seed2 | mlp-s-seed3 | 100 | 820 | -170 | -329 | AA 3,636 | K4o -1,649 |
| cross | gru-seed3 | mlp-s-seed3 | 100 | 966 | -145 | -213 | KK 4,063 | J9o -1,222 |
| cross | mlp-l-seed1 | mlp-s-seed3 | 100 | 692 | -553 | -735 | KK 3,625 | J3s -2,266 |
| cross | mlp-l-seed2 | mlp-s-seed3 | 100 | 665 | -485 | -569 | AA 3,878 | Q4o -1,719 |
| cross | mlp-l-seed3 | mlp-s-seed3 | 100 | 780 | -431 | -582 | AA 3,316 | Q3o -1,692 |
| cross | mlp-s-seed1 | mlp-s-seed3 | 100 | 890 | -77 | -105 | AA 3,519 | 65s -734 |
| cross | mlp-s-seed2 | mlp-s-seed3 | 100 | 791 | -78 | -82 | KK 3,325 | K3s -968 |
| formats | gru-seed3 | call | 50 | 1,737 | 213 | 165 | AA 7,328 | Q9s -783 |
| formats | gru-seed3 | call | 100 | 1,847 | 212 | 181 | AA 9,252 | Q9s -1,408 |
| formats | gru-seed3 | call | 200 | 537 | 49 | 31 | AA 3,651 | A3s -1,246 |
| formats | gru-seed3 | call | 400 | -6 | -37 | -37 | KK 281 | J6s -56 |
| formats | gru-seed3 | random | 50 | 2,088 | 522 | 374 | AA 6,749 | 97s -475 |
| formats | gru-seed3 | random | 100 | 3,309 | 716 | 585 | AA 12,013 | T9s -1,083 |
| formats | gru-seed3 | random | 200 | 3,990 | 422 | 472 | AA 24,903 | A8s -2,254 |
| formats | gru-seed3 | random | 400 | 624 | 69 | 30 | AA 9,183 | QTo -815 |
| formats | gru-seed3 | tight | 50 | 198 | -3 | -32 | AA 1,249 | T4s -505 |
| formats | gru-seed3 | tight | 100 | 157 | -8 | -29 | AA 1,482 | A9s -809 |
| formats | gru-seed3 | tight | 200 | 100 | 17 | -9 | AQs 1,912 | A8s -507 |
| formats | gru-seed3 | tight | 400 | -19 | -19 | -19 | JJ -5 | 96s -33 |
| formats | mlp-s-seed2 | call | 50 | 458 | 152 | 59 | AA 1,928 | T9s -587 |
| formats | mlp-s-seed2 | call | 100 | 854 | 213 | 125 | AA 2,749 | 85s -484 |
| formats | mlp-s-seed2 | call | 200 | 1,212 | 129 | -14 | AA 3,658 | Q2s -959 |
| formats | mlp-s-seed2 | call | 400 | 3,291 | 360 | -267 | AA 8,296 | 72s -3,401 |
| formats | mlp-s-seed2 | random | 50 | 2,274 | 246 | 138 | AA 5,413 | T2s -1,431 |
| formats | mlp-s-seed2 | random | 100 | 3,927 | 249 | 108 | AA 13,594 | Q2s -1,815 |
| formats | mlp-s-seed2 | random | 200 | 2,007 | -292 | -390 | AA 12,043 | Q2s -3,702 |
| formats | mlp-s-seed2 | random | 400 | 825 | -1,505 | -1,202 | KK 13,184 | Q2s -12,389 |
| formats | mlp-s-seed2 | tight | 50 | 144 | -12 | -27 | AA 990 | 77 -544 |
| formats | mlp-s-seed2 | tight | 100 | 225 | -17 | -33 | AA 1,513 | AJs -819 |
| formats | mlp-s-seed2 | tight | 200 | 77 | -58 | -100 | AA 1,715 | JJ -889 |
| formats | mlp-s-seed2 | tight | 400 | -159 | -562 | -688 | AA 2,682 | AJs -1,594 |
| formats | mlp-s-seed3 | call | 50 | 808 | 126 | 122 | KK 2,774 | K3s -328 |
| formats | mlp-s-seed3 | call | 100 | 1,201 | 339 | 245 | KK 4,745 | K7s -393 |
| formats | mlp-s-seed3 | call | 200 | 1,435 | 194 | 244 | KK 7,364 | 85s -1,629 |
| formats | mlp-s-seed3 | call | 400 | 3,253 | 316 | 57 | 88 21,755 | 85s -854 |
| formats | mlp-s-seed3 | random | 50 | 2,078 | 168 | 110 | AA 5,303 | 93o -743 |
| formats | mlp-s-seed3 | random | 100 | 3,697 | 295 | 201 | AA 11,970 | K3s -1,518 |
| formats | mlp-s-seed3 | random | 200 | 5,160 | 579 | 117 | AA 18,476 | 97s -2,512 |
| formats | mlp-s-seed3 | random | 400 | 5,637 | 871 | 197 | K8s 19,978 | K6s -7,332 |
| formats | mlp-s-seed3 | tight | 50 | 207 | -8 | -17 | AA 1,080 | J7s -388 |
| formats | mlp-s-seed3 | tight | 100 | 288 | 26 | 1 | AA 1,141 | KQs -437 |
| formats | mlp-s-seed3 | tight | 200 | 98 | 11 | 12 | AA 1,571 | Q8s -833 |
| formats | mlp-s-seed3 | tight | 400 | 154 | -16 | -51 | 88 497 | Q8s -1,703 |
| formats | random-bot | call | 50 | 1,505 | -41 | -347 | AA 3,852 | Q2s -1,955 |
| formats | random-bot | call | 100 | 2,997 | 147 | -576 | AA 9,369 | 72s -4,664 |
| formats | random-bot | call | 200 | 4,896 | 125 | -1,093 | AA 14,281 | 72s -7,712 |
| formats | random-bot | call | 400 | 12,416 | 1,036 | -1,720 | AA 36,698 | 72s -15,881 |
| formats | random-bot | random | 50 | 1,202 | -10 | -89 | AA 2,967 | 83s -2,020 |
| formats | random-bot | random | 100 | 2,420 | 106 | -181 | QQ 6,250 | 83s -5,439 |
| formats | random-bot | random | 200 | 5,245 | 621 | -689 | AA 15,214 | 72s -9,854 |
| formats | random-bot | random | 400 | 11,088 | 387 | -852 | AA 26,862 | 72s -14,805 |
| formats | random-bot | tight | 50 | 57 | -290 | -391 | AA 919 | T4s -1,076 |
| formats | random-bot | tight | 100 | 168 | -507 | -784 | AA 1,487 | 52s -1,990 |
| formats | random-bot | tight | 200 | 353 | -1,151 | -1,495 | AA 4,561 | 43o -4,049 |
| formats | random-bot | tight | 400 | 583 | -2,276 | -3,216 | K3s 5,997 | T4s -8,690 |
| suite | gru-seed1 | bluff | 100 | 3,801 | 870 | 370 | KK 12,233 | J5o -1,263 |
| suite | gru-seed1 | call | 100 | 980 | 153 | 23 | QQ 3,964 | K3o -526 |
| suite | gru-seed1 | lag | 100 | 477 | -271 | -419 | AA 2,110 | K9o -1,447 |
| suite | gru-seed1 | potodds | 100 | 70 | -2 | -11 | TT 288 | Q6s -278 |
| suite | gru-seed1 | random | 100 | 3,661 | 1,116 | 724 | KK 13,079 | T7s -888 |
| suite | gru-seed1 | tag | 100 | 18 | -283 | -323 | AA 1,027 | Q6s -1,337 |
| suite | gru-seed1 | tight | 100 | 154 | -20 | -42 | AA 1,008 | AQs -610 |
| suite | gru-seed2 | bluff | 100 | 3,702 | 764 | 379 | AA 13,871 | T4s -1,570 |
| suite | gru-seed2 | call | 100 | 1,743 | 403 | 160 | AA 9,924 | J4o -499 |
| suite | gru-seed2 | lag | 100 | 617 | -183 | -258 | AA 2,744 | A3s -1,222 |
| suite | gru-seed2 | potodds | 100 | 72 | -40 | -35 | AA 479 | K5s -344 |
| suite | gru-seed2 | random | 100 | 3,472 | 991 | 668 | KK 11,721 | T7s -1,117 |
| suite | gru-seed2 | tag | 100 | 74 | -165 | -187 | AA 1,177 | T4s -781 |
| suite | gru-seed2 | tight | 100 | 177 | -51 | -76 | AA 1,204 | T8s -545 |
| suite | gru-seed3 | bluff | 100 | 4,601 | 868 | 497 | KK 15,402 | Q8s -533 |
| suite | gru-seed3 | call | 100 | 2,082 | 330 | 180 | AA 7,934 | Q8s -332 |
| suite | gru-seed3 | lag | 100 | 506 | -121 | -198 | AA 2,191 | A3s -1,359 |
| suite | gru-seed3 | potodds | 100 | 65 | 8 | -4 | AA 652 | A5s -274 |
| suite | gru-seed3 | random | 100 | 3,444 | 917 | 577 | KK 11,187 | Q8o -439 |
| suite | gru-seed3 | tag | 100 | 23 | -123 | -168 | AA 957 | A3s -1,125 |
| suite | gru-seed3 | tight | 100 | 125 | 14 | -20 | AA 1,030 | AQs -485 |
| suite | mlp-l-seed1 | bluff | 100 | 3,701 | 692 | 242 | KK 12,878 | 53s -1,491 |
| suite | mlp-l-seed1 | call | 100 | 1,323 | 334 | 133 | KK 6,309 | 64s -902 |
| suite | mlp-l-seed1 | lag | 100 | 458 | -419 | -563 | AA 1,990 | 62o -1,389 |
| suite | mlp-l-seed1 | potodds | 100 | 82 | -29 | -38 | AA 372 | 65s -295 |
| suite | mlp-l-seed1 | random | 100 | 3,351 | 393 | 268 | AA 9,981 | T7s -973 |
| suite | mlp-l-seed1 | tag | 100 | 2 | -352 | -391 | AA 994 | Q6s -994 |
| suite | mlp-l-seed1 | tight | 100 | 231 | -41 | -93 | AA 1,123 | 63o -444 |
| suite | mlp-l-seed2 | bluff | 100 | 3,129 | 447 | 60 | KK 10,304 | 63s -1,220 |
| suite | mlp-l-seed2 | call | 100 | 310 | -24 | -60 | KK 870 | Q8s -388 |
| suite | mlp-l-seed2 | lag | 100 | 364 | -309 | -403 | AA 2,028 | A3s -1,072 |
| suite | mlp-l-seed2 | potodds | 100 | 55 | -9 | -17 | AA 290 | A3s -150 |
| suite | mlp-l-seed2 | random | 100 | 3,102 | 338 | 58 | KK 10,515 | 63s -920 |
| suite | mlp-l-seed2 | tag | 100 | 33 | -271 | -289 | AA 970 | Q8s -836 |
| suite | mlp-l-seed2 | tight | 100 | 171 | -30 | -51 | AA 860 | A3s -507 |
| suite | mlp-l-seed3 | bluff | 100 | 3,884 | 795 | 356 | AA 11,248 | J6s -1,692 |
| suite | mlp-l-seed3 | call | 100 | 899 | 143 | 53 | AA 2,913 | J7s -560 |
| suite | mlp-l-seed3 | lag | 100 | 355 | -368 | -462 | AA 2,310 | Q6s -1,546 |
| suite | mlp-l-seed3 | potodds | 100 | 89 | -7 | -11 | AA 502 | Q6s -360 |
| suite | mlp-l-seed3 | random | 100 | 3,697 | 745 | 371 | AA 9,748 | 96s -1,073 |
| suite | mlp-l-seed3 | tag | 100 | -48 | -300 | -314 | AA 1,042 | Q6s -1,370 |
| suite | mlp-l-seed3 | tight | 100 | 227 | -58 | -58 | AA 1,108 | Q6s -742 |
| suite | mlp-s-seed1 | bluff | 100 | 3,503 | 450 | 324 | AA 10,839 | 76s -1,124 |
| suite | mlp-s-seed1 | call | 100 | 788 | 94 | 38 | KK 2,528 | KJs -241 |
| suite | mlp-s-seed1 | lag | 100 | 645 | -197 | -196 | AA 2,506 | A3s -1,303 |
| suite | mlp-s-seed1 | potodds | 100 | 79 | 1 | -4 | J9s 226 | KJs -155 |
| suite | mlp-s-seed1 | random | 100 | 3,557 | 349 | 159 | KK 10,449 | T7s -908 |
| suite | mlp-s-seed1 | tag | 100 | 68 | -150 | -139 | AA 1,061 | A3s -619 |
| suite | mlp-s-seed1 | tight | 100 | 211 | -43 | -29 | AA 1,251 | A9s -529 |
| suite | mlp-s-seed2 | bluff | 100 | 2,311 | 179 | 170 | KK 10,940 | 96s -992 |
| suite | mlp-s-seed2 | call | 100 | 766 | 216 | 145 | TT 2,127 | Q6s -408 |
| suite | mlp-s-seed2 | lag | 100 | 486 | -175 | -185 | AA 1,835 | QJs -857 |
| suite | mlp-s-seed2 | potodds | 100 | 58 | -2 | -23 | AA 330 | AQs -306 |
| suite | mlp-s-seed2 | random | 100 | 3,987 | 188 | 96 | KK 11,880 | 97s -1,229 |
| suite | mlp-s-seed2 | tag | 100 | 69 | -157 | -151 | AA 980 | QJs -477 |
| suite | mlp-s-seed2 | tight | 100 | 255 | 2 | -35 | AA 1,085 | AQs -407 |
| suite | mlp-s-seed3 | bluff | 100 | 3,108 | 542 | 456 | KK 10,243 | 73s -626 |
| suite | mlp-s-seed3 | call | 100 | 1,281 | 373 | 286 | KK 7,222 | Q8s -305 |
| suite | mlp-s-seed3 | lag | 100 | 492 | -178 | -229 | AA 2,374 | A3s -974 |
| suite | mlp-s-seed3 | potodds | 100 | 88 | 25 | 17 | AA 257 | 72s -98 |
| suite | mlp-s-seed3 | random | 100 | 3,795 | 217 | 191 | KK 11,062 | Q7s -1,123 |
| suite | mlp-s-seed3 | tag | 100 | 63 | -140 | -153 | AA 1,113 | Q6s -741 |
| suite | mlp-s-seed3 | tight | 100 | 157 | -3 | -4 | AA 976 | Q6s -694 |
| suite | random-bot | bluff | 100 | 2,791 | -253 | -1,034 | AA 7,356 | 62o -3,728 |
| suite | random-bot | call | 100 | 2,601 | 226 | -507 | KK 7,657 | 32s -2,810 |
| suite | random-bot | lag | 100 | 479 | -716 | -1,060 | AA 2,326 | 62o -2,445 |
| suite | random-bot | potodds | 100 | -90 | -226 | -286 | AA 516 | J2o -786 |
| suite | random-bot | random | 100 | 2,944 | 328 | -317 | AA 6,878 | 52o -3,013 |
| suite | random-bot | tag | 100 | -132 | -508 | -617 | J9s 693 | 63s -1,390 |
| suite | random-bot | tight | 100 | 75 | -623 | -759 | AA 1,484 | 62o -1,706 |
