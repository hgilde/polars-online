<!-- kind: record -->
# Earlier figures of the regime measurements

*Status, 2026-11-08: a record. The figures `docs/REGIMES.md` first reported for its sections 1 to 5, from builds before a change that moved them, moved here from that page's dated notes, with the earlier run times of its experiments. Each reads as of its date. `docs/REGIMES.md` gives the figures of the current build, and says when each section was run.*

## 1. Does `hmm` recover the stream that made it?

**Measured 2026-09-06:** the seeded row read 0.559, or 56 % (worst seed
0.507), with the learned state means at 0.65. The other three rows read
as they do now.

## 2. Is the monitor the size its paper says?

**Measured 2026-09-27, before the kernel moved to WKD's `1 − l/γ` (it
was Newey–West's `1 − l/(γ+1)`), at 20,000 replications a cell.** In
the table's row order, the shared-scale `t_5` read 0.040,
0.031, 0.040, 0.035, 0.033 and 0.038; the independent `t_5` 0.042, 0.042,
0.040, 0.041, 0.042 and 0.039; Gaussian pairs 0.045, 0.041, 0.046, 0.046,
0.042 and 0.046. No cell moved by more than 0.001.

**Measured 2026-09-23, at 2000 replications a cell.** In the table's row
order, the shared-scale `t_5` read 0.045, 0.024, 0.043, 0.033, 0.035 and
0.044; the independent `t_5` 0.046, 0.044, 0.041, 0.036, 0.041 and 0.041;
Gaussian pairs 0.046, 0.045, 0.052, 0.043, 0.034 and 0.048. The shared
scale's 0.024 at `T = 500` and `rho = 0`, 2.2 standard errors from WKD's
0.035, reads 0.031 at ten times the draws.

**Measured 2026-09-06, before `D-hat`'s gradient was corrected on
2026-09-19.** In the table's row order, the shared-scale `t_5` read 0.075, 0.025,
0.076, 0.071, 0.040 and 0.081. The independent `t_5` read 0.028, 0.046,
0.025, 0.018, 0.040 and 0.021. Gaussian pairs read 0.048, 0.045, 0.054,
0.042, 0.034 and 0.051.

## 3. And is it the power its paper says?

**Measured 2026-09-27, before the kernel moved to WKD's.** In the
table's row order, as power, size-adjusted and empirical 95 %: at `T = 500`, t5 read
0.553, 0.582 and 1.321, t5_indep 0.585, 0.613 and 1.327, and normal 0.830,
0.851 and 1.327. At `T = 1000`, t5 read 0.832, 0.844 and 1.324, t5_indep
0.832, 0.857 and 1.307, and normal 0.994, 0.994 and 1.355.

**Measured 2026-09-06, before `D-hat`'s gradient was corrected on
2026-09-19.** In the table's row order, as power, size-adjusted
and empirical 95 %: at `T = 500`, t5 read 0.525, 0.428 and 1.498,
t5_indep 0.400, 0.472 and 1.256, and normal 0.824, 0.837 and 1.335. At
`T = 1000`, t5 read 0.779, 0.733 and 1.452, t5_indep 0.648, 0.750 and
1.195, and normal 0.994, 0.994 and 1.358.

## 4. Where the heavy tails go: `D-hat`

**Measured 2026-09-27, before the kernel moved to WKD's.** In the
table's row order, as mean, median and sd, normal read 0.739, 0.742 and 0.060, then
0.749, 0.751 and 0.033. The shared-scale t5 read 1.071, 1.020 and 0.257, then
1.184, 1.097 and 0.353; the independent t5 0.811, 0.796 and 0.113, then
0.828, 0.815 and 0.073.

**Measured 2026-09-06, before `D-hat`'s gradient was corrected on
2026-09-19.** In the table's row order, as mean, median and sd:
normal read 0.739, 0.739 and 0.063, then 0.750, 0.750 and 0.034. The
shared-scale t5 read 1.050, 0.979 and 0.446, then 1.143, 1.032 and 0.464.
The independent t5 read 0.932, 0.877 and 0.216, then 0.953, 0.915 and
0.156. The shared-scale `t_5` then read about 15 % low, with an sd 40 %
of its level, and a denominator 15 % too small inflates `Q` by about
18 %.

## 5. Two changepoint detectors on the same break

**Measured before the window test's critical value moved to the standard
permutation rule on 2026-10-26:** 53.67 false alarms per 1000 quiet rows,
a median delay of 25 rows and 56 at worst. With 100 shuffles at `alpha`
0.05, the rule moved from the 95th smallest shuffle to the 96th, and the
size from 5.94% to 4.95%.

## 10. Running them

Earlier runs took 40.3 s on 2026-09-23, with 2000 draws a cell, and about 36 seconds on 2026-09-06, when the page `docs/REGIMES.md` was first written. Every figure the 2026-09-23 run gave that the runs of 2026-09-27 and 2026-09-28 re-ran is unchanged.
