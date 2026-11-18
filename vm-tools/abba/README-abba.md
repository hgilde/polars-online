# abba.py: shared ABBA timing for builds of polars-online

Python 3.12, stdlib + polars + numpy. Run it with any interpreter; it starts the
builds' own pythons as subprocesses. Each case is generated (seeded) inside the
worker, so there are no data files.

## Run (one layout)

    python vm-tools/abba/abba.py run \
      --build A=/path/venvA --build B=/path/venvB --control A2=/path/venvA2 \
      --cases bocpd_diag_k4_g100,corrchange_window_w100 --rounds 9 --reps 3 \
      --max-load 4 --out layout1.json

A path is a venv directory, a python executable, or a `.whl` (then a venv with
the wheel and numpy is made under `--workdir`, default `.abba-venvs`). The first
`--build` is the baseline A. The control is a second copy of A's wheel in its own
venv. `--cases all` runs everything. Other options: `--stat min|median` (of the
reps inside a process), `--positions mean|min` (a build runs twice per round,
order A B A2 A2 B A; its two figures are combined, mean cancels drift), `--scale`
(multiplies every case's rows), `--floor` (extra 0.5% each side of the control
band), `--max-load N` (wait, up to `--max-wait` s per round, for the 1-minute load
to fall under N; load is NaN where `os.getloadavg` does not exist), `--env K=V`
(for the workers, e.g. `POLARS_MAX_THREADS=2`), `--warmup`.
Each (case, build, round, position) is a fresh process. `abba.py once CASE` times a
case once on the current interpreter, for sizing.

## Two layouts

Run once per code layout (default and aligned builds), then:

    python vm-tools/abba/abba.py combine layout1.json layout2.json

A case is SLOWER only if it is SLOWER in both; slower in one is reported as layout.

## Output

A text table, and `--out` JSON (per case: rows, per-round seconds per build,
ratios, medians, verdicts; run: per-round loads, imported package files/versions,
args). Per case and build: B/A median (above 1 = slower) with min..max over
rounds, the control A2/A median and range, and a verdict: SLOWER / FASTER when
B's median is outside the control's range (widened to include 1 and by `--floor`)
and every round is on that side of 1, otherwise "within control". Check the printed
package files/versions in the JSON (`files`) to be sure the builds differ.

## Cases (`abba.py list`)

window_ewm1_g100, window_ewm4_g100 (irregular datetime clock, 100 keys, 5s half-life,
60s window; 1 and 4 operators); bocpd_diag_k{1,2,4,8,20,200}_g{1,100};
bocpd_full_k4_chunked, bocpd_full_k20_chunked (fit_predict_batches);
corrchange_window_w{20,100,1000,2500}, corrchange_monitor_w2500; hmm_d4_k2;
ew_cov_g1, ew_cov_g100, ew_cov_g1000, ew_cov_lags_g1, ew_cov_lags_g100; seqtest;
ewridge_k5_g1000; kmeans_k4, kmeans_k20; save_400k_g20k, load_400k_g20k;
parallel_eight_one_bank.

Sizes are tuned to take 0.04-0.4 s on an M-series Mac (about 2-3x that on a 2 GHz
Xeon); the cheap g100 variants are short, so raise `--scale` for them. The bocpd
data is stationary, so k=1 at one group is dominated by long run lengths.
Add a case by adding a factory to `CASES` returning `(callable, rows)`.
