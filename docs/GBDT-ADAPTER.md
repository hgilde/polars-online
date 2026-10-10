# Training LightGBM and XGBoost from Arrow files, as fast as they go

**Status as of 2026-10-10: research for PLAN task 235, measured, with
nothing built in the package and the decision left to the user.** The user stated
the goal the same day: "read from the arrow files and maximize training
speed by dynamically constructing inputs as needed". Five ways to do that,
options A to E, were built as prototypes outside the package and timed
against each other and against each library's usual path from Polars. The
data were generated, from 10 million rows to 100 million, and the larger
shapes do not fit in memory. The libraries were read at their tagged
sources, XGBoost 3.4.1 and LightGBM 4.7.0. Upkeep is not weighed here (the
user: "We are not worried about upkeep"). [BOOSTED-TREES.md](BOOSTED-TREES.md)
is related work of another kind: an online tree model inside this library.

---

## 1. The answer

**For XGBoost, the fastest design measured is Rust reading the Arrow files
and driving XGBoost's C API, with the bins placed from a 200,000-row sample
and handed over through `ref` (options D and C together).** It reached the
first boosting round 2.8 times sooner than Polars batches in XGBoost's own
iterator on 100 million rows, and 2.1 times sooner on 10 million, with
trees within the accuracy that the choice of sample already varies by.
**For LightGBM, the fastest is the whole Arrow frame when it fits, and Rust
pushing float32 rows (option D) when it does not**, 18% sooner than the
Python `Sequence`. The time before the first round, in seconds, at equal
accuracy (§4.5):

| library | route | option | 1e7 × 50 | 2e6 × 500 | 1e8 × 50, larger than memory | peak memory at 1e8 × 50, GB |
|---|---|---|---:|---:|---:|---:|
| XGBoost | Rust, Arrow columns, cuts from a sample | D + C | **4.3** | **9.5** | **52.4** | 31.2 |
| XGBoost | Python iterator, Polars batches, cuts from a sample | C | 4.8 | 15.8 | 64.0 | 35.2 |
| XGBoost | `collect()`, numpy | baseline | 7.2 | 14.0 | does not fit | |
| XGBoost | Rust, Arrow columns, XGBoost's own sketch | D | 8.2 | 16.7 | 104.7 | 31.2 |
| XGBoost | Python iterator, Polars batches | B | 8.9 | 22.8 | 149.0 | 32.8 |
| LightGBM | `collect()`, the Polars frame | baseline | **1.6** | **3.9** | does not fit | |
| LightGBM | Rust, float32 rows | D | 1.9 | 5.9 | **21.9** | 9.5 |
| LightGBM | Rust, float64 rows, LightGBM's own trees | D | 2.2 | 6.2 | 23.6 | 9.5 |
| LightGBM | Python `Sequence`, float64 rows | A | 2.2 | 6.8 | 26.8 | 12.5 |

**XGBoost spends about half its time before boosting on a sketch that a
sample replaces.** It reads the data four times, and the second pass sketches
every row to place the bins (§3.1). Cuts from a sample, handed over through
`ref`, remove that pass, and the trees they grow score within the spread
of LightGBM's own sampling (§4.5). This library's t-digest places the same
cuts more slowly than a sample, so the cuts should come from a sample.

**Once the batches arrive without a copy, the libraries' own passes set
the pace.** Numpy, Polars and pyarrow batches and Rust's Arrow columns took
the same time to within 14% on 1e7 × 50, because XGBoost's passes and
LightGBM's pushes do the work. Rust pulled ahead where Python's handover
grew: 30% on 1e8 rows, and 27% at 500 features. It used half the memory of
Python on 1e7 × 50, where the binned data does not dominate.

**The input route changes only the time before the first round.** On 1e8 ×
50, a boosting round took 3.5 seconds for XGBoost and 3 to 4 for
LightGBM, whatever the route. A model of 500 rounds spends half an hour
boosting, and the fastest route saves 97 seconds of XGBoost's time before
it. Beyond memory, XGBoost's external-memory matrix bounds the footprint,
and boosting then takes 3 to 40 times as long. LightGBM has no such mode
(§5).

## 2. The options

Both libraries train in two stages. **Binning** turns each feature's values
into at most 256 bins, and **boosting** grows each tree from per-bin sums
of gradients. Boosting never reads the raw values again, so an input route
changes only what happens before it. That is four phases, which the
measurements time apart: reading the files, converting each batch into what
the library takes, placing the bins, and quantizing every value into its
bin. XGBoost places its bins by sketching each feature's quantiles over
every row, and LightGBM from a sample.

| option | what it does | prototype routes (`scripts/gbdt_adapter.py`) |
|---|---|---|
| baseline | the library's usual path: `collect()`, then numpy, pandas or the Polars frame | `xgb-numpy`, `xgb-polars`, `lgb-numpy`, `lgb-polars` |
| A. A Python iterator over Arrow batches, numpy per batch | `collect_batches` hands batches to XGBoost's `DataIter` and to a LightGBM `Sequence`, each batch converted to numpy | `xgb-iter-numpy`, `lgb-stream` |
| B. Arrow handed over natively, per batch | each batch a Polars frame or a pyarrow `Table`, which XGBoost reads column by column in its C++ | `xgb-iter`, `xgb-iter-pa` |
| C. Bins computed here | one pass computes each feature's cuts (a t-digest in Rust, or a sample); XGBoost takes them through `ref`, LightGBM through forced bins, or the batches arrive already quantized to bin codes | `xgb-iter-ref`, `rust-xgb-columnar-sample`, `rust-xgb-columnar-cuts`, `rust-xgb-codes`, `rust-lgb-forced`, `rust-lgb-codes` |
| D. Rust driving the C APIs, loaded at run time | a Rust program reads the Arrow IPC files, converts each batch in parallel, and calls XGBoost's and LightGBM's C functions from the shared libraries their wheels ship (`dlopen`; nothing linked) | `rust-xgb-dense`, `rust-xgb-columnar`, `rust-lgb-f64`, `rust-lgb-f32` |
| E. Larger than memory | XGBoost's external-memory matrix, which caches the binned pages on disk; a spill of the batches to Arrow IPC, read back for each pass | `xgb-extmem`, `xgb-replay-spill`, `lgb-replay-spill` |

Options combine: D with C is Rust placing the cuts and handing XGBoost its
Arrow columns (`rust-xgb-columnar-sample`) or codes (`rust-xgb-codes`), and
C with B is the Python iterator over Polars batches with cuts from a sample
(`xgb-iter-ref`). The Rust driver is
[`scripts/gbdt_driver`](../scripts/gbdt_driver), a crate of its own that
the repository's build does not touch.

## 3. What the libraries take

Each claim cites the file it was read from, in `xgboost` at tag `v3.4.1` and
`LightGBM` at tag `v4.7.0`.

### 3.1 XGBoost 3.4.1

**A `DataIter` is pulled four times, and the sketch pass is the one that
can go.** `IterativeDMatrix::InitFromCPU` (`src/data/iterative_dmatrix.cc:60`)
runs `GetDataShape` over every batch, to count each column's values ("CPU
impl needs an additional loop for accumulating the column size",
`src/data/quantile_dmatrix.cc:121`). It then runs `MakeSketches` over every
batch, unless a `ref` matrix supplies the cuts (`quantile_dmatrix.cc:149`),
then pushes every batch into the bin index, and then every batch again into
the column matrix. A counting iterator saw 16 calls of `next` over 4
batches, and 12 with `ref`. The raw values are not kept. On 1e7 × 500, the
four passes of one run took 33, 89, 41 and 66 seconds (at a load average of
28), so the sketch was the largest of them.

**Any cuts can be handed over through `ref`.** The `ref` argument takes the
cuts from another matrix, which is how a validation set shares the training
set's bins. Build the `ref` from a small matrix whose column holds a
feature's wanted cuts and one value below the lowest. Its cuts then come
back as exactly those values, in float32, and the lowest value becomes the
feature's minimum (checked on 4 and on 40 values). The C API takes the same `ref`
(`XGQuantileDMatrixCreateFromCallback`, `include/xgboost/c_api.h:528`). The
sketch's own target is a rank error of `1 / max_bin`, a 256th of the rows
(`src/common/quantile.h:740`).

**Arrow is read per batch, column by column.** A batch handed to the
`DataIter` as a Polars frame goes through `df.to_arrow()`, which needs
pyarrow (`python-package/xgboost/data.py:927`), and as a pyarrow `Table` it
goes straight on (`data.py:801`). Each column is `combine_chunks()`ed and
handed to the C++ as an array interface on its own buffers
(`_data_utils.py:367`), the validity bitmap as its `mask`. The reader
returns NaN for a null and casts each value to float32
(`src/data/adapter.h:404`). A column of several Arrow chunks is
concatenated first, and every batch read here spanned two row groups, so
every column of every batch was copied once.

**The bin index takes four bytes a value once any value is missing.** A
matrix with no missing value stores each bin in one byte, and any other
matrix stores it in four (`src/data/gradient_index.cc:197`). Building a
`QuantileDMatrix` from a 2e6 × 50 numpy array added 2.6 bytes a value to
the footprint with no NaN, and 5.5 with 1% NaN. The data here has nulls,
which is why XGBoost's peak on 1e8 × 50 was 33 GB in every route that kept
the matrix in memory.

### 3.2 LightGBM 4.7.0

**LightGBM bins from a sample, then pushes every value.** Its bins come
from `bin_construct_sample_cnt` rows, 200,000 by default, drawn by
`LGBM_SampleIndices` and seeded by `data_random_seed`
(`docs/Parameters.rst:828`). The full pass is the push of every value into
its bin, which needs the bins first.

**Arrow goes in whole, never per batch.** A Polars frame reaches
`LGBM_DatasetCreateFromArrowStream` through narwhals and Polars' own
`__arrow_c_stream__`, with no copy and no pyarrow
(`python-package/lightgbm/basic.py:2171`, `basic.py:2439`). The C++
consumes the stream to its end and keeps every chunk
(`src/arrow/array.hpp:60`), then pushes each column in parallel
(`src/c_api.cpp:1651`). The streaming interfaces take rows only:
`LGBM_DatasetPushRows` reads a dense row-major array (`c_api.cpp:1140`),
and a `Sequence` must return numpy. So a per-batch Arrow route does not
exist for LightGBM, and every streamed batch is transposed to rows first.

**A `Sequence` needs random access, which a query does not offer.**
`__init_from_seqs` (`basic.py:2252`) asks for each of the 200,000 sampled
rows by index, through `__sample` (`basic.py:2224`), builds the bins with
`LGBM_DatasetCreateFromSampledColumn`, and then pushes `seq[start:end]`
batch by batch. LightGBM's guide states the design: "Data sampling uses
random access, thus does not go through the whole dataset"
(`docs/Python-Intro.rst:101`). The prototypes ask LightGBM for its sample
indices, gather those rows in a first pass, and answer the single-row
requests from them, so they build the same `Dataset` as numpy does.

**Forced bins take at most `max_bin − 3` bounds.** `forcedbins_filename`
names a JSON file of bin upper bounds per feature (`docs/Parameters.rst:966`).
`FindBinWithPredefinedBin` places bounds at −0, +0 and infinity first, then
the forced bounds in the order given while fewer than `max_bin` are placed,
and silently drops the rest (`src/io/bin.cpp:162`). With `max_bin = 255`,
the 253rd and 254th bounds of 254 were dropped, so each feature's highest
values shared one bin, and validation RMSE rose 1.8% in a 5-round check.
Handed 252 bounds, it scored within the spread of LightGBM's own seeds
(§4.5).

**The binned dataset stays in memory, and nothing in LightGBM pages it
out.** Its only out-of-memory setting, `two_round`, is for loading a text
file (`docs/Parameters.rst:876`). Bagging and GOSS sample rows per iteration
from a dataset already built (`docs/Parameters.rst:191`, `:362`). A feature
of at most 16 bins is stored in four bits, any other in a byte or more
(`src/io/bin.cpp:618`). On the data here, building a dataset of 1e8 × 50
peaked at 9.5 GB in the Rust driver and 12.5 GB in Python.

### 3.3 Three sharp edges

| what | what happens | what to do |
|---|---|---|
| a `pl.Enum`, `pl.Categorical` or string column handed to LightGBM 4.7.0 | the process ends on SIGABRT, from an uncaught C++ exception ("Unsupported Arrow type: dictionary", `array.hpp:243`), with no Python exception to catch | pass the integer codes, `pl.col(c).to_physical()`, and name the column in `categorical_feature` |
| `seed` set without `data_random_seed`, in LightGBM | a `Sequence` and a numpy array sample different rows, and so build different bins and trees. The Python side drops `seed` when it samples a `Sequence`, because `Dataset.get_params` keeps only dataset parameters (`basic.py:1990`). The C++ side derives the sample's seed from `seed` (`src/io/config.cpp:266`) | set `data_random_seed` too: with it, the two built identical trees here |
| a Polars frame handed to XGBoost without pyarrow installed | ``ImportError: `pyarrow` is required for polars.`` (`data.py:910`) | install pyarrow, or hand numpy batches, at a copy of each batch |

## 4. The measurements

**How they were taken.** Every route ran in a fresh process, with 8
threads for the library, for Polars and for Rust, on an Apple M4 Pro (14
cores, 48 GB). Other jobs shared the machine, and its load average rose
from 5 to over 40 during the day. So the comparisons that matter were run
interleaved, Python and Rust in turn, up to three times, and each table
gives the median with the load it ran under. Memory is the peak physical footprint
(what Activity Monitor shows as Memory). It counts what a process
allocated, and leaves out clean pages of files, which the system can drop
at will.

**The data.** `scripts/gbdt_adapter.py generate` writes seeded parquet
parts, and `to-ipc` writes the same rows as uncompressed Arrow IPC files,
which both Polars and the Rust driver read. Each row holds Float64 features
of four kinds (normal, lognormal, uniform, and a normal rounded to 0.1),
with 5% nulls in every fifth column. Each row also holds one `pl.Enum` of
32 levels drawn unevenly, and a target built from a sine, a product, a threshold and
the category, plus noise. A separate `valid.parquet`, from another seed,
scores every route.

| shape | rows | features | parquet | Arrow IPC | fits in 48 GB with a route's copies |
|---|---:|---:|---:|---:|---|
| 1e7 × 50 | 10,000,000 | 50 and the Enum | 3.5 GB | 4.1 GB | yes |
| 2e6 × 500 | 2,000,000 | 500 and the Enum | 6.7 GB | 8.0 GB | yes |
| 1e8 × 50 | 100,000,000 | 50 and the Enum | 32.0 GB | 41.0 GB | no |
| 1e7 × 500 | 10,000,000 | 500 and the Enum | 31.4 GB | — | no |

**How accuracy was held equal.** Each route's trees were hashed, from
XGBoost's JSON dump and LightGBM's `dump_model()`. Two routes that place the
same bins grow the same trees, and the hashes say whether they did. A route
that places its bins from other cuts grows other trees, and was held to the
spread of the validation RMSE over five seeds of a sample (§4.5).

### 4.1 XGBoost, 1e7 × 50

**Cuts handed over through `ref` halve the time before boosting, and the
reader hardly matters.** The time is everything before the first round:
reading, converting, and XGBoost's passes. Each route's passes are given
as XGBoost makes them: counting, sketching (absent with `ref`), indexing,
and the column matrix. Median of three interleaved runs, at a load of 10 to
16 (no run was more than 10% from its route's median):

| option | route | time before boosting, s | passes, s | cuts placed, s | peak memory, GB | trees |
|---|---|---:|---|---:|---:|---|
| baseline | `collect()`, numpy, one batch | 7.2 | | | 8.8 | X |
| A | `DataIter`, numpy batches | 8.9 | 1.0, 4.4, 1.8, 1.7 | | 5.5 | X |
| B | `DataIter`, Polars batches | 8.9 | 0.8, 4.6, 1.7, 1.8 | | 5.2 | X |
| B | `DataIter`, pyarrow batches from `pyarrow.dataset` | 8.9 | 0.8, 4.6, 1.7, 1.8 | | 7.2 | X |
| D | Rust, Arrow columns | 8.2 | 0.7, 4.3, 1.5, 1.6 | | 2.8 | X |
| D | Rust, dense float32 rows | 7.8 | 0.8, 4.1, 1.7, 1.3 | | 2.7 | X |
| C | `DataIter`, Polars batches, cuts from 40 slices | 4.8 | 0.8, 1.7, 1.9 | 0.43 | 7.3 | sample |
| D + C | Rust, Arrow columns, cuts from a 200,000-row sample | **4.3** | 0.7, 1.5, 1.6 | 0.46 | 2.8 | sample |
| D + C | Rust, Arrow columns, cuts from the t-digest | 6.9 | 0.7, 1.6, 1.6 | 2.98 | 2.8 | digest |
| D + C | Rust, one-byte codes from the t-digest's cuts | 7.3 | 0.9, 1.6, 1.8 | 2.97 | 2.9 | digest |

X is one hash: the routes that let XGBoost sketch grew the same trees. The
same order held in a first interleaved sweep at a load of 20 to 32, where
every time was 20 to 50% longer.

**Option C's gain is the sketch pass, and option D's is memory.** Passing
`ref` removes XGBoost's second pass, the largest of the four, and the other
three run as before. A 200,000-row sample places the cuts in under half a
second. The t-digest, at the compression of 1,000 that 254 cuts need,
takes 3.0 seconds over 500 million values on 8 threads, nearly the time
the sketch pass saves. Feeding one-byte codes saves nothing more, because
XGBoost still searches each code's bin among the cuts. Whatever hands
XGBoost its batches, its passes take the same time, so the readers are
within 14% of each other. Rust needs half the memory of Python, because no
Polars or Python runs beside the library.

### 4.2 LightGBM, 1e7 × 50

**For LightGBM, every route takes about the same time, and the routes
differ in memory.** LightGBM's work is the push of every value into its
bin, the same in every route. Median of three interleaved runs, at a load
of 13 to 16:

| option | route | time before boosting, s | peak memory, GB | trees |
|---|---|---:|---:|---|
| baseline | `collect()`, the Polars frame (Arrow, column by column) | **1.6** | 9.0 | L |
| A | `Sequence`, LightGBM's sample gathered in a first pass, batches pushed as float64 rows in a second | 2.2 | 2.8 | L |
| D | Rust, the same two passes, rows built in parallel, float64 | 2.2 | 1.1 | L |
| D | Rust, the same, float32 | 1.9 | 1.2 | L at one round; moved at 50 |
| D + C | Rust, the t-digest's cuts as 252 forced bins | 5.1 | 1.1 | digest |
| D + C | Rust, rows of one-byte codes from the t-digest's cuts | 5.0 | 1.2 | digest |

L is one hash. The last two rows are single runs of 50 rounds, at a load
of 18 to 21, and each includes its 3.0-second t-digest pass: LightGBM's own
sample places its bins faster than any sketch here.

### 4.3 Larger than memory, 1e8 × 50

**At 100 million rows the ranking holds, and the gaps widen.** These are
single runs, in the same interleaved order, at a load of 13 to 20, from the
41 GB of Arrow IPC files. One pass of Polars over the files with nothing
consuming the batches took 12.7 seconds, longer than the 6.8 of the parquet
files, because 41 GB of uncompressed Arrow does not stay in the page cache.

| library | option | route | time before boosting, s | passes, s | peak memory, GB | trees |
|---|---|---|---:|---|---:|---|
| XGBoost | B | `DataIter`, Polars batches | 149.0 | 16, 82, 26, 24 | 32.8 | X |
| XGBoost | D | Rust, Arrow columns | 104.7 | 10, 52, 17, 25 | 31.2 | X |
| XGBoost | C | `DataIter`, cuts from 40 slices | 64.0 | 2 (slices), 13, 22, 27 | 35.2 | sample |
| XGBoost | D + C | Rust, Arrow columns, cuts from a 200,000-row sample | **52.4** | 7 (sample), 8, 16, 21 | 31.2 | sample |
| XGBoost | D + C | Rust, Arrow columns, cuts from the t-digest | 75.6 | 30 (digest), 8, 16, 22 | 31.2 | digest |
| XGBoost | D + C | Rust, one-byte codes | 77.5 | 29 (digest), 8, 17, 23 | 31.2 | digest |
| LightGBM | A | `Sequence`, float64 rows | 26.8 | 8, 19 | 12.5 | L |
| LightGBM | D | Rust, float64 rows | 23.6 | 7, 17 | 9.5 | L |
| LightGBM | D | Rust, float32 rows | **21.9** | 8, 14 | 9.5 | moved |

The Rust sample read every batch of the files once, 7.3 seconds. The
Python sample read 40 slices of them, 2.2 seconds. A Rust reader that
seeks to a few record batches would read as little, and was not built.

### 4.4 Wide, 2e6 × 500

**At 500 features the converted routes slow down, and option D's lead over
Python grows.** XGBoost reads a Polars or pyarrow batch column by column,
and at 500 columns a batch's handover took 3.6 seconds over four passes,
against 0.16 for Rust's. Median of two interleaved runs at a load of 11 to
19. The sample-cut route and the baseline ran in a second sweep at a load
of 5 to 10, beside a third and fourth run of Rust's columns (16.3 seconds
there, against 16.7):

| library | option | route | time before boosting, s | passes, s | peak memory, GB | trees |
|---|---|---|---:|---|---:|---|
| XGBoost | baseline | `collect()`, numpy | 14.0 | | 17.5 | X |
| XGBoost | B | `DataIter`, Polars batches | 22.8 | 2.4, 9.6, 5.0, 5.8 | 8.2 | X |
| XGBoost | A | `DataIter`, numpy batches | 19.0 | 2.1, 8.2, 4.6, 4.1 | 8.5 | X |
| XGBoost | D | Rust, Arrow columns | 16.7 | 1.3, 7.8, 3.8, 3.8 | 5.4 | X |
| XGBoost | D | Rust, dense float32 rows | 16.2 | 1.6, 8.0, 3.9, 2.8 | 5.3 | X |
| XGBoost | C | `DataIter`, cuts from 40 slices | 15.8 | 2.4 (slices), 2.4, 5.2, 5.9 | 12.0 | sample |
| XGBoost | D + C | Rust, Arrow columns, cuts from a 200,000-row sample | **9.5** | 1.1 (sample), 1.2, 3.4, 3.8 | 5.4 | sample |
| XGBoost | D + C | Rust, Arrow columns, cuts from the t-digest | 15.2 | 5.6 (digest), 1.4, 4.2, 4.0 | 5.4 | digest |
| LightGBM | baseline | `collect()`, the Polars frame | **3.9** | | 19.7 | L |
| LightGBM | A | `Sequence`, float64 rows | 6.8 | 2.5, 4.3 | 5.7 | L |
| LightGBM | D | Rust, float64 rows | 6.2 | | 2.3 | L |
| LightGBM | D | Rust, float32 rows | 5.9 | | 2.4 | L at one round |

LightGBM's in-memory Arrow path pushes columns in parallel, and wins by
more at this width: its streaming paths must build rows of 500 values.

### 4.5 Accuracy

**A route that lets the library place its bins grows the library's own
trees, to the bit.** On 1e7 × 50 at 50 rounds, every XGBoost route that
sketched for itself grew one set of trees. They were numpy, pandas, the
Polars frame, Polars, numpy and pyarrow batches, the external-memory
matrix, the replays, and the Rust driver's dense and columnar feeds. For
LightGBM, numpy, pandas, the Polars frame, the `Sequence` and the Rust
driver's float64 rows did too. LightGBM's bins are
float64, so float32 rows moved some trees: on 1e7 × 50 the validation RMSE
was the same to six places, 1.231957 in both.

**Other cuts grow other trees, within the spread that LightGBM's own
sampling already has.** The validation RMSE on 1e7 × 50 at 50 rounds:

| how the bins were placed | validation RMSE |
|---|---|
| XGBoost's own sketch | 1.2379 |
| XGBoost, cuts from a uniform sample, five seeds | 1.2347 to 1.2411 |
| XGBoost, cuts from 40 slices, five seeds | 1.2351 to 1.2414 |
| XGBoost, cuts from the Rust driver's sample | 1.2315 |
| XGBoost, the t-digest at compression 1,000 (and its codes) | 1.2389 |
| XGBoost, the t-digest at the library's own 200 | 1.2260 |
| LightGBM's own sample, five seeds of `data_random_seed` | 1.2298 to 1.2443 |
| LightGBM, the t-digest's cuts as 252 forced bins | 1.2421 |
| LightGBM, codes from the t-digest's cuts | 1.2375 |

On 2e6 × 500, XGBoost's own sketch gave 1.2234, cuts from five samples
1.2224 to 1.2345, and LightGBM's five seeds 1.2227 to 1.2330. The cuts a
route chooses move the RMSE by no more than LightGBM's choice of seed
does. The one value outside that range is the t-digest at 200, which
scored lower than all the rest, a single run that no second one checked.

## 5. Larger than memory

**XGBoost's binned matrix is what stops fitting, and its external-memory
form trades boosting speed for it.** The input never has to fit: every
streaming route holds a batch or two of it. What a route keeps is the
library's binned matrix. XGBoost's, at four bytes a bin once any value is
missing (§3.1), peaked at 33 GB on 1e8 × 50 and 40 GB on 1e7 × 500, on a
48 GB machine. `ExtMemQuantileDMatrix` caches the binned pages on disk and
reads them back in each round, so it bounds the footprint and slows
boosting:

| shape | in memory: peak, GB | boost, s | external memory: peak, GB | boost, s | slower by |
|---|---:|---:|---:|---:|---:|
| 1e7 × 50, 50 rounds | 4.9 | 17.1 | 2.2 | 50.8 | 3.0 times |
| 2e6 × 500, 50 rounds | 12.7 | 25.9 | 8.4 | 95.0 | 3.7 times |
| 1e8 × 50, 20 rounds | 33.1 | 70.1 | 6.3 | 255.0 | 3.6 times |
| 1e7 × 500, 20 rounds | 39.9 | 51.2 | 14.9 | 2046.0 | 40 times |

The run on 1e7 × 500 used 7.8 cores while it boosted and read 8 MB/s from
disk, so the time went to computing over the pages, and the cache sat in
the page cache. Its trees were the in-memory route's, every time.

**LightGBM has no external memory.** Building its dataset of 1e8 × 50
peaked at 9.5 GB in Rust and 12.5 GB in Python, two to two and a half bytes
a value with the process around it. Past
memory, the settings that shrink it change the model: `max_bin` of 15 or
fewer stores each bin in four bits (§3.2), and fewer rows is a sample.
Bagging and GOSS sample rows from a dataset already built, so neither
lowers its footprint.

**A spill of the batches is faster only when a pass of the query is
slow.** A spill reads the query once and writes its batches to
uncompressed Arrow IPC, so that every later pass reads the spill. On these
parquet files, re-running the query was the faster of the two: on 1e8 × 50,
XGBoost's four passes took 105 seconds re-running it and 160 reading a
41 GB spill, and LightGBM's two took 26 and 41. A query that joins,
filters or computes features makes each pass slower, and the spill's
single read worth more.

## 6. The recommendation

**Build options D and C together for XGBoost, and option D for LightGBM,
as one Rust component of the package that loads the libraries at run
time.** For XGBoost it was the fastest at every shape measured, 32% to 65%
sooner than the best Python route without cuts (numpy in memory, or
batches beyond it). For LightGBM, Rust's rows were the fastest beyond
memory, 13% to 18% sooner than the `Sequence`, and the whole Arrow frame
stays the fastest when it fits. Rust needs half the memory of Python where
the binned data does not dominate. The Python form of option C, a
`DataIter` over Polars batches with cuts from a sample, is the fallback. It
needs no Rust and about 50 lines, and was 12% to 66% slower than the Rust
form (4.8 against 4.3 seconds, 64.0 against 52.4, 15.8 against 9.5).

**What the component does**, as `scripts/gbdt_driver` does it:

| step | XGBoost | LightGBM |
|---|---|---|
| read | Arrow IPC files, on a reader thread that keeps four batches ahead, as measured; or a query's batches through `collect_batches`, which the package already hands to Rust, and which the driver did not try | the same |
| place the bins | a uniform sample of 200,000 rows (measured: rows kept by a hash of their index, over one full pass; random record batches of a file would read less) and exact quantiles of it, built into a `ref` matrix | LightGBM's own: `LGBM_SampleIndices`, those rows gathered in the first pass, `LGBM_DatasetCreateFromSampledColumn` |
| hand over | each batch's Arrow columns as array interfaces, validity bitmaps as masks, no copy, into `XGQuantileDMatrixCreateFromCallback`; XGBoost then makes three passes | each batch built into float32 rows in parallel (float64 for LightGBM's own trees, 5% to 16% slower), pushed with `LGBM_DatasetPushRows` |
| train | the boosting loop through the C API, or the matrix handed to Python | the same |

**How it reaches Python is the one open choice of shape.** The prototype
trains through the C API and saves the model, which `xgb.Booster` and
`lgb.Booster` load unchanged. That path is the measured one. It gives up
the libraries' Python training loop (callbacks, evaluation sets, early
stopping) unless the component reimplements them. The alternative hands the
built matrix or dataset back as the library's own Python object, by setting
the handle that `xgb.DMatrix` and `lgb.Dataset` keep. That keeps
`xgb.train` and `lgb.train` as they are, rests on those classes' private
attributes, and was not prototyped.

**What building it takes**, with upkeep set aside at the user's word:

| what | the cost |
|---|---|
| code | about 1,200 lines of Rust in the prototype, and the Python functions in front of it |
| linking | none. The libraries are loaded with `dlopen` from the user's installed Python packages (`xgboost/lib/libxgboost.dylib`, `lightgbm/lib/lib_lightgbm.dylib`, and their Linux and Windows names), through `libloading`, a pure-Rust crate. Nothing is linked into the wheel or the CLI, as CLAUDE.md's rule 12 requires |
| dependencies | the package's stay `polars`, and the libraries stay the user's to install. An optional extra naming them would change the assertion in `tests/test_dependency_policy.py` that `numpy` is the only extra |
| tests | xgboost and lightgbm in the dev group, held to the libraries' own trees by hash as here. XGBoost's Linux wheel is 57 MB and requires `nvidia-nccl-cu13`, and both macOS wheels need `libomp` on the CI runners |
| accuracy | the XGBoost trees differ from XGBoost's own sketch, within the spread of a sample's seed (§4.5); a user who wants XGBoost's own trees passes no `ref`, at two to three times the time before boosting |

**What is not worth building.** Cuts from this library's t-digest: a sample
places them faster and as well (§4.1). Feeding bin codes: no faster than
handing over cuts (§4.1). LightGBM's forced bins: slower than its own
sample, and they keep only 252 bounds (§3.2). A spill of the batches:
slower than re-running a parquet query (§5).

## 7. Reproducing the measurements

Neither library is a dependency of the package or of its dev group, so a
run names both, at the versions measured. The script writes the data where
it is told, under the gitignored `.cache/` here, and prints one JSON line
per route. The Rust driver builds on its own, and the script finds it
through `GBDT_DRIVER`:

```bash
W="uv run --with xgboost==3.4.1 --with lightgbm==4.7.0 python scripts/gbdt_adapter.py"
$W generate .cache/gbdt/r1e7_f50 --rows 1e7 --features 50
# the same rows as uncompressed Arrow IPC files, which the Rust driver reads
$W to-ipc .cache/gbdt/r1e7_f50 .cache/gbdt/r1e7_f50_ipc
CARGO_TARGET_DIR=.cache/gbdt/target cargo build --release --manifest-path scripts/gbdt_driver/Cargo.toml
export GBDT_DRIVER=.cache/gbdt/target/release/gbdt-driver
# each route in a fresh process, 50 rounds, 8 threads for the library, Polars and Rust
$W sweep .cache/gbdt/r1e7_f50_ipc xgb-numpy xgb-iter xgb-iter-ref rust-xgb-dense rust-xgb-codes \
    lgb-polars lgb-stream rust-lgb-f32 --rounds 50 --threads 8 --spill .cache/gbdt/spill
# the sample's seed, for the spread of a route that bins from a sample
$W sweep .cache/gbdt/r1e7_f50 xgb-replay-ref lgb-numpy --sample-seed 1 --spill .cache/gbdt/spill
```

On macOS, both libraries' wheels load `@rpath/libomp.dylib`. Homebrew's
`libomp` provides it. These measurements used the copy that scikit-learn's
wheel bundles, through `DYLD_LIBRARY_PATH=<site-packages>/sklearn/.dylibs`,
which the Rust driver needs too.

| route | what it does |
|---|---|
| `source-pass` | one pass of the query, with nothing downstream |
| `xgb-numpy`, `lgb-numpy` | `collect()`, `to_numpy()`, then the library |
| `xgb-pandas`, `lgb-pandas` | `collect()`, `to_pandas()`, then the library |
| `xgb-polars`, `lgb-polars` | `collect()`, then the frame itself |
| `xgb-iter` | a `DataIter` over `collect_batches`, each batch a Polars frame, into a `QuantileDMatrix` |
| `xgb-iter-numpy` | the same, each batch a numpy copy |
| `xgb-iter-pa` | the same over `pyarrow.dataset`, each batch a pyarrow `Table` |
| `xgb-iter-ref` | `xgb-iter` with a `ref` built from a 200,000-row sample of 40 slices |
| `xgb-extmem` | `xgb-iter` into an `ExtMemQuantileDMatrix`, its cache under `--spill` |
| `xgb-replay`, `xgb-replay-spill` | the source read once and held in memory, or spilled to Arrow IPC under `--spill`, then replayed for each pass |
| `xgb-replay-ref`, `xgb-replay-spill-ref` | the same, with a `ref` from a uniform sample drawn during the one read |
| `lgb-stream` | a `Sequence` over `collect_batches`: LightGBM's own sample rows gathered in a first pass, then the batches pushed in a second |
| `lgb-stream-ref` | the label column read alone, a 40-slice sample as a `reference`, then one pass of pushes |
| `lgb-replay`, `lgb-replay-spill` | `lgb-stream` with its sample drawn during one read, which is held or spilled and replayed for the pushes |
| `rust-xgb-dense`, `rust-xgb-columnar` | the Rust driver: each batch converted to a dense float32 matrix in parallel, or handed over as Arrow columns, into XGBoost's own sketch |
| `rust-xgb-columnar-sample`, `rust-xgb-columnar-cuts`, `rust-xgb-dense-sample`, `rust-xgb-dense-cuts`, `rust-xgb-dense-cuts200` | the same with a first Rust pass for the cuts (the t-digest at compression 1,000, at the library's own 200, or a 200,000-row sample), handed over through `ref` |
| `rust-xgb-codes` | the t-digest's cuts, and every batch quantized in Rust to one-byte codes, which XGBoost bins one to one |
| `rust-lgb-f64`, `rust-lgb-f32` | the Rust driver: LightGBM's own sample gathered in one pass, then each batch transposed to rows in parallel and pushed |
| `rust-lgb-forced`, `rust-lgb-codes` | the same with the t-digest's cuts as forced bins (252 of them), or with every batch quantized to codes |

`--f32` casts the features to Float32 as the replays read them.
