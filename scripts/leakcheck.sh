#!/usr/bin/env bash
# Native leak check for the FFI boundary, beyond what tests/test_ffi_memory.py
# can see. RSS plateau tests catch a leak that grows; this catches allocated
# blocks that nothing references any more, however small.
#
#   macOS  -- `leaks`, which walks the heap for unreachable blocks.
#   Linux  -- valgrind with CPython's own suppression file, if both are present.
#
# Not part of scripts/gate.sh: `leaks` needs a live process and valgrind makes
# the suite ~50x slower. Run it after touching anything in crates/online-py or
# the extraction path in crates/online-polars.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh 2>/dev/null || true

# The project venv, not whatever `python3` resolves to -- the system one has no
# polars, and the workload then does nothing while still reporting "0 leaks".
PYBIN=$(uv run python -c 'import sys; print(sys.executable)')
echo "interpreter: $PYBIN"

WORK=$(cat <<'PY'
import gc
import numpy as np, polars as pl, polars_online as po
rng = np.random.default_rng(0)
spec = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"],
                       halflife=50.0, min_periods=2.0)
cat = None
for i in range(300):
    df = pl.DataFrame({"x0": rng.standard_normal(700), "x1": rng.standard_normal(700)})
    df = df.with_columns(y=pl.col("x0") * 2)
    out = po.ModelBank([spec]).fit_predict(df)
    _ = out["m"].struct.field("pred_y").sum()
    df.with_columns(pl.col("y").online.ewridge(features=["x0"], halflife=50.0,
                                               min_periods=2.0))
    if cat is None:
        cat = df.with_columns(c=pl.col("x1").cast(pl.String).cast(pl.Categorical))
    try:                       # the error path, where release is easiest to miss
        po.ModelBank([spec]).fit_predict(cat)
    except Exception:
        pass
    del df, out
    if i % 50 == 0:
        gc.collect()
gc.collect()
PY
)

case "$(uname -s)" in
  Darwin)
    echo "--- leaks (macOS) ---"
    MallocStackLogging=1 leaks --atExit --quiet -- "$PYBIN" -c "$WORK" 2>&1 \
      | grep -E "leaks for|total leaked|Leak:|LEAK" | head -30
    ;;
  Linux)
    if command -v valgrind >/dev/null; then
      SUPP=$("$PYBIN" -c "import sysconfig,os;p=os.path.join(sysconfig.get_paths()['data'],'share','doc','python3','valgrind-python.supp');print(p if os.path.exists(p) else '')")
      echo "--- valgrind (Linux) ---"
      PYTHONMALLOC=malloc valgrind --leak-check=full --show-leak-kinds=definite --error-exitcode=99 \
        ${SUPP:+--suppressions="$SUPP"} "$PYBIN" -c "$WORK" 2>&1 | grep -E "definitely lost|ERROR SUMMARY" | head
    else
      echo "valgrind not installed; skipping (apt-get install valgrind)"
    fi
    ;;
  *) echo "no native leak checker wired up for $(uname -s)" ;;
esac
