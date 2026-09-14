#!/usr/bin/env bash
# Native leak check for the FFI boundary, beyond what tests/test_ffi_memory.py
# can see. That test watches RSS, so it needs a leak big enough to move it;
# this one counts blocks nothing references any more, however small.
#
# What it sees, measured on 2026-09-03 (docs/PLAN.md task 18): Python objects
# -- a pyo3 reference that is never released -- and only under
# PYTHONMALLOC=malloc, because pymalloc hands objects out of mmap'd arenas that
# neither tool walks: with the default allocator `leaks` reported "0 leaks"
# against a deliberate 128 MB refcount leak. What it cannot see: anything
# allocated through polars' allocator (mimalloc on macOS, jemalloc on Linux),
# which is every Rust-side allocation here -- 160 MB of deliberately leaked
# Series buffers did not move the count. That side is test_ffi_memory.py's.
#
# The count is differential. The interpreter leaves ~11k blocks (~700 KB)
# unreachable at exit whatever the workload, so one run says nothing; a leak
# per iteration shows as growth from a 1-iteration run to a 1000-iteration
# one. Two runs of the same workload differed by <150 blocks; the threshold
# is 500 blocks or 64 KiB. LEAKCHECK_CONTROL=1 adds a deliberate one-block
# leak per iteration, and must fail: that is the proof the check still sees.
#
#   macOS  -- `leaks`, which walks the malloc zones for unreachable blocks.
#   Linux  -- valgrind, "definitely lost" blocks.
#
# Not part of scripts/gate.sh: `leaks` needs a live process and valgrind is
# ~50x slower. Weekly in CI (.github/workflows/leakcheck.yml); run it by hand
# after touching crates/online-py or the extraction path in online-polars.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh 2>/dev/null || true

# The project venv, not whatever `python3` resolves to -- the system one has no
# polars, and the workload then does nothing while still reporting "0 leaks".
PYBIN=$(uv run python -c 'import sys; print(sys.executable)')
echo "interpreter: $PYBIN"

WORK=$(cat <<'PY'
import gc, os, sys, warnings
import numpy as np, polars as pl, polars_online as po
# The expression form warns on every call (it is O(data)); this is a frame in
# memory, and a thousand warnings would bury the leak numbers.
warnings.filterwarnings("ignore", category=po.InMemoryExpressionWarning)
n = int(sys.argv[1])
control = os.environ.get("LEAKCHECK_CONTROL") == "1"
if control:
    import ctypes
rng = np.random.default_rng(0)
spec = po.spec.ewridge("m", targets=["y"], features=["x0", "x1"],
                       halflife=50.0, min_periods=2.0)
cat = None
for i in range(n):
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
    if control:                # one Python object per iteration, never released
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(bytearray(64)))
    del df, out
    if i % 50 == 0:
        gc.collect()
gc.collect()
PY
)

# Where a leak comes from, on Linux: valgrind's loss records -- each a count of
# definitely-lost blocks with the stack that allocated them -- compared
# between the short run and the long one, largest growth first. Two runs of
# the same stack are one record apiece, so the difference is the leak.
GROWTH=$(cat <<'PY'
import re, sys
from collections import defaultdict
head = re.compile(r"==\d+== ([\d,]+)(?: \([\d,]+ direct, [\d,]+ indirect\))? "
                  r"bytes in ([\d,]+) blocks are definitely lost")
frame = re.compile(r"==\d+==\s+(?:at|by) 0x[0-9A-Fa-f]+: (.*)")

def records(path):
    out = defaultdict(lambda: [0, 0])
    cur, stack = None, []
    for line in list(open(path, errors="replace")) + [""]:
        m = head.search(line)
        if m:
            cur = (int(m.group(2).replace(",", "")), int(m.group(1).replace(",", "")))
            stack = []
        elif cur is not None and (f := frame.search(line)):
            stack.append(f.group(1))
        elif cur is not None:
            key = "\n".join(stack)
            out[key][0] += cur[0]
            out[key][1] += cur[1]
            cur = None
    return out

small, large = records(sys.argv[1]), records(sys.argv[2])
rows = sorted(((b - small.get(k, [0, 0])[0], y - small.get(k, [0, 0])[1], k)
               for k, (b, y) in large.items()), reverse=True)
for blocks, nbytes, stack in [r for r in rows if r[0] > 0][:5]:
    print(f"+{blocks} blocks, +{nbytes} bytes, allocated at:")
    for f in stack.split("\n")[:14]:
        print("    " + f)
PY
)

# Each measurement's whole report, kept for when it cannot be parsed.
RAW=$(mktemp -d)
trap 'rm -rf "$RAW"' EXIT

# measure N -> "blocks bytes" left unreachable at exit after N iterations. The
# checker's report goes to $RAW/N.txt, its exit status on the last line.
#
# No MallocStackLogging: it records where each block was allocated, which the
# count does not read, and without it the check and the control give the same
# verdicts on macOS 15.7.3, in seconds. Set it by hand to see where a leaked
# block came from. On GitHub's macOS runners `leaks` gives no report at all,
# on 15.7.9 and 26.6.2 alike: see the matrix in .github/workflows/leakcheck.yml.
measure() {
  local raw="$RAW/$1.txt"
  case "$(uname -s)" in
    Darwin)
      PYTHONMALLOC=malloc leaks --atExit --quiet -- "$PYBIN" -c "$WORK" "$1" >"$raw" 2>&1
      echo "(exit status $?)" >>"$raw"
      sed -n 's/.*: \([0-9]*\) leaks for \([0-9]*\) total leaked bytes.*/\1 \2/p' "$raw" | head -1
      ;;
    Linux)
      PYTHONMALLOC=malloc valgrind --leak-check=full --show-leak-kinds=definite --errors-for-leak-kinds=definite \
        "$PYBIN" -c "$WORK" "$1" >"$raw" 2>&1
      echo "(exit status $?)" >>"$raw"
      sed -n 's/.*definitely lost: \([0-9,]*\) bytes in \([0-9,]*\) blocks.*/\2 \1/p' "$raw" | tr -d , | head -1
      ;;
  esac
}

case "$(uname -s)" in
  Darwin) command -v leaks >/dev/null || { echo "leaks not found (Xcode command line tools)"; exit 2; } ;;
  Linux) command -v valgrind >/dev/null || { echo "valgrind not installed; skipping (apt-get install valgrind)"; exit 2; } ;;
  *) echo "no native leak checker wired up for $(uname -s)"; exit 2 ;;
esac

# One unmeasured run first, so that both measured ones import every module the
# same way. The first process to import a module compiles it from source and
# writes its .pyc; every later one loads the .pyc, and a module loaded that way
# leaves more blocks unreachable at exit than one compiled. On a fresh checkout
# the short run was that first process, and the difference came out as growth:
# +611 blocks on the CI runners, +3,143 on a Mac here with every cache cold, +1
# with them warm (2026-09-14). valgrind put all of it in marshal_loads, under
# the import machinery, and none in this package.
"$PYBIN" -c "$WORK" 1 >/dev/null 2>&1
small=$(measure 1)
large=$(measure 1000)
if [[ -z "$small" || -z "$large" ]]; then
  echo "could not parse the leak report (small='$small' large='$large')"
  # What the checker printed instead. Without it the line above was the whole
  # story: the weekly runs of 2026-09-07 and -14 on macOS said nothing more.
  for n in 1 1000; do
    echo "--- the report for $n iteration(s), last 40 lines:"
    tail -n 40 "$RAW/$n.txt"
  done
  exit 2
fi
read -r b1 y1 <<<"$small"
read -r b2 y2 <<<"$large"
db=$((b2 - b1)); dy=$((y2 - y1))
printf '%-14s %10s %12s\n' "iterations" "blocks" "bytes"
printf '%-14s %10d %12d\n' 1 "$b1" "$y1"
printf '%-14s %10d %12d\n' 1000 "$b2" "$y2"
printf '%-14s %10d %12d\n' "growth" "$db" "$dy"
if (( db > 500 || dy > 65536 )); then
  echo "LEAK: unreachable blocks grow with the workload${LEAKCHECK_CONTROL:+ (control: expected)}"
  # Where it was allocated, when there is a stack to say so: valgrind keeps
  # one per record, `leaks` without stack logging does not. Not for the
  # control, whose leak is its own.
  if [[ "${LEAKCHECK_CONTROL:-}" != "1" && "$(uname -s)" == Linux ]]; then
    echo "--- where the growth was allocated (1000 iterations against 1):"
    "$PYBIN" -c "$GROWTH" "$RAW/1.txt" "$RAW/1000.txt"
  fi
  exit 1
fi
echo "clean: growth within noise${LEAKCHECK_CONTROL:+ -- CONTROL LEAK NOT SEEN, the check is blind}"
[[ "${LEAKCHECK_CONTROL:-}" == "1" ]] && exit 3
exit 0
