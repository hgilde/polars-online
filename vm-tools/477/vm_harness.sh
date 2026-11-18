#!/usr/bin/env bash
# W477 harness for the x86 VM (Linux, valgrind installed). Compares rls's step_into on main and on branch task-477-rls-triangle.
#   usage: vm_harness.sh <main checkout> <branch checkout>
# Each checkout gets rls_factor_bench.rs copied into crates/online-core/examples/ (untracked; delete after), is built with
# `cargo build --release -p online-core --example rls_factor_bench`, and then:
#  1. Callgrind: Ir per row = (Ir of 4000 rows - Ir of 2000 rows) / 2000 for k_total 4, 8, 20, 21, 40 (features = k_total-1), 1 target
#     and 3 targets at 20; 200 uses 40 and 20 rows. Fixed costs (setup, the final state digest) cancel in the difference.
#  2. Time: abba.py (5 rounds ABBA, best-of-3) over k_total 4, 8, 20, 40, 200 at 1 and 10,000 groups; compare the two ratios to
#     a control run of main against a second copy of itself (control spread). Quiet machine; log `uptime`.
# Compare: the Callgrind Ir/row of the branch against main's (the branch should be at or below, every row); the digests
# printed on both lines of each ABBA cell must say "digests equal".
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MAIN="$1"; NEW="$2"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
for d in "$MAIN" "$NEW"; do
  cp "$HERE/rls_factor_bench.rs" "$d/crates/online-core/examples/rls_factor_bench.rs"
  (cd "$d" && cargo build --release -p online-core --example rls_factor_bench 2>&1 | tail -1)
done
BM="$MAIN/target/release/examples/rls_factor_bench"; BN="$NEW/target/release/examples/rls_factor_bench"
cp "$BM" "$HERE/vm-main"; cp "$BM" "$HERE/vm-main2"; cp "$BN" "$HERE/vm-new"
ir() { # binary features targets rows
  valgrind --tool=callgrind --callgrind-out-file=/dev/null "$1" "$2" "$3" 1 never "$4" 1 2>&1 | awk '/Collected/{print $4}'
}
echo "== Callgrind Ir per row (x86 baseline build; uptime: $(uptime))"
for spec in "3 1 2000 4000" "7 1 2000 4000" "19 1 2000 4000" "20 1 2000 4000" "39 1 2000 4000" "19 3 2000 4000" "199 1 20 40"; do
  set -- $spec
  for b in main new; do
    lo=$(ir "$HERE/vm-$b" $1 $2 $3); hi=$(ir "$HERE/vm-$b" $1 $2 $4)
    echo "k_total $(( $1 + 1 )) targets $2 $b: $(( (hi - lo) / ($4 - $3) )) Ir/row"
  done
done
echo "== control (main vs a copy of itself), then main vs branch"
python3 "$HERE/abba.py" "$HERE/vm-main" "$HERE/vm-main2" | tee "$HERE/vm-control.log"
python3 "$HERE/abba.py" "$HERE/vm-main" "$HERE/vm-new" | tee "$HERE/vm-abba.log"
rm -f "$MAIN/crates/online-core/examples/rls_factor_bench.rs" "$NEW/crates/online-core/examples/rls_factor_bench.rs"
