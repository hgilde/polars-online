# Task 452 ABBA harness (from the Mac worker W215)

- `heldbench.rs`: copy to `crates/online-bench/examples/heldbench.rs` in each checkout (c260a0d0, vm/452a2, vm/452b), then build with `cargo build --release -p online-bench --example heldbench` and copy the binary to `bin/<label>` (labels: `main`, `a2`, `b`).
- It times a CI bench case's measured block: `HB_REPS` fresh setups, keeping the fastest, and prints `id us_best checksum_bits`.
- `abba.py ROUNDS CASE...` with `VARS=main,a2` (or `main,b`) runs the binaries in ABBA order and prints median ratios to `main` (with min ratios). The checksum bits differ by design for 452's variants.
- Aligned layout: rebuild with `RUSTFLAGS='-C llvm-args=-align-all-functions=6'` into a separate target dir and label the binaries `main-al`, `a2-al`, `b-al`.
