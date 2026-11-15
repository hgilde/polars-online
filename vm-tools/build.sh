#!/bin/bash
W=/tmp/claude-0/-home-user-polars-online/656039a6-d4fb-5436-a48f-6fa25c493b59/scratchpad/v64
export RUSTUP_TOOLCHAIN=1.99.1 CARGO_BUILD_JOBS=3
for t in A B; do
 [ -x $W/probe-$t ] || (cd $W/tree-$t && CARGO_TARGET_DIR=$W/target-$t cargo build --release -q -p online-polars --example hmm_ew_probe > $W/out/build-$t.log 2>&1 && cp $W/target-$t/release/examples/hmm_ew_probe $W/probe-$t)
done
touch $W/out/built
