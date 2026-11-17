"""VM (Linux x86-64) part: Callgrind instructions, branches and simulated cache misses per fed row, for the trimmed task 443 matrix.
usage: python3 vm_cg.py BIN OUT.tsv [N=20000] [filter]
Per case and direction (B backward windowed, F forward windowed, R backward running; cases.py names), four runs under
`valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes`: (N rows, WB_REPS=1), (N, 2), (2N, 1), (2N, 2).
The feed's cost a row = (events(N,2) - events(N,1)) / N : the data generation and start-up cancel; the same at 2N must agree
(linearity). Events: Ir, Bc (conditional branches), Bcm (mispredicted), D1mr + D1mw (L1D misses), DLmr + DLmw (last-level misses)."""
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cases as C
BIN, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
FILT = sys.argv[4] if len(sys.argv) > 4 else ""
TRIM = ["plain", "ws_short", "ws_long", "ops4", "quantile", "hl_inf", "cap_fires", "restart", "nulls10", "ties20", "clock_float", "clock_int",
        "clock_rowcount", "closed_left", "session", "g1", "g100", "g10000", "g10000_str", "g10000_cap_restarts", "chunk1000", "gap_exp1", "gap_regular"]
def events(env, rows, reps):
    e = {**os.environ, **env, "WB_ROWS": str(rows), "WB_REPS": str(reps)}
    out = f"/tmp/cg.{os.getpid()}.out"
    subprocess.run(["valgrind", "--tool=callgrind", "--cache-sim=yes", "--branch-sim=yes", f"--callgrind-out-file={out}", BIN], capture_output=True, text=True, env=e, check=True)
    ev = summ = None
    for l in open(out):
        if l.startswith("events:"): ev = l.split()[1:]
        if l.startswith("summary:"): summ = [int(x) for x in l.split()[1:]]; break
    os.remove(out)
    d = dict(zip(ev, summ))
    return d
allc = [(n, e) for d in "BFR" for n, e in C.cases(d) if n[2:] in TRIM]
f = open(OUT, "a")
done = {l.split("\t")[0] for l in open(OUT)} if os.path.exists(OUT) else set()
for name, env in allc:
    if FILT and not re.search(FILT, name): continue
    if name in done: continue
    row = [name]
    for rows in (N, 2 * N):
        a = events(env, rows, 1); b = events(env, rows, 2)
        diff = {k: b[k] - a[k] for k in a}
        row += [f"{diff['Ir']/rows:.0f}", f"{diff['Bc']/rows:.1f}", f"{diff['Bcm']/rows:.2f}", f"{(diff['D1mr']+diff['D1mw'])/rows:.2f}", f"{(diff['DLmr']+diff['DLmw'])/rows:.3f}"]
    f.write("\t".join(row) + "\n"); f.flush(); print("\t".join(row), flush=True)
