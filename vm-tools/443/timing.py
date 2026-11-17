"""python3 timing.py BIN ROUNDS OUT [filter]: per feature, ABBA between the directions (B backward windowed, F forward windowed,
R backward running) on one binary: each round runs each direction twice in a palindromic order; ns a row min and p25 per
direction, load before/after, digest per direction (digests of B/F/R differ by design; the check is equality across repeats)."""
import os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as C
BIN, R, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
FILT = sys.argv[4] if len(sys.argv) > 4 else ""
base = dict(WB_ROWS=os.environ.get("TROWS", "2000000"), WB_REPS="3")
names, b, f, r = C.pairs()
done = set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(l.split("\t")[0])
def one(env):
    e = {**os.environ, **base, **env}
    o = subprocess.run([BIN], capture_output=True, text=True, env=e).stdout
    m = re.search(r"ns_row=([\d.]+) digest=(\w+)", o)
    if not m: raise RuntimeError(o)
    return float(m.group(1)), m.group(2)
def quiet(limit=11.0, wait=1500):
    t = time.time()
    while os.getloadavg()[0] > limit and time.time() - t < wait: time.sleep(15)
out = open(OUT, "a")
q = lambda v: sorted(v)[len(v)//4] if v else float("nan")
mn = lambda v: min(v) if v else float("nan")
for n in names:
    if FILT and not re.search(FILT, n): continue
    if n in done: continue
    dirs = {"B": b.get("B_" + n), "F": f.get("F_" + n), "R": r.get("R_" + n)}
    dirs = {k: v for k, v in dirs.items() if v is not None}
    for attempt in range(3):
        quiet()
        res = {k: [] for k in dirs}; dg = {k: set() for k in dirs}
        l0 = os.getloadavg()[0]
        keys = list(dirs)
        for rd in range(R):
            order = keys + keys[::-1] if rd % 2 == 0 else keys[::-1] + keys
            for k in order:
                ns, d = one(dirs[k]); res[k].append(ns); dg[k].add(d)
        l1 = os.getloadavg()[0]
        if max(l0, l1) <= 15: break
    cells = []
    for k in "BFR":
        cells += [f"{mn(res.get(k, [])):.1f}", f"{q(res.get(k, [])):.1f}"]
    stable = all(len(v) == 1 for v in dg.values())
    line = "\t".join([n] + cells + [f"{l0:.1f}-{l1:.1f}", "stable" if stable else "UNSTABLE"])
    out.write(line + "\n"); out.flush(); print(line, flush=True)
