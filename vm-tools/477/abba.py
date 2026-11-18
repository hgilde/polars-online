#!/usr/bin/env python3
"""ABBA of two builds of rls_factor_bench: abba.py <A> <B> [grid]  (ROUNDS=5 REPS=3 env).
Per cell: best of the rounds' best-of-REPS ns/row for A and B, their ratio A/B (>1 means B is faster), the load at the
start and the highest seen, and whether the two builds' digests (outputs, coefficients, saved states) agree.
Rounds alternate A B / B A. Waits while the 1-minute load is above 15."""
import subprocess, os, re, sys, time
A, B = sys.argv[1], sys.argv[2]
ROUNDS = int(os.environ.get("ROUNDS", "5")); REPS = os.environ.get("REPS", "3")
MODES = os.environ.get("MODES", "never").split(",")
bins = [("A", A), ("B", B)]
def run(k, m, g, mode, rows):
    res = {n: [] for n, _ in bins}; dig = {}
    l0 = os.getloadavg()[0]; mx = l0
    for r in range(ROUNDS):
        for n, p in (bins if r % 2 == 0 else bins[::-1]):
            while os.getloadavg()[0] > 15: time.sleep(10)
            o = subprocess.run([p, str(k), str(m), str(g), mode, str(rows), REPS], capture_output=True, text=True).stdout
            mm = re.search(r"min ([\d.]+) ns/row digest (\w+)", o)
            res[n].append(float(mm.group(1))); dig[n] = mm.group(2); mx = max(mx, os.getloadavg()[0])
    a, b = min(res["A"]), min(res["B"])
    pa, pb = sorted(res["A"])[len(res["A"]) // 4], sorted(res["B"])[len(res["B"]) // 4]
    spread = lambda v: (max(v) - min(v)) / min(v) * 100
    same = "digests equal" if len(set(dig.values())) == 1 else "DIGESTS DIFFER"
    print(f"k_total={k+1:<3} targets={m} groups={g:<5} {mode:<5}: A {a:8.1f} B {b:8.1f} ratio A/B {a/b:.3f}  (25th pct {pa/pb:.3f})  "
          f"spread within A {spread(res['A']):.1f}% B {spread(res['B']):.1f}%  load {l0:.1f}->{mx:.1f}  {same}", flush=True)
rows1 = {4: 400000, 8: 300000, 20: 200000, 40: 120000, 200: 20000}
rows2 = {4: 400000, 8: 400000, 20: 400000, 40: 300000, 200: 40000}
for g, rows in ((1, rows1), (10000, rows2)):
    for kt in (4, 8, 20, 40, 200):
        for mode in MODES: run(kt - 1, 1, g, mode, rows[kt])
if os.environ.get("TARGETS3"):
    run(19, 3, 1, "coef", 100000); run(19, 3, 10000, "coef", 400000)
print("GRID DONE")
