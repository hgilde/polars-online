import subprocess, sys, re, os, collections, statistics
# usage: abba469.py <layout> <intercept 0|1> <rounds> [orders...]
sp = "/private/tmp/claude-501/-Users-hgilde-dev-polars-online/1500c11d-aa39-4246-b368-45247642bebb/scratchpad/p469/bin"
lay, inter, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
orders = sys.argv[4:] or [str(i) for i in range(1, 25)] + ["28", "32", "40", "48", "56", "64", "80", "100", "128", "150", "200"]
res = collections.defaultdict(lambda: collections.defaultdict(list)); loads = []
for r in range(rounds):
    loads.append(os.getloadavg()[0])
    for t in "ABCDDCBA":
        out = subprocess.run([f"{sp}/probe-{t}-{lay}", inter] + orders, capture_output=True, text=True).stdout
        for l in out.splitlines():
            m = re.match(r"wald order\s+(\d+)\s+([\d.]+) ns/call", l)
            if m: res[int(m.group(1))][t].append(float(m.group(2)))
print(f"layout {lay}, intercept {inter}, rounds {rounds}, loads {[round(x,1) for x in loads]}")
print("order  main(us)  468/main  469/main  control/main | med: 468 469 ctl | spread(main)")
for o, d in sorted(res.items()):
    a, b, c, e = (sorted(d[t]) for t in "ABCD")
    med = lambda x: statistics.median(x) / statistics.median(d["A"])
    print(f"{o:5} {a[0]/1e3:9.3f} {b[0]/a[0]:8.3f} {c[0]/a[0]:8.3f} {e[0]/a[0]:8.3f} | {med(d['B']):.3f} {med(d['C']):.3f} {med(d['D']):.3f} | {max(a)/a[0]:.3f}")
