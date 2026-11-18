import subprocess, sys, re, os, collections, statistics
sp=sys.argv[1]; lay=sys.argv[2]; rounds=int(sys.argv[3])
orders=["20","24","32","64","100","200"]
res=collections.defaultdict(lambda: collections.defaultdict(list)); loads=[]
for r in range(rounds):
    loads.append(os.getloadavg()[0])
    for t in "ABCCBA":
        out=subprocess.run([f"{sp}/probe-{t}-{lay}"]+orders,capture_output=True,text=True).stdout
        for l in out.splitlines():
            m=re.match(r"wald order\s+(\d+)\s+([\d.]+) ns/call",l)
            if m: res[int(m.group(1))][t].append(float(m.group(2)))
print(f"layout {lay}, rounds {rounds}, loads {[round(x,1) for x in loads]}")
print("order  A min(us)  B/A min  C/A min(control)  B/A med  C/A med  A spread")
for o,d in sorted(res.items()):
    a,b,c=(sorted(d[t]) for t in "ABC")
    print(f"{o:5} {a[0]/1e3:10.1f} {b[0]/a[0]:8.3f} {c[0]/a[0]:8.3f} {statistics.median(b)/statistics.median(a):8.3f} {statistics.median(c)/statistics.median(a):8.3f} {max(a)/a[0]:8.3f}")
