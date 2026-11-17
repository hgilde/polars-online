import subprocess, sys, json, statistics, os
V = os.environ.get("VARS","main,a1,a1e,a2,a3").split(",")
cases = sys.argv[2:]; rounds = int(sys.argv[1])
res = {v:{c:[] for c in cases} for v in V}; ck = {}
for r in range(rounds):
    order = V if r%2==0 else V[::-1]
    for v in order:
        out = subprocess.run(["bin/"+v,*cases],capture_output=True,text=True,env={**os.environ,"HB_REPS":"8"}).stdout
        la = open("/dev/null")
        load = os.getloadavg()[0]
        for ln in out.splitlines():
            c,t,_,bits = ln.split()
            res[v][c].append(float(t)); ck[(v,c)] = bits
    print("round",r,"load",os.getloadavg()[0],flush=True)
print("case", *V)
for c in cases:
    base = statistics.median(res["main"][c])
    print(c, f"main {base:.0f}us", *[f"{v} {statistics.median(res[v][c])/base:.3f} (min {min(res[v][c])/min(res['main'][c]):.3f})" for v in V[1:]], "bits-eq:", *[ck[(v,c)]==ck[("main",c)] for v in V[1:]])
