"""Counters per row inside WindowsRun::feed_inner from an Instruments polars-cpu-counters trace.
usage: xct.py TRACE ROWS_FED   (ROWS_FED = rows x reps) -> prints cycles instr ipc mispred l1d per row, samples"""
import os, subprocess, sys, xml.etree.ElementTree as ET
env = dict(os.environ, DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer")

def counters(trace, rows_fed, proc="winbench", inside="feed_inner", diff=False, dump=False):
    xml = subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath",
        '/trace-toc/run[@number="1"]/data/table[@schema="counters-profile"]'],
        capture_output=True, text=True, env=env, check=True).stdout
    root = ET.fromstring(xml)
    ids = {}
    for e in root.iter():
        if "id" in e.attrib:
            ids[e.attrib["id"]] = e
    def res(e):
        if e is None: return None
        if "ref" in e.attrib: return ids.get(e.attrib["ref"], e)
        return e
    tot = [0]*6; n = 0; allv = [0]*6
    last = {}
    for r in root.findall(".//row"):
        th = res(r.find("thread"))
        if th is None or proc not in th.attrib.get("fmt", ""): continue
        pmc = res(r.find("pmc-events"))
        if pmc is None or not (pmc.text or "").strip(): continue
        vals = [int(v) for v in pmc.text.split()]
        if diff:
            tid = th.attrib.get("fmt"); prev = last.get(tid); last[tid] = vals
            if prev is None: continue
            vals = [a-b for a, b in zip(vals, prev)]
            if any(x < 0 for x in vals): continue
        for i, x in enumerate(vals[:6]): allv[i] += x
        bt = res(r.find("backtrace"))
        hit = False
        if bt is not None:
            for f in bt.findall("frame"):
                f = res(f)
                if inside in f.attrib.get("name", ""):
                    hit = True; break
        if hit:
            n += 1
            for i, x in enumerate(vals[:6]): tot[i] += x
    return tot, n, allv

if __name__ == "__main__":
    tr, rows = sys.argv[1], float(sys.argv[2])
    for diff in (False, True):
        tot, n, allv = counters(tr, rows, diff=diff)
        cyc, ins, mis, l1 = tot[0], tot[3], tot[4], tot[5]
        print(f"diff={diff} samples={n} cyc/row={cyc/rows:.1f} ins/row={ins/rows:.1f} IPC={ins/max(cyc,1):.2f} mis/row={mis/rows:.2f} l1d/row={l1/rows:.2f}  all-process cyc/row={allv[0]/rows:.1f}")
