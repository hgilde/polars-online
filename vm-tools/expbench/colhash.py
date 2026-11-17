"""colhash.py DIR: sha256 (first 16 hex) of the args column and of each implementation's column, per <fn>_<set>.bin."""
import hashlib, os, sys
d = sys.argv[1]
for name in sorted(f for f in os.listdir(d) if f.endswith(".bin")):
    stem = name[:-4]
    cols = ["args"] + open(os.path.join(d, stem + ".txt")).read().strip().split(",")
    data = open(os.path.join(d, name), "rb").read()
    n = len(data) // (8 * len(cols))
    out = [f"{c}={hashlib.sha256(data[i*n*8:(i+1)*n*8]).hexdigest()[:16]}" for i, c in enumerate(cols)]
    print(stem, *out)
