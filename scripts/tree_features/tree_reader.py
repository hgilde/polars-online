"""Candidate 5's reader, de-risked: evaluate XGBoost (JSON) and LightGBM (text)
model dumps in plain Python and compare every row's leaf, in every tree, with
the library's own pred_leaf. Inputs carry NaNs, a categorical with partition
splits, unseen and negative codes, values on thresholds, and +-inf."""

import json

import lightgbm as lgb
import numpy as np
import xgboost as xgb

rng = np.random.default_rng(0)
n, p = 20000, 6
X = rng.standard_normal((n, p))
X[:, 1] = np.round(X[:, 1], 1)  # ties on thresholds
cat = rng.integers(0, 30, n).astype(float)
X[:, 5] = cat
y = np.sin(2 * X[:, 0]) + (cat % 7) * 0.3 + X[:, 1] * (X[:, 2] > 0) + 0.3 * rng.standard_normal(n)
X[rng.random((n, p)) < 0.1] = np.nan  # 10% missing, everywhere
Xt = X.copy()
Xt[:200, 5] = 35.0  # unseen category
Xt[200:300, 5] = -1.0  # negative category
Xt[300:400, 0] = np.inf
Xt[400:500, 0] = -np.inf
Xt[500:600, 3] = 0.0  # exact zeros

# ---------------- XGBoost ----------------
d = xgb.DMatrix(X, y, feature_types=["q"] * 5 + ["c"], enable_categorical=True)
bx = xgb.train(
    {"tree_method": "hist", "max_depth": 6, "eta": 0.2, "max_cat_to_onehot": 1, "nthread": 4}, d, 60
)
raw = bytearray(bx.save_raw("json"))
m = json.loads(raw)
trees = m["learner"]["gradient_booster"]["model"]["trees"]


def xgb_leaf(tree, row):
    L, R = tree["left_children"], tree["right_children"]
    si, sc, dl, st = (
        tree["split_indices"],
        tree["split_conditions"],
        tree["default_left"],
        tree["split_type"],
    )
    cn, cseg, csz, cats = (
        tree["categories_nodes"],
        tree["categories_segments"],
        tree["categories_sizes"],
        tree["categories"],
    )
    catmap = {node: set(cats[cseg[k] : cseg[k] + csz[k]]) for k, node in enumerate(cn)}
    nid = 0
    while L[nid] != -1:
        f = np.float32(row[si[nid]])
        if np.isnan(f):
            nid = L[nid] if dl[nid] else R[nid]
        elif st[nid] == 1:  # categorical: in the set goes right; invalid goes left
            inval = f < 0 or f >= 16777216
            left = True if inval else (int(f) not in catmap[nid])
            nid = L[nid] if left else R[nid]
        else:
            nid = L[nid] if f < np.float32(sc[nid]) else R[nid]
    return nid


Xx = np.where(np.isinf(Xt), np.sign(Xt) * 1e30, Xt)  # XGBoost refuses inf in input
dt = xgb.DMatrix(Xx, feature_types=["q"] * 5 + ["c"], enable_categorical=True)
lib = bx.predict(dt, pred_leaf=True).astype(int)
mine = np.array([[xgb_leaf(t, r) for t in trees] for r in Xx])
print(
    "XGBoost",
    xgb.__version__,
    "trees",
    len(trees),
    "rows",
    n,
    "leaf mismatches:",
    int((lib != mine).sum()),
    "| categorical split nodes:",
    sum(len(t["categories_nodes"]) for t in trees),
)

# ---------------- LightGBM ----------------
dl_ = lgb.Dataset(X, y, categorical_feature=[5], free_raw_data=False)
bl = lgb.train(
    {
        "num_leaves": 31,
        "learning_rate": 0.2,
        "verbose": -1,
        "num_threads": 4,
        "max_cat_to_onehot": 1,
        "min_data_per_group": 10,
        "cat_smooth": 1,
        "zero_as_missing": False,
    },
    dl_,
    60,
)
txt = bl.model_to_string()


def parse_lgb(txt):
    out = []
    for block in txt.split("Tree=")[1:]:
        kv = {}
        for line in block.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k] = v
        if "end of trees" in block and "split_feature" not in kv:
            continue
        t = {"num_leaves": int(kv["num_leaves"])}
        if t["num_leaves"] > 1:
            t["sf"] = list(map(int, kv["split_feature"].split()))
            t["th"] = list(map(float, kv["threshold"].split()))
            t["dt"] = list(map(int, kv["decision_type"].split()))
            t["lc"] = list(map(int, kv["left_child"].split()))
            t["rc"] = list(map(int, kv["right_child"].split()))
            if "cat_boundaries" in kv:
                t["cb"] = list(map(int, kv["cat_boundaries"].split()))
                t["ct"] = list(map(int, kv["cat_threshold"].split()))
        out.append(t)
    return out


def lgb_leaf(t, row):
    if t["num_leaves"] == 1:
        return 0
    node = 0
    while node >= 0:
        fv = float(row[t["sf"][node]])
        dtp = t["dt"][node]
        if dtp & 1:  # categorical
            if np.isnan(fv) or int(fv) < 0:
                node = t["rc"][node]
                continue
            iv = int(fv)
            ci = int(t["th"][node])
            lo, hi = t["cb"][ci], t["cb"][ci + 1]
            words = t["ct"][lo:hi]
            hit = iv // 32 < len(words) and (words[iv // 32] >> (iv % 32)) & 1
            node = t["lc"][node] if hit else t["rc"][node]
        else:
            mt = (dtp >> 2) & 3
            if np.isnan(fv) and mt != 2:
                fv = 0.0
            if (mt == 1 and -1e-35 <= fv <= 1e-35) or (mt == 2 and np.isnan(fv)):
                node = t["lc"][node] if dtp & 2 else t["rc"][node]
            else:
                node = t["lc"][node] if fv <= t["th"][node] else t["rc"][node]
    return ~node


lt = parse_lgb(txt)
libl = bl.predict(Xt, pred_leaf=True).astype(int)
minel = np.array([[lgb_leaf(t, r) for t in lt] for r in Xt])
mt_counts = {}
for t in lt:
    for d_ in t.get("dt", []):
        mt_counts[(d_ & 1, (d_ >> 2) & 3)] = mt_counts.get((d_ & 1, (d_ >> 2) & 3), 0) + 1
print(
    "LightGBM",
    lgb.__version__,
    "trees",
    len(lt),
    "leaf mismatches:",
    int((libl != minel).sum()),
    "| (categorical, missing type) node counts:",
    mt_counts,
)
print(
    "model sizes: xgb json bytes",
    len(raw),
    "ubj bytes",
    len(bx.save_raw("ubj")),
    "lgb text bytes",
    len(txt),
)
for extra in ({"zero_as_missing": True}, {"use_missing": False}):
    b2 = lgb.train(
        {"num_leaves": 31, "learning_rate": 0.2, "verbose": -1, "num_threads": 4, **extra},
        lgb.Dataset(
            np.where(np.isnan(X), 0.0, X) if extra.get("zero_as_missing") else X,
            y,
            categorical_feature=[5],
            free_raw_data=False,
        ),
        30,
    )
    t2 = parse_lgb(b2.model_to_string())
    l2 = b2.predict(Xt, pred_leaf=True).astype(int)
    m2 = np.array([[lgb_leaf(t, r) for t in t2] for r in Xt])
    mc = {}
    for t in t2:
        for d_ in t.get("dt", []):
            mc[(d_ & 1, (d_ >> 2) & 3)] = mc.get((d_ & 1, (d_ >> 2) & 3), 0) + 1
    print("LightGBM", extra, "leaf mismatches:", int((l2 != m2).sum()), "| node kinds:", mc)
