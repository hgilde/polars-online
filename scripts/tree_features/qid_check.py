import numpy as np
import xgboost as xgb


class It(xgb.DataIter):
    def __init__(s, parts):
        s.parts = parts
        s.i = 0
        super().__init__()

    def reset(s):
        s.i = 0

    def next(s, input_data):
        if s.i == len(s.parts):
            return False
        X, y, q, m = s.parts[s.i]
        input_data(data=X, label=y, qid=q, base_margin=m)
        s.i += 1
        return True


rng = np.random.default_rng(0)
X = rng.standard_normal((6, 2))
y = np.array([1, 0, 2, 1, 0, 1.0])
m = np.arange(6.0)
q = np.array([0, 0, 1, 1, 2, 2])
d = xgb.QuantileDMatrix(It([(X[:3], y[:3], q[:3], m[:3]), (X[3:], y[3:], q[3:], m[3:])]))
print("group_ptr, group 1 straddles the batches:", d.get_uint_info("group_ptr"))
print("base_margin carried:", d.get_base_margin())
d2 = xgb.QuantileDMatrix(It([(X[:2], y[:2], q[:2], m[:2]), (X[2:], y[2:], q[2:], m[2:])]))
print("group_ptr, cut at a boundary:", d2.get_uint_info("group_ptr"))
try:
    d3 = xgb.QuantileDMatrix(
        It([(X[:3], y[:3], np.array([2, 2, 2]), m[:3]), (X[3:], y[3:], np.array([0, 0, 1]), m[3:])])
    )
    print("decreasing qid across batches accepted; group_ptr", d3.get_uint_info("group_ptr"))
except Exception as e:
    print("refused", str(e).splitlines()[0])
