"""The task 443 matrix: every feature off and on, identically for backward (B) and forward (F, reverse=True).

A forward kernel needs a window_size, so the matrix is the WINDOWED case (a 60 s window, half-life 5 s, mean gap 0.5 s:
about 120 rows in the window) for both directions, and the backward RUNNING form (no window) is listed beside it (R) as the
reference: the running form has no forward twin.
cases(direction) -> list of (name, env) ; direction in {"B", "F", "R"}.
"""
BASE = dict(WB_CAPSEC="100000")

# the window and half-life strings of each clock kind: (WB_KIND, window short, default, long, half-life in the harness)
CLK = {
    "dt": dict(WB_KIND="dt"),
    "f": dict(WB_KIND="f"),
    "i": dict(WB_KIND="i"),
    "rc": dict(WB_KIND="none"),
}
WS = {  # window (short, default, long) per clock kind
    "dt": ("2s", "60s", "600s"),
    "f": ("2.0", "60.0", "600.0"),
    "i": ("2000", "60000", "600000"),
    "rc": ("4.0", "120.0", "1200.0"),  # rows: the same 4 / 120 / 1200 rows a window holds as 2 / 60 / 600 s
}


def feats():
    """feature name -> overrides (clock kind keyed 'clk' picks the window string)."""
    f = []
    f.append(("plain", {}))
    f.append(("ws_short", dict(ws=0)))
    f.append(("ws_long", dict(ws=2)))
    f.append(("ops4", dict(WB_OPS="4", WB_OPSET="msrq")))
    f.append(("quantile", dict(WB_OPSET="quant")))
    f.append(("hl_inf", dict(WB_HL="inf")))
    f.append(("cap_fires", dict(WB_CAP="fire", WB_CAPSEC="300")))
    f.append(("restart", dict(WB_RESTART="1")))
    f.append(("nulls10", dict(WB_NULLS="0.1")))
    f.append(("ties20", dict(WB_TIES="0.2")))
    f.append(("clock_float", dict(clk="f")))
    f.append(("clock_int", dict(clk="i")))
    f.append(("clock_rowcount", dict(clk="rc")))
    f.append(("closed_left", dict(WB_CLOSED="left")))
    f.append(("closed_both", dict(WB_CLOSED="both")))
    f.append(("closed_none", dict(WB_CLOSED="none")))
    f.append(("session", dict(WB_SESSION="1000")))
    f.append(("session_str", dict(WB_SESSION="1000", WB_SESSKEY="str")))
    f.append(("session_cat", dict(WB_SESSION="1000", WB_SESSKEY="cat")))
    f.append(("g1", dict(WB_GROUPS="1")))
    f.append(("g100", dict(WB_GROUPS="100")))
    f.append(("g10000", dict(WB_GROUPS="10000")))
    f.append(("g100_str", dict(WB_GROUPS="100", WB_KEY="str")))
    f.append(("g100_cat", dict(WB_GROUPS="100", WB_KEY="cat")))
    f.append(("g10000_str", dict(WB_GROUPS="10000", WB_KEY="str")))
    f.append(("g10000_cat", dict(WB_GROUPS="10000", WB_KEY="cat")))
    f.append(("g100_ops4", dict(WB_GROUPS="100", WB_OPS="4", WB_OPSET="msrq")))
    f.append(("g100_cap_restarts", dict(WB_GROUPS="100", WB_CAPSEC="30")))
    f.append(("g10000_cap_restarts", dict(WB_GROUPS="10000", WB_CAPSEC="300")))
    f.append(("chunk1000", dict(WB_CHUNK="1000")))
    f.append(("chunk100", dict(WB_CHUNK="100")))
    f.append(("gap_exp1", dict(WB_GAP="exp1")))
    f.append(("gap_regular", dict(WB_GAP="reg")))
    f.append(("clock_step4", dict(WB_QUANT="step")))
    f.append(("clock_ms", dict(WB_QUANT="ms")))
    return f


# backward only: the std / var kernels have no forward form
B_ONLY = [("ops4_mixed", dict(WB_OPS="4", WB_OPSET="mixed"))]


def cases(direction):
    out = []
    items = feats() + (B_ONLY if direction in ("B", "R") else [])
    for name, ov in items:
        env = dict(BASE)
        ov = dict(ov)
        clk = ov.pop("clk", "dt")
        env.update(CLK[clk])
        wsi = ov.pop("ws", 1)
        if direction in ("B", "F"):
            env["WB_WS"] = WS[clk][wsi]
        else:
            # the running form: a window-length feature is meaningless
            if name in ("ws_short", "ws_long", "hl_inf"):
                continue
            if name == "quantile":
                pass  # a running quantile exists (ewm_quantile)
        if direction == "F":
            env["WB_REVERSE"] = "1"
        # the half-life of a non-datetime clock is set by the harness; hl_inf stays "inf"
        env.update(ov)
        out.append((f"{direction}_{name}", env))
    return out


def pairs():
    """(feature, B env, F env, R env or None) for the paired runs."""
    b = dict(cases("B")); f = dict(cases("F")); r = dict(cases("R"))
    names = []
    for n in b:
        names.append(n[2:])
    return names, b, f, r


if __name__ == "__main__":
    for d in "BFR":
        c = cases(d)
        print(d, len(c))
        for n, e in c[:3]:
            print("  ", n, e)
