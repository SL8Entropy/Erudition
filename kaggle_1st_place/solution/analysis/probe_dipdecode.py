"""Second-order (TVT level, u-dip) forward-backward over saved column posteriors, with
tempered emissions p**tau, against the team's first-order decode (posterior_fb_decode.py).

    set PROBE_RUN=results/<dump dir>
    python analysis/probe_dipdecode.py

u = TVT + Z.  Each column step the level moves by (dip - dZ); the dip stays put except for
rare jumps (eps_dip); the starting dip comes from the known prefix.  Writes
analysis/probe_dipdecode_sse_<dump>.csv (per-well SSE for every decoder); probe_robust.py
turns that into cross-fitted RMSE, k*, bootstrap and median-well figures.

Result (2026-09-29/10-10): helps weak models (60-epoch baseline -0.18, FastViT student
-0.17 ft cross-fitted) but only -0.03 on a 150-epoch ConvNeXt-small.  Untempered (tau=1)
it is worse than the mean decode, which is why the old first-order decode failed.
"""
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from probe_common import DS, RUN, SOL, columns, load_csv  # noqa: E402

sys.path.insert(0, str(SOL))
from posterior_fb_decode import forward_backward as fb1  # noqa: E402  the first-order decode

STEP = 0.5
DIPS = np.round(np.arange(-3.0, 3.0001, 0.1), 3)          # u-dip grid, ft per 32-row column
X = (np.arange(400) - 199.5) * STEP                         # == the dump's "levels"
JUMP = np.exp(-0.5 * ((X[:, None] - X[None, :]) / 6.0) ** 2)
JUMP /= JUMP.sum(0, keepdims=True)


def dip_kernel(eps, wide=0.6):
    K = eps * np.exp(-0.5 * ((DIPS[:, None] - DIPS[None, :]) / wide) ** 2)
    K /= K.sum(1, keepdims=True)
    K = K + (1 - eps) * np.eye(len(DIPS))
    return K / K.sum(1, keepdims=True)


def shift(a, delta_bins):
    """out[i,k] = a[i - delta_k, k] with linear interpolation, zero outside."""
    L = a.shape[0]
    m = np.floor(delta_bins).astype(int)
    f = delta_bins - m
    i = np.arange(L)[:, None]

    def g(off):
        j = i - off[None, :]
        ok = (j >= 0) & (j < L)
        return np.where(ok, a[np.clip(j, 0, L - 1), np.arange(a.shape[1])[None, :]], 0.0)

    return (1 - f)[None, :] * g(m) + f[None, :] * g(m + 1)


def fb2(P, dZ, d0, tau, eps_dip, sd0, eps_jump=0.01, start_sig=2.0):
    T = len(P)
    K = dip_kernel(eps_dip)
    E = np.clip(P, 1e-12, None) ** tau
    prior_d = np.exp(-0.5 * ((DIPS - d0) / sd0) ** 2)
    a = np.empty((T, 400, len(DIPS)))
    a0 = (E[0] * np.exp(-0.5 * (X / start_sig) ** 2))[:, None] * prior_d[None, :]
    a[0] = a0 / a0.sum()

    def trans(v, t):   # level moves by (dip - dZ_t), dip may change, rare level jump
        v = shift(v, (DIPS - dZ[t]) / STEP)
        v = v @ K
        return (1 - eps_jump) * v + eps_jump * (JUMP @ v)

    for t in range(1, T):
        v = trans(a[t - 1], t) * E[t][:, None]
        a[t] = v / v.sum()
    b = np.ones((400, len(DIPS)))
    g = np.empty((T, 400))
    g[-1] = a[-1].sum(1)
    for t in range(T - 1, 0, -1):          # transposed operators in reverse order
        v = b * E[t][:, None]
        v = (1 - eps_jump) * v + eps_jump * (JUMP.T @ v)
        v = v @ K.T
        v = shift(v, -(DIPS - dZ[t]) / STEP)
        b = v / v.sum()
        g[t - 1] = (a[t - 1] * b).sum(1)
    g /= g.sum(1, keepdims=True)
    return g @ X


def well_inputs(wid, z):
    df = load_csv(wid)
    suf, rb = columns(df)
    bins = z[f"{wid}__bins"].astype(int)
    Zr = df.Z.to_numpy()[suf]
    Zc = np.array([Zr[rb == b].mean() for b in bins])
    kn = df[df.TVT_input.notna()].tail(320)
    u = (kn.TVT_input + kn.Z).to_numpy()
    d0 = np.polyfit(np.arange(len(u)), u, 1)[0] * DS if len(u) > 20 else 0.0
    return z[f"{wid}__p"].astype(np.float64), np.r_[0.0, np.diff(Zc)], float(d0)


TAUS = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05)
EPS = (0.02, 0.05, 0.15)
SD0 = (0.2, 0.6)
CFG2 = list(itertools.product(TAUS, EPS, SD0))


def run_well(wid):
    z = np.load(RUN / "posterior_all.npz")
    P, dZ, d0 = well_inputs(wid, z)
    P /= P.sum(1, keepdims=True)
    out = {"well_id": wid, "d0": d0}
    for tau in TAUS:
        Pt = P ** tau
        Pt /= Pt.sum(1, keepdims=True)
        out[f"fo_t{tau}"] = fb1(Pt, X)[0]
    for tau, e, s in CFG2:
        out[f"so_t{tau}_e{e}_s{s}"] = fb2(P, dZ, d0, tau, e, s)
    return out


def main():
    z = np.load(RUN / "posterior_all.npz")
    wells = [str(w) for w in z["focus"]]
    cols = pd.read_parquet(RUN / "posterior_columns.parquet")
    wtab = pd.read_csv(RUN / "posterior_wells.csv").set_index("well_id")
    with Pool(4) as pool:          # 8 processes ran a 16 GB laptop out of memory
        res = pool.map(run_well, wells)
    rows, keys = [], None
    for r in res:
        wid = r["well_id"]
        bins = z[f"{wid}__bins"].astype(int)
        c = cols[cols.well_id == wid].set_index("bin").loc[bins]
        n, y, mu = (c[k].to_numpy(float) for k in ("n_rows", "y", "mu"))
        within = n.sum() * wtab.loc[wid, "rmse_mean"] ** 2 - (n * (mu - y) ** 2).sum()
        keys = ["mean"] + [k for k in r if k.startswith(("fo_", "so_"))]
        row = {"well_id": wid, "rows": n.sum(), "d0": r["d0"]}
        for k in keys:
            v = mu if k == "mean" else r[k]
            row[k] = (n * (v - y) ** 2).sum() + within
        rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(__file__).parent / f"probe_dipdecode_sse_{RUN.name}.csv"
    df.to_csv(out, index=False)
    N = df.rows.sum()
    ins = {k: float(np.sqrt(df[k].sum() / N)) for k in keys}
    print("in-sample pooled RMSE (optimistic), best 5:")
    for k, v in sorted(ins.items(), key=lambda kv: kv[1])[:5]:
        print(f"  {k:24s} {v:.4f}")
    print(f"  mean {ins['mean']:.4f} | first-order raw {ins['fo_t1.0']:.4f}")
    print(f"wrote {out}; run probe_robust.py for cross-fitted and robustness figures")


if __name__ == "__main__":
    main()
