#!/usr/bin/env python3
"""Ensemble the AnchorCNN with the 1st-place ConvNeXt, on the shared 155-well holdout.

Both solutions are scored on byte-identical rows (same split, same 754,122
``(well_id, submit_index)`` keys), and their catastrophic wells barely overlap -- e.g.
``ea3a0e38`` is 2.8 ft for the ConvNeXt and 23.2 ft for the AnchorCNN, ``f5859199`` the
reverse.  That is the precondition for an ensemble to beat both members.

Methods, all scored the same way:

    convnext    the 1st-place model alone
    anchor      the AnchorCNN alone
    mean        fixed 50/50 average; nothing is fitted
    global_w    one blend weight, fitted out of fold (closed form)
    gate_ce     the same gate with 3rd place's second loss term (--gate-ce-w > 0), which
                rewards committing to whichever model is actually closer instead of hedging
    gate        a per-row weight from a small BiGRU over the well, fitted out of fold --
                the 3rd-place "SoftMax gate" idea for two candidates.  The output is a
                convex combination, so it can only choose between the two, never invent
                a third path; 3rd place found an unconstrained refiner generalised worse
    oracle_row  the better of the two at every row, using the answer.  An upper bound,
                not a method

Why it is leak-free.  Neither base model saw a holdout well.  Everything fitted here is
cross-fitted: the 155 wells are split into ``--folds`` groups, and a well's blended
prediction always comes from a blender fitted on the *other* groups, with early stopping
on an inner split of those.  Hyperparameters are fixed in this file, not tuned on the
result.  The residual caveat is size: each blender is fitted on ~124 wells.

    python anchor_ensemble.py --anchor runs/C_seeds_tta8/holdout_predictions.pqt \
        --convnext ../kaggle_1st_place/solution/results/rb_v3_tta_ep150/holdout_predictions.pqt \
        --out runs/ens_convnext_anchor
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from anchor_data import eval_rows, load_wells, split_wells, well_names  # noqa: E402
from anchor_train import Logger                                         # noqa: E402

STRIDE = 32          # rows between gate steps (32 ft)
FEATS = ["diff", "absdiff", "a_off", "c_off", "a_std", "c_nbr", "pos", "md_ps", "gr", "gr_nan", "dip"]


# ------------------------------------------------------------------------------ data


def load_pair(anchor_path: Path, convnext_path: Path, pred_col: str) -> pd.DataFrame:
    a = pd.read_parquet(anchor_path)
    c = pd.read_parquet(convnext_path)
    keys = ["well_id", "submit_index"]
    a = a.sort_values(keys, kind="stable").reset_index(drop=True)
    c = c.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(a) != len(c) or not a[keys].equals(c[keys]):
        raise SystemExit("the two prediction files are not on the same rows")
    if np.abs(a["TVT"].to_numpy() - c["TVT"].to_numpy()).max() > 1e-3:
        raise SystemExit("the two prediction files disagree on the ground truth TVT")
    df = a[keys + ["TVT"]].copy()
    df["p_a"] = a["TVT_pred"].to_numpy(np.float64)
    df["p_c"] = c[pred_col].to_numpy(np.float64)
    df["a_std"] = a["TVT_std"].to_numpy(np.float64) if "TVT_std" in a else 0.0
    df["c_nbr"] = (np.log1p(c["geo_nbr_distance"].to_numpy(np.float64) / 100.0)
                   if "geo_nbr_distance" in c else 0.0)
    return df


def well_sequences(df: pd.DataFrame, wells: dict) -> dict:
    """Per-well arrays: every scored row, plus the gate's features at STRIDE-row steps."""
    out = {}
    for nm, g in df.groupby("well_id", sort=True):
        w = wells[nm]
        ev = eval_rows(w)
        idx = g["submit_index"].to_numpy()
        if not np.array_equal(idx, ev):
            raise SystemExit(f"{nm}: prediction rows do not match the well's eval rows")
        ti = w["tvt_input"]
        tvt_ps = float(ti[np.isfinite(ti)][-1])
        md = w["md"][ev].astype(np.float64)
        gr = w["gr"][ev].astype(np.float64)
        z = w["z"][ev].astype(np.float64)
        p_a, p_c = g["p_a"].to_numpy(), g["p_c"].to_numpy()
        n = len(ev)
        dip = np.gradient(z, md) if n > 1 else np.zeros(n)
        k = np.convolve(dip, np.ones(STRIDE) / STRIDE, mode="same")
        full = dict(
            diff=(p_a - p_c) / 10.0, absdiff=np.abs(p_a - p_c) / 10.0,
            a_off=(p_a - tvt_ps) / 20.0, c_off=(p_c - tvt_ps) / 20.0,
            a_std=g["a_std"].to_numpy() / 10.0, c_nbr=g["c_nbr"].to_numpy(),
            pos=np.arange(n) / max(1, n - 1), md_ps=(md - md[0]) / 5000.0,
            gr=np.where(np.isfinite(gr), (gr - 88.0) / 30.0, 0.0), gr_nan=(~np.isfinite(gr)).astype(float),
            dip=k * 10.0,
        )
        steps = np.unique(np.r_[np.arange(0, n, STRIDE), n - 1])
        out[nm] = dict(steps=steps, n=n, y=g["TVT"].to_numpy(), p_a=p_a, p_c=p_c,
                       X=np.nan_to_num(np.stack([full[f][steps] for f in FEATS], 1)).astype(np.float32))
    return out


# ------------------------------------------------------------------------------ gate


class Gate(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)          # w = 0.5 everywhere: starts as the plain mean

    def forward(self, X, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(X, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=X.shape[1])
        return torch.sigmoid(self.head(h))[..., 0]


def _batch(seqs, names, mu, sd, device):
    L = max(len(seqs[n]["steps"]) for n in names)
    F_ = len(FEATS)
    X = np.zeros((len(names), L, F_), np.float32)
    YA = np.zeros((len(names), L), np.float32)
    PA = np.zeros_like(YA); PC = np.zeros_like(YA); M = np.zeros_like(YA)
    lens = []
    for i, n in enumerate(names):
        s = seqs[n]; k = len(s["steps"])
        X[i, :k] = (s["X"] - mu) / sd
        YA[i, :k] = s["y"][s["steps"]]; PA[i, :k] = s["p_a"][s["steps"]]; PC[i, :k] = s["p_c"][s["steps"]]
        M[i, :k] = 1.0; lens.append(k)
    t = lambda a: torch.tensor(a, device=device)  # noqa: E731
    return t(X), t(YA), t(PA), t(PC), t(M), torch.tensor(lens)


def fit_gate(seqs, train_names, val_names, device, seed, ce_w=0.0, epochs=400, patience=40):
    # epochs/patience are passed through from the CLI so a slow variant can be bounded.
    """Fit one gate.  ``ce_w`` > 0 adds 3rd place's second loss term.

    Their gate is trained on ``MSE(blend) + CE(w, best candidate)``: the first term asks
    for the most accurate mixture, the second asks the weight to *commit* to whichever
    candidate is actually closer at that row.  With two candidates the CE is a binary
    cross-entropy on the weight itself.  It pushes the gate away from hedging near 0.5,
    which helps only if the gate can really tell the two apart -- so it is reported as a
    separate method rather than folded into the default.

    Early stopping always watches the pure validation MSE, because pooled RMSE is what
    the run is scored on; the CE term shapes training only.

    Note the scale: the MSE term is in ft^2 (order 20 here) and the CE term is order 0.7,
    so ``ce_w`` has to be of order 10 to matter.  Both terms are logged.
    """
    torch.manual_seed(seed)
    stack = np.concatenate([seqs[n]["X"] for n in train_names])
    mu, sd = stack.mean(0), stack.std(0) + 1e-6
    gate = Gate(len(FEATS)).to(device)

    # Start at the fitted constant weight, not at 0.5, and keep weight decay off the
    # output bias.  Initialised at 0.5 and decayed toward it, the first version of this
    # gate simply stayed there: it scored 5.017 against 4.905 for a single fitted weight,
    # even though that constant is inside its hypothesis space.  Starting from the
    # constant means the gate can only add per-row structure on top of it.
    a = np.concatenate([seqs[n]["p_a"] for n in train_names])
    c = np.concatenate([seqs[n]["p_c"] for n in train_names])
    y = np.concatenate([seqs[n]["y"] for n in train_names])
    w0 = float(np.clip(((y - c) * (a - c)).sum() / max(((a - c) ** 2).sum(), 1e-9), 0.02, 0.98))
    with torch.no_grad():
        gate.head.bias.fill_(float(np.log(w0 / (1 - w0))))
    decayed = [p for n, p in gate.named_parameters() if n != "head.bias"]
    opt = torch.optim.Adam([{"params": decayed, "weight_decay": 1e-4},
                            {"params": [gate.head.bias], "weight_decay": 0.0}], lr=3e-3)
    tr = _batch(seqs, train_names, mu, sd, device)
    va = _batch(seqs, val_names, mu, sd, device)

    def terms(batch):
        X, Y, PA, PC, M, L = batch
        w = gate(X, L)
        mse = (((w * PA + (1 - w) * PC - Y) ** 2) * M).sum() / M.sum()
        if ce_w <= 0:
            return mse, mse, torch.zeros((), device=mse.device)
        target = ((PA - Y).abs() < (PC - Y).abs()).to(w.dtype)   # 1 = anchor is closer
        bce = F.binary_cross_entropy(w.clamp(1e-6, 1 - 1e-6), target, reduction="none")
        bce = (bce * M).sum() / M.sum()
        return mse + ce_w * bce, mse, bce

    best, best_state, bad = float("inf"), None, 0
    last = (0.0, 0.0)
    for _ in range(epochs):
        gate.train()
        loss, mse, bce = terms(tr)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        opt.step()
        last = (float(mse.detach()), float(bce.detach()))
        gate.eval()
        with torch.no_grad():
            v = float(terms(va)[1])                              # pure MSE, always
        if v < best - 1e-6:
            best, best_state, bad = v, {k: x.detach().clone() for k, x in gate.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    gate.load_state_dict(best_state)
    return gate.eval(), mu, sd, last


@torch.no_grad()
def gate_weights(gate, mu, sd, seqs, names, device):
    X, _, _, _, _, L = _batch(seqs, names, mu, sd, device)
    W = gate(X, L).cpu().numpy()
    out = {}
    for i, n in enumerate(names):
        s = seqs[n]
        out[n] = np.interp(np.arange(s["n"]), s["steps"], W[i, :len(s["steps"])])
    return out


# ----------------------------------------------------------------------------- score


def pooled(seqs, preds, names):
    se = np.concatenate([(preds[n] - seqs[n]["y"]) ** 2 for n in names])
    return float(np.sqrt(se.mean()))


def summary(seqs, preds, names):
    well = pd.Series({n: float(np.sqrt(((preds[n] - seqs[n]["y"]) ** 2).mean())) for n in names})
    return dict(pooled_rmse=pooled(seqs, preds, names), well_rmse_p50=float(well.median()),
                well_rmse_mean=float(well.mean()), well_rmse_p95=float(well.quantile(0.95)))


def main(args):
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{out_dir} exists and is not empty; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "ensemble.log")
    log(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}")
    device = torch.device(args.device)

    df = load_pair(Path(args.anchor), Path(args.convnext), args.convnext_pred_col)
    names_all = well_names(args.data)
    _, ho_names = split_wells(names_all, 0.8)
    present = sorted(df["well_id"].unique())
    if args.limit_wells:
        present = present[:args.limit_wells]
        df = df[df["well_id"].isin(present)]
    wells = load_wells(Path(args.data), names_all, Path(args.cache) if args.cache else None, log=log)
    if not set(present) <= set(ho_names):
        raise SystemExit("prediction files contain wells outside the 20% holdout")
    seqs = well_sequences(df, wells)
    names = sorted(seqs)
    log(f"{len(names)} wells, {len(df):,} rows; anchor std feature "
        f"{'present' if (df['a_std'] != 0).any() else 'absent'}; "
        f"convnext neighbour feature {'present' if (df['c_nbr'] != 0).any() else 'absent'}")

    gate_cfgs = [("gate", 0.0)] + ([("gate_ce", args.gate_ce_w)] if args.gate_ce_w > 0 else [])
    preds = {m: {} for m in ["convnext", "anchor", "mean", "global_w", "oracle_row"]
             + [g for g, _ in gate_cfgs]}
    weights_w = {}
    weights_gate = {g: {} for g, _ in gate_cfgs}
    for n in names:
        s = seqs[n]
        preds["convnext"][n] = s["p_c"]
        preds["anchor"][n] = s["p_a"]
        preds["mean"][n] = 0.5 * (s["p_a"] + s["p_c"])
        preds["oracle_row"][n] = np.where(np.abs(s["p_a"] - s["y"]) < np.abs(s["p_c"] - s["y"]),
                                          s["p_a"], s["p_c"])

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(names)
    folds = [sorted(order[i::args.folds]) for i in range(args.folds)]
    for fi, test in enumerate(folds):
        train = sorted(set(names) - set(test))
        # closed-form weight: argmin_w sum((w a + (1-w) c - y)^2)
        a = np.concatenate([seqs[n]["p_a"] for n in train])
        c = np.concatenate([seqs[n]["p_c"] for n in train])
        y = np.concatenate([seqs[n]["y"] for n in train])
        w = float(np.clip(((y - c) * (a - c)).sum() / max(((a - c) ** 2).sum(), 1e-9), 0.0, 1.0))
        weights_w[fi] = w
        for n in test:
            preds["global_w"][n] = w * seqs[n]["p_a"] + (1 - w) * seqs[n]["p_c"]

        inner = np.random.default_rng(args.seed + fi).permutation(train)
        n_val = max(1, len(inner) // 5)
        val, fit = sorted(inner[:n_val]), sorted(inner[n_val:])
        log(f"fold {fi}: {len(test)} test / {len(fit)} fit / {len(val)} val wells  "
            f"global_w(anchor)={w:.3f}  "
            f"convnext {pooled(seqs, preds['convnext'], test):.3f}")
        for name, ce_w in gate_cfgs:
            W = {n: np.zeros(seqs[n]["n"]) for n in test}
            last = (0.0, 0.0)
            for gs in range(args.gate_seeds):
                gate, mu, sd, last = fit_gate(seqs, fit, val, device, ce_w=ce_w,
                                              seed=args.seed * 100 + fi * 10 + gs,
                                              epochs=args.gate_epochs, patience=args.gate_patience)
                for n, wn in gate_weights(gate, mu, sd, seqs, test, device).items():
                    W[n] += wn / args.gate_seeds
            for n in test:
                weights_gate[name][n] = W[n]
                preds[name][n] = W[n] * seqs[n]["p_a"] + (1 - W[n]) * seqs[n]["p_c"]
            flat = np.concatenate(list(W.values()))
            log(f"    {name:8s} ce_w={ce_w:<5g} mean w={flat.mean():.3f}  "
                f"decisiveness={2 * np.abs(flat - 0.5).mean():.3f}  "
                f"train MSE {last[0]:.3f} / CE {last[1]:.3f}  "
                f"fold RMSE {pooled(seqs, preds[name], test):.3f}")

    metrics = {m: summary(seqs, preds[m], names) for m in preds}
    metrics["global_w_per_fold"] = weights_w
    for name, _ in gate_cfgs:
        flat = np.concatenate([weights_gate[name][n] for n in names])
        metrics[name]["mean_weight_anchor"] = float(flat.mean())
        metrics[name]["decisiveness"] = float(2 * np.abs(flat - 0.5).mean())
    metrics["inputs"] = dict(anchor=str(args.anchor), convnext=str(args.convnext), folds=args.folds,
                             gate_seeds=args.gate_seeds, seed=args.seed, gate_ce_w=args.gate_ce_w)
    (out_dir / "ensemble_metrics.json").write_text(json.dumps(metrics, indent=2))

    base = df[["well_id", "submit_index", "TVT"]].reset_index(drop=True)
    for m in ["mean", "global_w"] + [g for g, _ in gate_cfgs]:
        sub = out_dir / m
        sub.mkdir(exist_ok=True)
        base.assign(TVT_pred=np.concatenate([preds[m][n] for n in names])).to_parquet(
            sub / "holdout_predictions.pqt", index=False)
    pd.DataFrame({"well_id": names,
                  **{f"rmse_{m}": [float(np.sqrt(((preds[m][n] - seqs[n]['y']) ** 2).mean())) for n in names]
                     for m in preds},
                  **{f"mean_w_anchor_{g}": [float(weights_gate[g][n].mean()) for n in names]
                     for g, _ in gate_cfgs}}
                 ).to_csv(out_dir / "ensemble_well_rmse.csv", index=False)

    log("")
    log(f"{'method':12s} {'pooled':>8s} {'median well':>12s} {'p95 well':>9s}  {'mean w':>7s} {'decisive':>9s}")
    for m in ["convnext", "anchor", "mean", "global_w"] + [g for g, _ in gate_cfgs] + ["oracle_row"]:
        r = metrics[m]
        extra = (f"  {r['mean_weight_anchor']:7.3f} {r['decisiveness']:9.3f}"
                 if "mean_weight_anchor" in r else "")
        note = "   (uses the answer; upper bound)" if m == "oracle_row" else ""
        log(f"{m:12s} {r['pooled_rmse']:8.4f} {r['well_rmse_p50']:12.3f} "
            f"{r['well_rmse_p95']:9.3f}{extra}{note}")
    return metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchor", required=True, help="AnchorCNN holdout_predictions.pqt (TVT_std optional)")
    p.add_argument("--convnext", required=True, help="1st-place holdout_predictions.pqt")
    p.add_argument("--convnext-pred-col", default="TVT_pred")
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(HERE / "data"))
    p.add_argument("--cache", default=str(HERE / "cache/wells.npz"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--gate-seeds", type=int, default=3)
    p.add_argument("--gate-epochs", type=int, default=400,
                   help="cap on gate fitting epochs. The gate_ce variant early-stops much later "
                        "than the plain gate, so this bounds its cost")
    p.add_argument("--gate-patience", type=int, default=40)
    p.add_argument("--gate-ce-w", type=float, default=0.0,
                   help="weight on 3rd place's extra term, a cross-entropy pushing the gate to "
                        "commit to whichever model is closer at that row. 0 = off (default, so "
                        "earlier runs reproduce). The MSE term is order 20 ft^2 and this one is "
                        "order 0.7, so values of order 10 are where it starts to bite. When > 0 a "
                        "second method 'gate_ce' is reported next to the plain gate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-wells", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    if not a.cache:
        a.cache = None
    return a


if __name__ == "__main__":
    main(parse_args())
