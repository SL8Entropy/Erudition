#!/usr/bin/env python3
"""8-bit quantization of a trained AnchorCNN, measured on the same holdout.

Two stages, both writing into ``--out``:

PTQ   post-training static quantization.  Takes minutes.  The trunk, decoder and heads
      are quantized with PyTorch's FX graph mode on the x86 backend (per-channel conv
      weights, histogram activation observers), calibrated on real training wells.
QAT   quantization-aware fine-tuning (``--qat-epochs N`` > 0).  Fake-quantization is
      inserted into the same graph and the model is fine-tuned from the checkpoint on
      the GPU with the original losses, then converted to int8.  This is the standard fix
      when PTQ loses accuracy, and it is the part that takes the hour.

Everything is scored the way the float model is: DP decode, columns mapped back to MD,
pooled RMSE over the 754,122 holdout rows, 1 phase and ``--tta`` phases.  Latency is
measured per well on the CPU for float32 and int8 at several thread counts, because int8
kernels only exist on the CPU in PyTorch and because a rig computer is a CPU.

Two things to know before reading the numbers.  First, EfficientNet-B0 uses SiLU, which
has no int8 kernel, so every block drops back to float around its activations; measured
on this machine the PTQ model is *slower* than float32 at 8 threads.  Second, the DP
decoder always stays in float -- only the network is quantized.

    python anchor_quantize.py --model runs/C_s1/model_last.pt --out runs/C_s1_int8 --row 1.0 --n-move 5 --tta 8
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import gr2tvt_data as gd                                                    # noqa: E402
from gr2tvt_model import anchor_loss, dp_expected_level, dzl_loss, dzl_target  # noqa: E402
from anchor_data import (AugCfg, AnchorWellDataset, build_item, collate,       # noqa: E402
                         eval_rows, load_wells, set_grid, split_wells, well_names)
from anchor_train import Logger, build_model, model_inputs, score           # noqa: E402

from torch.ao.quantization import (QConfigMapping, get_default_qat_qconfig_mapping,  # noqa: E402
                                   get_default_qconfig_mapping)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx  # noqa: E402


class StaticAnchorNet(nn.Module):
    """``GR2TVTAnchorNet.forward`` for the dzl_w1 configuration, with the shape baked in.

    FX symbolic tracing cannot unpack ``x.shape``, which the original forward does.  This
    is the same computation with ``tq`` and ``anchor_m`` fixed at construction; it is
    checked equal to the original (max abs difference 0.0) before anything is quantized.
    It returns the raw sub-bin logit rather than its sigmoid, and the dz_layer output.
    """

    def __init__(self, net, T: int, M: int):
        super().__init__()
        if net.n_head_ch or net.input_up != 1 or not net.dzl_head or net.dzl_film is None:
            raise ValueError("StaticAnchorNet supports the dzl_w1 configuration only")
        self.net, self.tq, self.am = net, T // 4, M

    def forward(self, x):
        n = self.net
        h = None
        for f, proj in zip(n.trunk(x), n.proj):
            f = F.interpolate(proj(f), size=(self.tq, self.am // n.fuse_div), mode="bilinear",
                              align_corners=False)
            h = f if h is None else h + f
        h = n.blocks(h)
        h = n.up_conv(F.interpolate(h, size=(self.tq, self.am), mode="bilinear", align_corners=False))
        d = n.dzl_proj(h.mean(dim=2))
        gamma, beta = n.dzl_film(d).chunk(2, dim=1)
        h = h * (1.0 + gamma.unsqueeze(2)) + beta.unsqueeze(2)
        return n.cls_head(h), n.b_head(h), d


def state_mb(module) -> float:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.tell() / 1e6


@torch.no_grad()
def predict(static, float_net, wells, names, phases, device, batch=4):
    """DP-decoded holdout predictions for a (possibly quantized) static module."""
    out, failed = {}, []
    for i in range(0, len(names), batch):
        chunk, acc = names[i:i + batch], {}
        for ph in phases:
            items, metas, keep = [], {}, []
            for nm in chunk:
                try:
                    o = build_item(wells[nm], float_net.ps_col, with_label=False, md_phase=float(ph))
                except (ValueError, IndexError):
                    continue
                items.append(o["item"]); metas[nm] = o["meta"]; keep.append(nm)
            if not items:
                continue
            x = model_inputs(items, device, float_net).contiguous()
            cls, b_raw, _ = static(x)
            lv = dp_expected_level(cls.float(), torch.sigmoid(b_raw.float())[:, 0], float_net.win,
                                   start_col=float_net.ps_col).cpu().numpy()
            for j, nm in enumerate(keep):
                ev = eval_rows(wells[nm])
                p = gd.cols_to_md(lv[j], metas[nm], wells[nm]["md"][ev].astype(np.float64))
                acc[nm] = p if nm not in acc else acc[nm] + p
        for nm in chunk:
            if nm in acc:
                out[nm] = acc[nm] / len(phases)
            else:
                failed.append(nm)
    return out, failed


@torch.no_grad()
def latency_ms(static, x, threads, reps=6):
    torch.set_num_threads(threads)
    for _ in range(2):
        static(x)
    t0 = time.perf_counter()
    for _ in range(reps):
        static(x)
    return (time.perf_counter() - t0) / reps * 1e3


def calibrate(prepared, float_net, wells, names, phases, device):
    with torch.no_grad():
        for nm in names:
            for ph in phases:
                try:
                    o = build_item(wells[nm], float_net.ps_col, with_label=False, md_phase=float(ph))
                except (ValueError, IndexError):
                    continue
                prepared(model_inputs([o["item"]], device, float_net).contiguous())


def qconfig_mapping(base: QConfigMapping, keep_float: str) -> QConfigMapping:
    """Optionally keep the most sensitive layers in float (mixed precision)."""
    if keep_float in ("ends", "ends+decoder"):
        base = base.set_module_name("net.trunk.conv_stem", None) \
                   .set_module_name("net.cls_head", None).set_module_name("net.b_head", None)
    if keep_float == "ends+decoder":
        for name in ("net.proj", "net.blocks", "net.up_conv", "net.dzl_proj", "net.dzl_film"):
            base = base.set_module_name(name, None)
    return base


def evaluate_block(tag, static, float_net, ho_wells, ho_names, phases, device, log):
    t0 = time.time()
    p1, f1 = predict(static, float_net, ho_wells, ho_names, (0.0,), device)
    m1, _ = score(ho_wells, p1, f1)
    res = dict(pooled_rmse_1phase=m1["pooled_rmse"], well_rmse_p50_1phase=m1["well_rmse_p50"])
    df = None
    if len(phases) > 1:
        pt, ft = predict(static, float_net, ho_wells, ho_names, phases, device)
        mt, df = score(ho_wells, pt, ft)
        res.update(pooled_rmse=mt["pooled_rmse"], well_rmse_p50=mt["well_rmse_p50"])
    log(f"  {tag:10s} 1-phase {res['pooled_rmse_1phase']:.4f} ft"
        + (f"   {len(phases)}-phase {res['pooled_rmse']:.4f} ft" if len(phases) > 1 else "")
        + f"   ({time.time() - t0:.0f}s)")
    return res, df


def main(args):
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{out_dir} exists and is not empty; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "quantize.log")
    log(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}")
    torch.backends.quantized.engine = "x86"
    cpu = torch.device("cpu")

    set_grid(row=args.row)
    names = well_names(args.data)
    if args.limit_wells:
        names = names[:args.limit_wells]
    tr_names, ho_names = split_wells(names, args.train_frac)
    wells = load_wells(Path(args.data), names, Path(args.cache) if args.cache else None, log=log)
    ho_wells = {n: wells[n] for n in ho_names}
    phases = tuple(np.arange(args.tta) * (gd.COLW / 4.0)) if args.tta > 1 else (0.0,)
    rng = np.random.default_rng(0)
    calib_names = list(rng.choice(tr_names, size=min(args.calib_wells, len(tr_names)), replace=False))
    log(f"holdout {len(ho_names)} wells, calibration {len(calib_names)} training wells x 2 phases, "
        f"eval phases {tuple(round(p, 1) for p in phases)}")

    margs = argparse.Namespace(arch="effnet", backbone="efficientnet_b0", pretrained=False,
                               backbone_weights="", cache=args.cache, stem_stride=1,
                               fuse_div=4, ps_col=16, n_move=args.n_move, win=128.0,
                               drop_path=0.0, dzl_w=1.0)
    net = build_model(margs, cpu, log=log).to(memory_format=torch.contiguous_format)
    sd = torch.load(args.model, map_location="cpu")
    net.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd)
    net.eval()
    static = StaticAnchorNet(net, gd.T, gd.H + net.ps_col).eval()

    probe = model_inputs([build_item(wells[ho_names[0]], net.ps_col, False)["item"]], cpu, net).contiguous()
    with torch.no_grad():
        ref, got = net(probe), static(probe)
    diff = max(float((ref[0] - got[0]).abs().max()),
               float((ref[1] - torch.sigmoid(got[1])[:, 0]).abs().max()))
    log(f"static graph matches the model: max |diff| = {diff:.2e}")
    if diff > 1e-4:
        raise SystemExit("static wrapper disagrees with the model; refusing to quantize")

    metrics = dict(model=str(args.model), row=args.row, n_move=args.n_move, tta=args.tta,
                   keep_float=args.keep_float, calib_wells=len(calib_names))
    x1 = probe[:1]

    log("float32 (CPU)")
    metrics["fp32"], _ = evaluate_block("fp32", static, net, ho_wells, ho_names, phases, cpu, log)
    metrics["fp32"]["size_mb"] = state_mb(static)

    log(f"PTQ int8, keep_float={args.keep_float}")
    prepared = prepare_fx(copy.deepcopy(static),
                          qconfig_mapping(get_default_qconfig_mapping("x86"), args.keep_float),
                          example_inputs=(x1,))
    calibrate(prepared, net, wells, calib_names, (0.0, 16.0), cpu)
    ptq = convert_fx(prepared)
    metrics["ptq"], df_ptq = evaluate_block("ptq int8", ptq, net, ho_wells, ho_names, phases, cpu, log)
    metrics["ptq"]["size_mb"] = state_mb(ptq)
    torch.save(ptq.state_dict(), out_dir / "ptq_int8_state_dict.pt")
    if df_ptq is not None:
        df_ptq.drop(columns=["sq_err"]).to_parquet(out_dir / "holdout_predictions_ptq.pqt", index=False)

    final = ("ptq", ptq)
    if args.qat_epochs > 0:
        final = ("qat", run_qat(args, static, net, wells, tr_names, x1, log))
        metrics["qat"], df_qat = evaluate_block("qat int8", final[1], net, ho_wells, ho_names,
                                                phases, cpu, log)
        metrics["qat"]["size_mb"] = state_mb(final[1])
        torch.save(final[1].state_dict(), out_dir / "qat_int8_state_dict.pt")
        if df_qat is not None:
            df_qat.drop(columns=["sq_err"]).to_parquet(out_dir / "holdout_predictions.pqt", index=False)
    elif df_ptq is not None:
        df_ptq.drop(columns=["sq_err"]).to_parquet(out_dir / "holdout_predictions.pqt", index=False)

    log("CPU latency per well (batch 1, one phase, network only)")
    metrics["latency_ms"] = {}
    for th in args.threads:
        fp = latency_ms(static, x1, th)
        q8 = latency_ms(final[1], x1, th)
        metrics["latency_ms"][str(th)] = dict(fp32=fp, int8=q8, speedup=fp / q8)
        log(f"  {th:2d} threads: fp32 {fp:7.1f} ms   int8 ({final[0]}) {q8:7.1f} ms   -> {fp / q8:.2f}x")

    (out_dir / "quantize_metrics.json").write_text(json.dumps(metrics, indent=2))
    log("")
    for k in ("fp32", "ptq", "qat"):
        if k in metrics:
            m = metrics[k]
            log(f"{k:5s} {m.get('pooled_rmse', m['pooled_rmse_1phase']):.4f} ft  "
                f"(1-phase {m['pooled_rmse_1phase']:.4f})  {m['size_mb']:.1f} MB")
    return metrics


def run_qat(args, static, net, wells, tr_names, x1, log):
    """Quantization-aware fine-tuning on the GPU, starting from the float checkpoint."""
    dev = torch.device(args.device)
    qat_src = copy.deepcopy(static).train()
    qat = prepare_qat_fx(qat_src, qconfig_mapping(get_default_qat_qconfig_mapping("x86"),
                                                  args.keep_float), example_inputs=(x1,))
    qat = qat.to(dev)
    net_dev = copy.deepcopy(net).to(dev)            # used only to assemble inputs
    ds = AnchorWellDataset({n: wells[n] for n in tr_names}, tr_names, AugCfg(), ps_col=net.ps_col,
                           seed=args.seed, epoch_len=args.qat_epoch_len or len(tr_names))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    opt = torch.optim.AdamW(qat.parameters(), lr=args.qat_lr, weight_decay=0.0)
    freeze_at = int(args.qat_epochs * 0.6)
    log(f"QAT: {args.qat_epochs} epochs x {len(dl)} batches on {dev}, lr {args.qat_lr}, "
        f"observers and BN statistics frozen from epoch {freeze_at}")
    for ep in range(args.qat_epochs):
        ds.set_epoch(ep)
        qat.train()
        if ep >= freeze_at:
            qat.apply(torch.ao.quantization.disable_observer)
            for mod in qat.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        t0, tot, n = time.time(), 0.0, 0
        for batch in dl:
            items = [b["item"] for b in batch]
            x = model_inputs(items, dev, net_dev).contiguous()
            y_bnd = torch.tensor(np.stack([it[6] for it in items]), device=dev)
            d_n = torch.tensor(np.stack([it[13] for it in items]), device=dev)
            cls, b_raw, d = qat(x)
            loss, _, _, _ = anchor_loss(cls.float(), torch.sigmoid(b_raw.float())[:, 0], y_bnd, win=net.win)
            tgt, valid = dzl_target(y_bnd, d_n, gd.COLW, gd.DZ_SLOPE_SD)
            l1, _ = dzl_loss(d[:, 0].float(), tgt, valid)
            loss = loss + l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); n += 1
        log(f"  qat epoch {ep:3d}/{args.qat_epochs}  loss {tot / max(1, n):.4f}  {time.time() - t0:.0f}s")
    qat = qat.cpu().eval()
    return convert_fx(qat)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(HERE / "data"))
    p.add_argument("--cache", default=str(HERE / "cache/wells.npz"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--limit-wells", type=int, default=0)
    p.add_argument("--row", type=float, default=1.0)
    p.add_argument("--n-move", type=int, default=5)
    p.add_argument("--tta", type=int, default=8)
    p.add_argument("--calib-wells", type=int, default=64)
    p.add_argument("--keep-float", choices=["none", "ends", "ends+decoder"], default="none",
                   help="layers left in float32: none, the input stem and output heads, or those "
                        "plus the whole decoder")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--qat-epochs", type=int, default=0, help="0 = PTQ only")
    p.add_argument("--qat-epoch-len", type=int, default=0)
    p.add_argument("--qat-lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--seed", type=int, default=46013)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    if not a.cache:
        a.cache = None
    return a


if __name__ == "__main__":
    main(parse_args())
