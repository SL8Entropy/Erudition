#!/usr/bin/env python3
"""Train one AnchorCNN on 80% of the wells and score the held-out 20%.

The model, the loss and the decoder are the author's, imported unmodified from ``src/``:
``GR2TVTAnchorNet`` (EfficientNet-B0 trunk, FPN-style fuse, 21-way move head plus a
sub-bin offset head and the auxiliary dz_layer head), ``anchor_loss`` (teacher-forced
cross-entropy at the anchor lying on the ground-truth path, plus the offset Huber) and
``dp_expected_level`` (exact marginalisation of the move field into an expected path).
Architecture defaults reproduce the released ``model/plan147/dzl_w1`` checkpoints exactly
-- 9 input channels, d=64, 2 fuse blocks, n_move=10, dzl_arch="linear", dzl_film="full"
-- as read back from ``ep119_seed46013_fold0.pt``.

Protocol
--------
Split      first 80% of the sorted well ids train (618), last 20% holdout (155).  This
           is the same rule and the same sort as the 1st-place repo's
           ``seq_NN_holdout_eval.py``, so the two solutions score identical wells.
Scored     every row of a holdout well whose ``TVT_input`` is NaN -- 754,122 rows.
Metric     pooled RMSE, one sqrt(mean((TVT - TVT_pred)**2)) over all of those rows, not
           a mean of per-well RMSEs.  Per-well figures are reported alongside because
           the pooled number is dominated by a handful of catastrophic wells.

Outputs land in ``--out`` and include ``holdout_predictions.pqt`` in the schema the
1st-place repo's ``seq_NN_robust_compare.py`` reads (well_id, submit_index, TVT,
TVT_pred), so a run here can be compared against a run there with that script.

Example (6 GB laptop GPU):

    python anchor_train.py --out runs/dzl_w1 --epochs 120 --tta 8 --score-best
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import gr2tvt_data as gd                                          # noqa: E402
from gr2tvt_model import (GR2TVTAnchorNet, anchor_loss, dzl_loss,  # noqa: E402
                          dzl_target, items_to_x)
from anchor_data import (AugCfg, AnchorWellDataset, build_item, collate,  # noqa: E402
                         eval_rows, grid_worker_init, load_wells, set_grid,
                         split_wells, well_names)


# ------------------------------------------------------------------------------ utils


class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


class EMA:
    """Exponential moving average of the weights, evaluated in place of the raw model.

    The decay is warmed up as ``min(decay, (1 + n) / (10 + n))``.  Without that, a
    constant 0.999 keeps 0.999^n of the *initial* weights, and since ``cls_head`` and
    ``b_head`` are zero-initialised the average stays a uniform move distribution --
    which decodes to a flat path at exactly the constant-TVT_PS baseline -- for the
    first few thousand steps.  Early epochs would report that, not the model.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.n = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(m.detach(), alpha=1.0 - d)
            else:
                s.copy_(m)


BACKBONE_URLS = {
    "efficientnet_b0": "https://huggingface.co/timm/efficientnet_b0.ra_in1k/resolve/main/model.safetensors",
    "efficientnet_b1": "https://huggingface.co/timm/efficientnet_b1.ra4_e3600_r240_in1k/resolve/main/model.safetensors",
}


def ensure_backbone_weights(args, log) -> Path | None:
    """Local ImageNet weights for the trunk, downloaded once with urllib.

    ``huggingface_hub`` (and anything else on ``requests``) fails TLS verification on
    this machine while ``urllib`` and ``curl`` to the very same host succeed, so the
    normal ``pretrained=True`` path cannot fetch anything.  Grabbing the file directly
    and handing it to timm through ``pretrained_cfg_overlay`` gives identical weights.
    """
    if not args.pretrained:
        return None
    if args.backbone_weights:
        p = Path(args.backbone_weights)
        if not p.exists():
            raise SystemExit(f"--backbone-weights {p} does not exist")
        return p
    url = BACKBONE_URLS.get(args.backbone)
    if url is None:
        return None                       # unknown backbone: let timm try the hub itself
    dest = Path(args.cache).parent / f"{args.backbone}.imagenet.safetensors" if args.cache \
        else HERE / "cache" / f"{args.backbone}.imagenet.safetensors"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    log(f"backbone: downloading {args.backbone} ImageNet weights -> {dest}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    log(f"backbone: {dest.stat().st_size/1e6:.1f} MB")
    return dest


def build_model(args, device, log=print):
    if getattr(args, "arch", "effnet") == "separable":
        from anchor_separable import SeparableAnchorNet
        model = SeparableAnchorNet(T=gd.T, M=gd.H + args.ps_col, ps_col=args.ps_col,
                                   win=args.win, n_move=args.n_move, fuse_div=args.fuse_div,
                                   dzl_head=args.dzl_w > 0)
        return model.to(device)

    from anchor_convnext import cached_weights, is_convnext, retrofit_stem
    cnx = is_convnext(args.backbone)
    cnx_file = cached_weights(args.backbone) if (cnx and args.pretrained) else None
    if cnx and args.pretrained and cnx_file is None:
        log(f"no cached weights for {args.backbone}; the hub is unreachable here, so the "
            f"trunk will be randomly initialised")

    kw = dict(backbone=args.backbone, in_chans=9, d=64, n_blocks=2,
              # make_trunk only allows stem_stride != 2 for EfficientNet; for ConvNeXt the
              # stem is retrofitted after construction instead (see anchor_convnext.py).
              stem_stride=2 if cnx else args.stem_stride,
              fuse_div=args.fuse_div, ps_col=args.ps_col,
              anchor_m=gd.H + args.ps_col, n_move=args.n_move,
              win=args.win, row=gd.ROW,
              ps_gr=True, z_dip=True, drop_path=args.drop_path,
              pretrained=args.pretrained)
    if args.dzl_w > 0:
        kw.update(dzl_head=True, dzl_arch="linear", dzl_film="full")

    weights = ensure_backbone_weights(args, log)
    import timm
    real_create = timm.create_model

    def create(name, **kwargs):
        # timm adapts the 3-channel stem to the 9 input channels for us, exactly as it
        # would when loading the same checkpoint from the hub.
        if weights is not None and name == args.backbone and kwargs.get("pretrained"):
            kwargs.setdefault("pretrained_cfg_overlay", dict(file=str(weights)))
        if cnx_file is not None and name == args.backbone and kwargs.get("pretrained"):
            kwargs.setdefault("pretrained_cfg_overlay", dict(file=str(cnx_file)))
        return real_create(name, **kwargs)

    timm.create_model = create
    try:
        model = GR2TVTAnchorNet(**kw)
    except Exception as exc:                       # pragma: no cover - network dependent
        if not args.pretrained:
            raise
        log(f"pretrained weights unavailable ({type(exc).__name__}: {exc}); "
            f"falling back to random init")
        model = GR2TVTAnchorNet(**dict(kw, pretrained=False))
    finally:
        timm.create_model = real_create
    if cnx:
        retrofit_stem(model, in_chans=9, stride=args.convnext_stem_stride, log=log)
    return model.to(device).to(memory_format=torch.channels_last)


def lr_at(step: int, total: int, args) -> float:
    """WSD with no decay: linear warmup, then a constant rate, as in the writeup."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    if args.final_decay <= 0:
        return args.lr
    start = int(total * (1.0 - args.final_decay))
    if step < start:
        return args.lr
    t = (step - start) / max(1, total - start)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))


def model_inputs(items, device, model):
    """Whatever this model consumes: the 9-channel grid, or the 1-D arrays themselves."""
    if getattr(model, "takes_items", False):
        return model.inputs_from_items(items, device)
    x, _ = items_to_x(items, device, model)
    return x.to(memory_format=torch.channels_last)


def load_teachers(paths, args, device, log=print):
    """Frozen teacher models for distillation; they share the student's geometry."""
    import copy as _copy
    targs = _copy.copy(args)
    targs.pretrained = False
    targs.arch = getattr(args, "teacher_arch", "effnet")
    teachers = []
    for p in paths:
        t = build_model(targs, device, log=log)
        sd = torch.load(p, map_location=device)
        t.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd)
        t.eval()
        for q in t.parameters():
            q.requires_grad_(False)
        teachers.append(t)
    return teachers


def distill_loss(student_logits, teacher_probs, temp, ncol, t_cover, ps_col):
    """KL(teacher || student) over the whole move field, not just the true path.

    Teacher-forced cross-entropy only ever supervises the one anchor lying on the true
    path.  The ensemble's field also says how the well would move *if* it were on a
    different layer, which is exactly what the DP marginalises over when a match is
    ambiguous; distilling every covered anchor is what transfers that.  Columns before
    the prediction start and beyond the well, and rows outside typewell coverage, are
    masked out.
    """
    B, V, Tq, M = student_logits.shape
    logp = torch.log_softmax(student_logits.float() / temp, dim=1)
    kl = (teacher_probs * (torch.log(teacher_probs.clamp_min(1e-8)) - logp)).sum(1)
    cols = torch.arange(M, device=kl.device)[None, :]
    colmask = (cols >= ps_col) & (cols < ncol[:, None])
    rowmask = t_cover.view(B, Tq, -1).mean(-1) > 0.5
    mask = (rowmask[:, :, None] & colmask[:, None, :]).float()
    return (kl * mask).sum() / mask.sum().clamp_min(1.0) * temp * temp


# ------------------------------------------------------------------------- evaluation


@torch.no_grad()
def predict_holdout(model, wells, names, device, args, phases=(0.0,), amp_dtype=None,
                    moments=None):
    """DP-decoded TVT for every scored row of every holdout well.

    Each MD phase is a separate build of the input grid, because re-phasing the 32 ft
    column boundaries changes the box-averaged GR itself.  Predictions are averaged
    after mapping back to raw MD, where all phases share one axis.

    If ``moments`` is a dict, it is filled with ``well -> E[level^2]`` averaged over the
    phases (level relative to TVT_PS), from which the spread of the belief follows.
    """
    from anchor_uncertainty import dp_moments
    model.eval()
    out, failed = {}, []
    bs = args.val_batch_size
    for i in range(0, len(names), bs):
        chunk = names[i:i + bs]
        acc, acc2 = {}, {}
        for ph in phases:
            items, metas, keep = [], {}, []
            for nm in chunk:
                try:
                    o = build_item(wells[nm], args.ps_col, with_label=False, md_phase=float(ph))
                except (ValueError, IndexError):
                    continue
                items.append(o["item"])
                metas[nm] = o["meta"]
                keep.append(nm)
            if not items:
                continue
            x = model_inputs(items, device, model)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                if moments is None:
                    lv = model.predict_level(x)
                else:
                    fwd = model(x)
            if moments is None:
                lv = lv.float().cpu().numpy()
            else:
                e1, e2 = dp_moments(fwd[0].float(), fwd[1].float(), model.win,
                                    start_col=model.ps_col)
                lv, lv2 = e1.cpu().numpy(), e2.cpu().numpy()
            for j, nm in enumerate(keep):
                ev = eval_rows(wells[nm])
                mdq = wells[nm]["md"][ev].astype(np.float64)
                p = gd.cols_to_md(lv[j], metas[nm], mdq)
                acc[nm] = p if nm not in acc else acc[nm] + p
                if moments is not None:
                    m_ = metas[nm]
                    xc = gd.col_centers(m_)
                    q = np.interp(mdq, xc, lv2[j][m_["ps_col"]: m_["ps_col"] + m_["ncol"]])
                    acc2[nm] = q if nm not in acc2 else acc2[nm] + q
        for nm in chunk:
            if nm in acc:
                out[nm] = acc[nm] / len(phases)
                if moments is not None:
                    moments[nm] = acc2[nm] / len(phases)
            else:
                failed.append(nm)
    return out, failed


def score(wells, preds, failed, log=print) -> tuple[dict, pd.DataFrame]:
    """Pooled RMSE over every scored row, plus the per-well breakdown.

    A well the grid could not be built for falls back to holding the last known
    TVT_input constant, so it is still scored rather than silently dropped.
    """
    rows = []
    for nm, w in wells.items():
        ev = eval_rows(w)
        if len(ev) == 0:
            continue
        true = w["tvt"][ev].astype(np.float64)
        if nm in preds:
            pred = np.asarray(preds[nm], dtype=np.float64)
        else:
            ti = w["tvt_input"][np.isfinite(w["tvt_input"])]
            pred = np.full(len(ev), float(ti[-1]) if len(ti) else float(np.nanmedian(true)))
        rows.append(pd.DataFrame(dict(well_id=nm, submit_index=ev, TVT=true, TVT_pred=pred)))
    df = pd.concat(rows, ignore_index=True).sort_values(
        ["well_id", "submit_index"], kind="stable").reset_index(drop=True)

    sq = (df["TVT"].to_numpy() - df["TVT_pred"].to_numpy()) ** 2
    pooled = float(np.sqrt(sq.mean()))
    per = pd.DataFrame(dict(well_id=df["well_id"], sq=sq)).groupby("well_id")["sq"]
    well = pd.DataFrame(dict(rmse=np.sqrt(per.mean()), sse=per.sum(), rows=per.size()))
    well = well.sort_values("sse", ascending=False)
    worst = well.head(5)
    metrics = dict(
        pooled_rmse=pooled, n_wells=int(df["well_id"].nunique()), n_rows=int(len(df)),
        well_rmse_mean=float(well["rmse"].mean()), well_rmse_p50=float(well["rmse"].median()),
        well_rmse_p95=float(well["rmse"].quantile(0.95)),
        sse_share_top5=float(worst["sse"].sum() / well["sse"].sum()),
        worst_wells={k: round(float(v), 3) for k, v in worst["rmse"].items()},
        failed_wells=list(failed),
    )
    return metrics, df.assign(sq_err=sq)


# ---------------------------------------------------------------------------- training


def train(args):
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{out_dir} exists and is not empty; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "train.log")
    log(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}")

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    grid = set_grid(row=args.row, gr_prefilter_ft=args.gr_prefilter_ft)
    log(f"grid: {gd.T} rows x {gd.H + args.ps_col} cols "
        f"(row {gd.ROW} ft, colw {gd.COLW} ft), state grid {gd.T // 4} bins of "
        f"{2 * args.win / (gd.T // 4):.1f} ft, stem_stride {args.stem_stride}, "
        f"arch {args.arch}, GR prefilter {args.gr_prefilter_ft:g} ft")

    names = well_names(args.data)
    if args.limit_wells:
        names = names[:args.limit_wells]
    tr_names, ho_names = split_wells(names, args.train_frac)
    log(f"wells: {len(names)} total -> {len(tr_names)} train / {len(ho_names)} holdout "
        f"(first holdout {ho_names[0]})")
    pd.DataFrame(dict(well_id=tr_names + ho_names,
                      split=["train"] * len(tr_names) + ["holdout"] * len(ho_names))
                 ).to_csv(out_dir / "holdout_split.csv", index=False)

    cache = Path(args.cache) if args.cache else None
    wells = load_wells(Path(args.data), names, cache, log=log)
    if args.sibling_w > 0:
        from anchor_sibling import apply_to_wells
        apply_to_wells(wells, tr_names, names, args.sibling_w, args.sibling_bin, log=log)
    tr_wells = {n: wells[n] for n in tr_names}
    ho_wells = {n: wells[n] for n in ho_names}
    log(f"scored rows in holdout: {sum(len(eval_rows(w)) for w in ho_wells.values()):,}")

    aug = AugCfg(crop_prob=args.crop_prob, md_flip_prob=args.md_flip,
                 level_flip_prob=args.level_flip, level_shift_ft=args.level_shift,
                 md_phase=not args.no_md_phase, specaug_prob=args.specaug,
                 synth_prob=args.synth_prob)
    ds = AnchorWellDataset(tr_wells, tr_names, aug, ps_col=args.ps_col, seed=args.seed,
                           epoch_len=args.epoch_len or len(tr_names),
                           pre_ps_label=args.pre_ps_label)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate, drop_last=True, pin_memory=False,
                    persistent_workers=args.num_workers > 0,
                    worker_init_fn=grid_worker_init(grid) if args.num_workers else None)

    model = build_model(args, device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"model: {type(model).__name__} {n_par/1e6:.2f}M params, "
        f"n_move={args.n_move}, dzl_head={args.dzl_w > 0}, "
        f"pretrained={args.pretrained and args.arch == 'effnet'}")
    teachers = load_teachers(args.distill_teachers, args, device, log) if args.distill_teachers else []
    if teachers:
        log(f"distillation: {len(teachers)} frozen teacher(s), weight {args.distill_w}, "
            f"temperature {args.distill_t}: {args.distill_teachers}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]
    scaler = torch.amp.GradScaler(enabled=amp_dtype is torch.float16)

    steps_per_epoch = max(1, len(dl) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    log(f"schedule: {args.epochs} epochs x {steps_per_epoch} steps "
        f"(batch {args.batch_size} x accum {args.grad_accum}), lr {args.lr}, "
        f"warmup {args.warmup_steps}")

    history, best = [], dict(pooled_rmse=float("inf"), epoch=-1)
    step = 0
    for epoch in range(args.epochs):
        ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        agg = dict(loss=0.0, ce=0.0, hb=0.0, oov=0.0, dzl=0.0, kd=0.0, n=0)
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(dl):
            items = [b["item"] for b in batch]
            x = model_inputs(items, device, model)
            y_bnd = torch.tensor(np.stack([it[6] for it in items]), device=device)
            d_n = torch.tensor(np.stack([it[13] for it in items]), device=device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                fwd = model(x, dzl=args.dzl_w > 0)
            cls_logits, b_pred = fwd[0].float(), fwd[1].float()
            loss, ce, hb, oov = anchor_loss(cls_logits, b_pred, y_bnd, win=args.win,
                                            huber_w=args.huber_w, soft=args.soft_ce)
            l1 = torch.zeros((), device=device)
            if args.dzl_w > 0:
                tgt, valid = dzl_target(y_bnd, d_n, gd.COLW, gd.DZ_SLOPE_SD)
                l1, tv = dzl_loss(fwd[2].float(), tgt, valid, tv_w=args.dzl_tv)
                loss = loss + args.dzl_w * l1 + args.dzl_tv * tv
            kd = torch.zeros((), device=device)
            if teachers:
                same_inputs = (getattr(model, "takes_items", False)
                               == getattr(teachers[0], "takes_items", False))
                tx = x if same_inputs else model_inputs(items, device, teachers[0])
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype,
                                                     enabled=amp_dtype is not None):
                    t_probs = sum(torch.softmax(t(tx)[0].float() / args.distill_t, dim=1)
                                  for t in teachers) / len(teachers)
                ncol = torch.tensor([it[5] for it in items], device=device)
                t_cover = torch.tensor(np.stack([it[3] for it in items]), device=device)
                kd = distill_loss(cls_logits, t_probs, args.distill_t, ncol, t_cover, args.ps_col)
                loss = loss + args.distill_w * kd

            scaler.scale(loss / args.grad_accum).backward()
            for k, v in (("loss", loss), ("ce", ce), ("hb", hb), ("oov", oov), ("dzl", l1),
                         ("kd", kd)):
                agg[k] += float(v.detach())
            agg["n"] += 1

            if (bi + 1) % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, total_steps, args)
                if args.clip_grad > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                step += 1

        n = max(1, agg["n"])
        line = (f"epoch {epoch:3d}/{args.epochs}  loss {agg['loss']/n:.4f}  "
                f"ce {agg['ce']/n:.4f}  huber {agg['hb']/n:.4f}  dzl {agg['dzl']/n:.4f}  "
                + (f"kd {agg['kd']/n:.4f}  " if teachers else "")
                + f"oov {agg['oov']/n:.5f}  lr {lr_at(step, total_steps, args):.2e}  "
                f"{time.time()-t0:.0f}s")

        last = epoch == args.epochs - 1
        if last or (args.eval_every and (epoch + 1) % args.eval_every == 0):
            net = ema.shadow if ema is not None else model
            preds, failed = predict_holdout(net, ho_wells, ho_names, device, args,
                                            phases=(0.0,), amp_dtype=amp_dtype)
            m, _ = score(ho_wells, preds, failed)
            line += f"  | holdout pooled RMSE {m['pooled_rmse']:.4f} ft"
            history.append(dict(epoch=epoch, **{k: v for k, v in m.items()
                                                if isinstance(v, (int, float))}))
            if m["pooled_rmse"] < best["pooled_rmse"]:
                best = dict(pooled_rmse=m["pooled_rmse"], epoch=epoch)
                torch.save(net.state_dict(), out_dir / "model_best.pt")
            model.train()
        log(line)
        torch.save((ema.shadow if ema is not None else model).state_dict(),
                   out_dir / "model_last.pt")

    # ------------------------------------------------------------------ final scoring
    net = ema.shadow if ema is not None else model
    if args.score_best and (out_dir / "model_best.pt").exists():
        net.load_state_dict(torch.load(out_dir / "model_best.pt", map_location=device))
        log(f"final scoring uses model_best.pt (epoch {best['epoch']})")
    # 8 ft steps, i.e. the author's [0, 8, ..., 56]: four sub-column phases, each also
    # seen with the window aligned one column earlier.  Not COLW/tta -- that would sample
    # phase more finely but never move the window.
    phases = tuple(np.arange(args.tta) * (gd.COLW / 4.0)) if args.tta > 1 else (0.0,)
    log(f"final holdout pass: {len(phases)} MD phase(s) {tuple(round(p, 1) for p in phases)}")
    preds, failed = predict_holdout(net, ho_wells, ho_names, device, args,
                                    phases=phases, amp_dtype=amp_dtype)
    metrics, df = score(ho_wells, preds, failed)
    # Always also score one phase: it is what a single-pass deployment pays for, and the
    # gap between the two is the TTA gain that the anti-aliasing experiment tries to absorb.
    if len(phases) > 1:
        p1, f1 = predict_holdout(net, ho_wells, ho_names, device, args,
                                 phases=(0.0,), amp_dtype=amp_dtype)
        m1, _ = score(ho_wells, p1, f1)
        metrics.update(pooled_rmse_1phase=m1["pooled_rmse"],
                       tta_gain=m1["pooled_rmse"] - metrics["pooled_rmse"])
    metrics.update(epochs=args.epochs, tta=args.tta, best_epoch=best["epoch"],
                   best_epoch_rmse=best["pooled_rmse"], scored="best" if args.score_best else "last",
                   arch=args.arch, row=args.row, n_move=args.n_move, seed=args.seed,
                   epoch_len=args.epoch_len or len(tr_names),
                   gr_prefilter_ft=args.gr_prefilter_ft,
                   distill_teachers=list(args.distill_teachers or []),
                   params_m=round(n_par / 1e6, 3))

    df.drop(columns=["sq_err"]).to_parquet(out_dir / "holdout_predictions.pqt", index=False)
    per = df.groupby("well_id")["sq_err"].agg(["mean", "sum", "size"])
    per = per.assign(rmse=np.sqrt(per["mean"])).rename(columns={"sum": "sse", "size": "rows"})
    per[["rmse", "sse", "rows"]].sort_values("sse", ascending=False).to_csv(
        out_dir / "holdout_well_rmse.csv")
    (out_dir / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    log("")
    log(f"HOLDOUT POOLED RMSE: {metrics['pooled_rmse']:.4f} ft "
        f"over {metrics['n_rows']:,} rows in {metrics['n_wells']} wells")
    if "pooled_rmse_1phase" in metrics:
        log(f"single phase: {metrics['pooled_rmse_1phase']:.4f} ft  "
            f"(TTA gain {metrics['tta_gain']:+.4f} ft)")
    log(f"per-well RMSE  mean {metrics['well_rmse_mean']:.3f}  "
        f"p50 {metrics['well_rmse_p50']:.3f}  p95 {metrics['well_rmse_p95']:.3f}")
    log(f"worst 5 wells carry {metrics['sse_share_top5']*100:.0f}% of the squared error: "
        f"{metrics['worst_wells']}")
    if failed:
        log(f"WARNING: {len(failed)} well(s) fell back to constant TVT: {failed}")
    return metrics


# --------------------------------------------------------------------------------- cli


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE / "data"), help="directory of *__horizontal_well.csv")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--cache", default=str(HERE / "cache/wells.npz"), help="'' disables the cache")
    p.add_argument("--force", action="store_true", help="overwrite a non-empty --out")

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--limit-wells", type=int, default=0, help="use only the first N wells (smoke test)")

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--val-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epoch-len", type=int, default=0, help="samples per epoch (default: n train wells)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--final-decay", type=float, default=0.0,
                   help="fraction of training spent on a cosine decay; 0 = constant (WSD)")
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999, help="0 disables EMA")
    p.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    p.add_argument("--seed", type=int, default=46013)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--backbone-weights", default="",
                   help="local ImageNet weights for the trunk; downloaded to --cache's "
                        "directory on first use if omitted")
    p.add_argument("--ps-col", type=int, default=16)
    p.add_argument("--fuse-div", type=int, default=4)
    p.add_argument("--n-move", type=int, default=10)
    p.add_argument("--win", type=float, default=128.0)
    p.add_argument("--drop-path", type=float, default=0.0)
    # resolution ablation.  Defaults reproduce the released dzl_w1 geometry exactly.
    p.add_argument("--stem-stride", type=int, default=1, choices=[1, 2],
                   help="1 = the released config (stage 0 at full input resolution); "
                        "2 = the stock EfficientNet stem, ~2.1x fewer FLOPs")
    p.add_argument("--row", type=float, default=0.5,
                   help="vertical grid sampling in ft; also sets the output bin size "
                        "(0.5 -> 512 rows / 2 ft bins, 1.0 -> 256 rows / 4 ft bins)")
    # cost experiments.  Every default reproduces the runs already in runs/.
    p.add_argument("--convnext-stem-stride", type=int, default=2,
                   help="replacement stem stride for a ConvNeXt backbone. 2 reproduces the "
                        "EfficientNet baseline's feature-map geometry (2:1 against the output "
                        "grid); ConvNeXt's own stride-4 patchify would give 1:1, which failed "
                        "in the resolution ablation")
    p.add_argument("--arch", choices=["effnet", "separable"], default="effnet",
                   help="effnet = the released EfficientNet-B0 trunk; separable = 1-D "
                        "encoders + matching volume (anchor_separable.py)")
    p.add_argument("--sibling-w", type=float, default=0.0,
                   help="blend the typewell channel with a sibling-lateral reference GR at this "
                        "weight (0 = typewell only, the default).  Siblings are TRAINING wells "
                        "sharing a master typewell system; a well is never its own sibling.")
    p.add_argument("--sibling-bin", type=float, default=1.0,
                   help="depth bin for the sibling GR profile, feet")
    p.add_argument("--gr-prefilter-ft", type=float, default=0.0,
                   help="anti-aliasing moving average on GR before column binning; "
                        "32 matches the column width. 0 = off")
    p.add_argument("--distill-teachers", nargs="*", default=[],
                   help="frozen teacher checkpoints with this run's geometry; their mean "
                        "move field is distilled into the student")
    p.add_argument("--teacher-arch", choices=["effnet", "separable"], default="effnet")
    p.add_argument("--distill-w", type=float, default=1.0)
    p.add_argument("--distill-t", type=float, default=1.0)

    p.add_argument("--huber-w", type=float, default=1.0)
    p.add_argument("--soft-ce", action="store_true", help="interpolate the CE target between bins")
    p.add_argument("--dzl-w", type=float, default=1.0, help="auxiliary dz_layer head weight; 0 removes it")
    p.add_argument("--dzl-tv", type=float, default=0.0)
    p.add_argument("--pre-ps-label", action="store_true", help="also supervise the known columns")

    p.add_argument("--crop-prob", type=float, default=1.0)
    p.add_argument("--md-flip", type=float, default=0.5)
    p.add_argument("--level-flip", type=float, default=0.5)
    p.add_argument("--level-shift", type=float, default=32.0)
    p.add_argument("--no-md-phase", action="store_true")
    p.add_argument("--specaug", type=float, default=0.5)
    p.add_argument("--synth-prob", type=float, default=0.0,
                   help="partial synthetic-well generator; see anchor_data.synth_well")

    p.add_argument("--eval-every", type=int, default=5, help="0 evaluates only at the end")
    p.add_argument("--tta", type=int, default=1, help="MD phases averaged in the final pass")
    p.add_argument("--score-best", action="store_true",
                   help="score the best checkpoint instead of the last")
    a = p.parse_args(argv)
    if not a.cache:
        a.cache = None
    return a


if __name__ == "__main__":
    train(parse_args())
