"""Training: dataset, loop, and out-of-fold prediction.

Fold safety is enforced structurally. One `FeatureProvider` is built per fold
with `allowed_full` set to that fold's *training* wells, so the XY-neighbour
cloud and the sibling GR bank can never see a validation label -- and a query
well never sees its own. Without that, the CV is fiction: the label re-enters
through a channel.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augment import augment_canvas, augment_well
from .canvas import build_canvas
from .data import group_kfold, save_json
from .decode import predict_well
from .features import FeatureProvider
from .losses import total_loss
from .metrics import score_report
from .model import EMA, build_model, count_params
from .synth import build_path_bank, synth_well


# ---------------------------------------------------------------------- #
class CanvasDataset(Dataset):
    """One item = one canvas. Length is n_wells * (1 + synth_per_epoch): the
    extra items are forced-synthetic, which is 1st place's joint training of
    simulated and real data rather than a separate pretraining stage."""

    def __init__(self, wells, cfg, provider, train: bool, bank=None,
                 epoch: int = 0):
        self.wells = wells
        self.cfg = cfg
        self.spec = cfg.canvas_spec()
        self.provider = provider
        self.train = train
        self.bank = bank
        self.epoch = epoch
        self.n_real = len(wells)
        self.n_synth = int(round(len(wells) * cfg.synth_per_epoch)) if train else 0

    def __len__(self):
        return self.n_real + self.n_synth

    def set_epoch(self, e: int):
        self.epoch = e

    def __getitem__(self, i: int):
        forced_synth = i >= self.n_real
        idx = i % self.n_real
        w = self.wells[idx]
        rng = np.random.default_rng(
            (hash(w.well_id) ^ (self.epoch * 7919) ^ (i * 104729)) & 0x7FFFFFFF)

        if self.train:
            if forced_synth and self.bank is not None:
                w = synth_well(w.copy(), self.bank, rng)
                w.meta["synthetic"] = True
            w = augment_well(w, self.cfg.aug, rng, self.bank if not forced_synth else None)

        spec = self.spec
        col_md, row_tvt = _grid(w, spec)
        extras = self.provider.extras(w, col_md, row_tvt, w.anchor_tvt,
                                      cache=not self.train) if self.provider else {}
        tw = self.provider.typewell_gr(w, cache=not self.train) if self.provider else None
        img, meta = build_canvas(w, spec, extras=extras, tw_gr_override=tw)

        if self.train:
            img = augment_canvas(img, spec.channels, self.cfg.aug, rng)

        y = np.nan_to_num(meta.y_col).astype(np.float32)
        valid = meta.y_valid & meta.col_valid
        # A column whose true TVT lies outside the canvas cannot be represented:
        # the smoothed CE target would pile all its mass on the edge row and
        # teach the model to predict the boundary. This happens where the
        # visible window reaches back into the build section, and after level
        # augmentation, so drop those columns from the loss rather than lie.
        limit = spec.n_rows * spec.row_ft / 2.0 - 2.0 * spec.row_ft
        valid = valid & (np.abs(meta.y_col) <= limit)
        cw = np.where(meta.col_known, self.cfg.vis_col_weight, 1.0).astype(np.float32)
        return dict(
            img=torch.from_numpy(img),
            y_col=torch.from_numpy(y),
            y_valid=torch.from_numpy(valid),
            col_weight=torch.from_numpy(cw),
            gr_gap=torch.from_numpy(meta.gr_gap),
            dip=torch.from_numpy(meta.extras["dip"].astype(np.float32)),
        )


def _grid(well, spec, md_phase: float = 0.0):
    md0 = well.anchor_md + md_phase - spec.n_vis_cols * spec.md_stride
    edges = md0 + np.arange(spec.n_cols + 1) * spec.md_stride
    col_md = 0.5 * (edges[:-1] + edges[1:])
    row_tvt = (np.arange(spec.n_rows) - (spec.n_rows - 1) / 2.0) * spec.row_ft
    return col_md, row_tvt


# ---------------------------------------------------------------------- #
def _cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def train_fold(cfg, wells, fold: int, seed: int, split: int, out_dir: Path,
               device: str = "cuda", verbose: bool = True):
    """Train one (split, seed, fold) model and return its OOF predictions.

    The checkpoint written at the end is the trained model -- what `infer`,
    `oof` and the benchmark load. Re-running trains from scratch and overwrites
    it, with a warning first.
    """
    torch.manual_seed(seed * 1000 + fold)
    np.random.seed(seed * 1000 + fold)

    assign = group_kfold([w.well_id for w in wells], cfg.n_folds, seed=split)
    tr = [w for w in wells if assign[w.well_id] != fold]
    va = [w for w in wells if assign[w.well_id] == fold]
    allowed = {w.well_id for w in tr}

    provider = FeatureProvider(
        wells, allowed_full=allowed, use_pf=cfg.use_pf, use_xy=cfg.use_xy,
        use_sibling=cfg.use_sibling, correct_typewell=cfg.correct_typewell)
    bank = build_path_bank(tr)

    spec = cfg.canvas_spec()
    out_dir = Path(out_dir)
    tag = f"split{split}_seed{seed}_fold{fold}"
    ck_path = out_dir / f"{tag}.pt"
    if ck_path.exists() and verbose:
        done = int(torch.load(ck_path, map_location="cpu",
                              weights_only=False).get("epochs_done", 0))
        print(f"[train] {tag}: overwriting an existing checkpoint "
              f"({done} epochs) -- training from scratch")

    model = build_model(cfg, spec.in_chans).to(device)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    amp_dtype = torch.bfloat16 if cfg.amp == "bf16" else torch.float16

    ds = CanvasDataset(tr, cfg, provider, train=True, bank=bank)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, drop_last=True,
                    pin_memory=(device != "cpu"),
                    persistent_workers=cfg.num_workers > 0)
    steps_per_epoch = max(len(dl) // cfg.accum_steps, 1)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(total_steps * cfg.warmup_frac)

    row_tvt = torch.from_numpy(
        (np.arange(spec.n_rows) - (spec.n_rows - 1) / 2.0) * spec.row_ft
    ).float().to(device)

    if verbose:
        print(f"[train] split{split} seed{seed} fold{fold}: "
              f"{len(tr)} train / {len(va)} val wells, "
              f"{spec.in_chans} channels, {count_params(model)} params")

    step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        ds.set_epoch(epoch)
        model.train()
        run = {}
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                enabled=(device != "cpu")):
                out = model(batch["img"])
            out = {k: v.float() for k, v in out.items()}
            loss, parts = total_loss(out, batch, cfg, row_tvt)
            (loss / cfg.accum_steps).backward()

            if (it + 1) % cfg.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                for g in opt.param_groups:
                    g["lr"] = _cosine_lr(step, total_steps, warmup, cfg.lr)
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
                step += 1
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
        if verbose:
            msg = "  ".join(f"{k}={v / max(len(dl), 1):.4f}" for k, v in run.items())
            print(f"  ep{epoch + 1:02d}/{cfg.epochs}  {msg}  "
                  f"({time.time() - t0:.0f}s)")

    # ---- OOF with the EMA weights ---------------------------------- #
    eval_model = build_model(cfg, spec.in_chans, pretrained=False).to(device)
    eval_model.load_state_dict(ema.state_dict())
    eval_model.eval()

    oof = predict_wells(eval_model, va, cfg, provider, device,
                        n_phases=cfg.n_phases, reanchor=cfg.reanchor,
                        adaptive_canvas=cfg.adaptive_canvas,
                        dp_weight=cfg.dp_weight)
    rep = score_report({k: v["pred"] for k, v in oof.items()}, va)
    if verbose:
        print(f"[train] fold OOF pooled RMSE = {rep['pooled_rmse']:.4f}  "
              f"(well-mean {rep['well_mean_rmse']:.4f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=ema.state_dict(), in_chans=spec.in_chans,
                    variant=cfg.name, tag=tag, report=rep,
                    epochs_done=cfg.epochs),
               ck_path)
    np.savez_compressed(
        out_dir / f"oof_{tag}.npz",
        **{f"pred__{k}": v["pred"] for k, v in oof.items()},
        **{f"sigma__{k}": (v["sigma"] if v["sigma"] is not None
                           else np.zeros(1)) for k, v in oof.items()})
    save_json(rep, out_dir / f"report_{tag}.json")
    # records which variant these weights are, so a run directory can never be
    # handed to a model with different heads
    save_json(dict(variant=cfg.name, in_chans=spec.in_chans,
                   n_rows=cfg.n_rows, row_ft=cfg.row_ft,
                   md_stride=cfg.md_stride, n_vis_cols=cfg.n_vis_cols,
                   n_tgt_cols=cfg.n_tgt_cols, backbone=cfg.backbone,
                   head_dz=cfg.head_dz, head_move=cfg.head_move,
                   head_sigma=cfg.head_sigma),
              out_dir / "manifest.json")
    return oof, rep


def predict_wells(model, wells, cfg, provider, device, **kw):
    out = {}
    for w in wells:
        pred, det = predict_well(model, w, cfg.canvas_spec(), cfg, provider,
                                 device, return_detail=True, **kw)
        out[w.well_id] = dict(pred=pred, sigma=det.get("sigma"))
    return out
