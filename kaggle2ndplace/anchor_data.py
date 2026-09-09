#!/usr/bin/env python3
"""Training-side data handling for the AnchorCNN (ROGII wellbore geology, 2nd place).

The inference modules in ``src/`` are the author's, unmodified.  They already contain
everything needed to *build* a sample (``gr2tvt_data.build_compact``) and to *train* on
one (``gr2tvt_model.anchor_loss``); what they do not contain is the training-side
plumbing.  That is this file: well loading and caching, the well-level train/holdout
split, the augmentation stack, and a ``torch`` dataset whose items are exactly the
tuples ``gr2tvt_model.items_to_x`` consumes.

Grid (32 ft family, the ``plan147/dzl_w1`` checkpoints):

    vertical    512 rows, typewell level TVT - TVT_PS over +-128 ft at 0.5 ft/row
    horizontal  336 columns of 32 ft: 16 known pre-PS columns then 320 predicted ones
    channels    9, assembled by ``GR2TVTAnchorNet.assemble``:
                0 GR mismatch vs typewell   1 lateral GR   2 GR validity
                3 typewell coverage         4 known-TVT ridge   5 known-column flag
                6 GR mismatch vs pre-PS profile   7 its coverage   8 dip z

Split.  ``split_wells`` takes the first 80% of the sorted well ids for training and the
last 20% for holdout, which is byte-for-byte the split the 1st-place repo's
``seq_NN_holdout_eval.py`` uses (618 train / 155 holdout, first holdout well
``d07aed8f``, 754,122 scored rows).  Numbers from the two solutions are therefore
directly comparable, and predictions written by ``anchor_train.py`` can be fed to that
repo's ``seq_NN_robust_compare.py`` unchanged.

Augmentation.  Only transforms that leave the GR <-> TVT correspondence exactly intact
are applied to real wells:

    re-anchor   pick a new prediction-start row, so one well yields many samples
    level shift move the +-128 ft window by up to +-32 ft (``tvt_shift``)
    MD phase    move the 32 ft column boundaries within one column (``md_phase``)
    MD flip     traverse the well backwards
    level flip  mirror TVT, the typewell TVT axis and Z about one common constant
    specaug     blank spans of GR, in the same format as real missing data

Structural scaling in z_layer space -- stretching the formation slope -- is *not*
applied to real wells, and that is deliberate.  It maps TVT by an amount that varies
along the well, so the observed GR would no longer be f(TVT) for any typewell f: it is
only consistent when the GR is regenerated from the typewell, i.e. on synthetic wells.
See ``synth_well`` below and the notes in ``TRAINING.md``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gr2tvt_data as gd  # noqa: E402

NULLS = ["nan", "NaN", "NA", ""]
HW_COLS = ["MD", "Z", "TVT", "GR", "TVT_input"]
CACHE_VERSION = 1


# --------------------------------------------------------------------------- loading


def well_names(data_dir: Path) -> list[str]:
    """Every well id under ``data_dir``, sorted.  The split is defined on this order."""
    return sorted({p.name.split("__")[0] for p in Path(data_dir).glob("*__horizontal_well.csv")})


def _read_well(data_dir: Path, name: str) -> dict[str, np.ndarray]:
    hw = pl.read_csv(data_dir / f"{name}__horizontal_well.csv", null_values=NULLS,
                     columns=HW_COLS).sort("MD")
    tw = pl.read_csv(data_dir / f"{name}__typewell.csv", null_values=NULLS)
    twt = tw["TVT"].cast(pl.Float64).to_numpy()
    twg = tw["GR"].cast(pl.Float64).to_numpy()
    m = np.isfinite(twt) & np.isfinite(twg)
    twt, twg = twt[m], twg[m]
    o = np.argsort(twt, kind="stable")
    get = lambda c: hw[c].cast(pl.Float64).to_numpy().astype(np.float32)  # noqa: E731
    return dict(md=get("MD"), z=get("Z"), tvt=get("TVT"), gr=get("GR"),
                tvt_input=get("TVT_input"),
                tw_tvt=twt[o].astype(np.float32), tw_gr=twg[o].astype(np.float32))


def _cache_path(cache: Path, names: list[str]) -> Path:
    """Give each distinct well list its own cache file.

    Keyed by the list, not just the path, so a ``--limit-wells`` smoke run cannot
    overwrite the full 773-well cache that a real run depends on.
    """
    import hashlib
    h = hashlib.sha1("\n".join(names).encode()).hexdigest()[:8]
    cache = Path(cache)
    return cache.with_name(f"{cache.stem}__{len(names)}_{h}{cache.suffix}")


def load_wells(data_dir: Path, names: list[str], cache: Path | None = None,
               log=print) -> dict[str, dict]:
    """Load wells as float32 arrays, through a flat ``.npz`` cache when one is given.

    float32 is deliberate: at 1 ft MD spacing and depths of order 10^4 ft its spacing is
    ~1e-3 ft, three orders below the ~1 ft errors being measured, and it keeps the whole
    773-well set near 130 MB so DataLoader workers (spawn on Windows) stay affordable.
    Everything downstream promotes to float64 inside ``build_compact``.
    """
    data_dir = Path(data_dir)
    if cache is not None:
        cache = _cache_path(cache, names)
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if int(z["version"]) == CACHE_VERSION and list(z["names"]) == list(names):
            log(f"wells: cache hit {cache}")
            hw, tw, hw_off, tw_off = z["hw"], z["tw"], z["hw_off"], z["tw_off"]
            out = {}
            for i, n in enumerate(names):
                a, b = int(hw_off[i]), int(hw_off[i + 1])
                c, d = int(tw_off[i]), int(tw_off[i + 1])
                out[n] = dict(md=hw[0, a:b], z=hw[1, a:b], tvt=hw[2, a:b], gr=hw[3, a:b],
                              tvt_input=hw[4, a:b], tw_tvt=tw[0, c:d], tw_gr=tw[1, c:d])
            return out
        log(f"wells: cache at {cache} does not match this well list, rebuilding")

    out = {}
    for i, n in enumerate(names):
        out[n] = _read_well(data_dir, n)
        if (i + 1) % 100 == 0:
            log(f"wells: {i + 1}/{len(names)}")
    if cache is not None:
        cache = Path(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        hw_off = np.cumsum([0] + [len(out[n]["md"]) for n in names]).astype(np.int64)
        tw_off = np.cumsum([0] + [len(out[n]["tw_tvt"]) for n in names]).astype(np.int64)
        hw = np.stack([np.concatenate([out[n][k] for n in names])
                       for k in ("md", "z", "tvt", "gr", "tvt_input")])
        tw = np.stack([np.concatenate([out[n][k] for n in names]) for k in ("tw_tvt", "tw_gr")])
        np.savez(cache, version=CACHE_VERSION, names=np.array(names), hw=hw, tw=tw,
                 hw_off=hw_off, tw_off=tw_off)
        log(f"wells: cached {len(names)} wells to {cache}")
    return out


# ----------------------------------------------------------------------------- split


def split_wells(names: list[str], train_frac: float = 0.8) -> tuple[list[str], list[str]]:
    """Contiguous first-``train_frac`` / remainder split of the sorted well ids.

    Same rule, same rounding and same sort order as the 1st-place repo's
    ``seq_NN_holdout_eval.split_wells``, so both solutions score the same 155 wells.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be strictly between 0 and 1, got {train_frac}")
    n_train = int(np.floor(len(names) * train_frac))
    if n_train < 1 or n_train >= len(names):
        raise ValueError(f"train_frac={train_frac} leaves {n_train}/{len(names) - n_train} wells")
    return list(names[:n_train]), list(names[n_train:])


def eval_rows(w: dict) -> np.ndarray:
    """Row indices to predict: exactly where ``TVT_input`` is NaN (the well's suffix)."""
    return np.where(~np.isfinite(w["tvt_input"]))[0]


# ---------------------------------------------------------------------- augmentation


@dataclass
class AugCfg:
    """Probabilities and ranges for the real-well augmentation stack."""

    crop_prob: float = 1.0             # re-anchor at a random prediction-start row
    pre_ft: tuple = (700.0, 2400.0)    # pre-PS context to keep, ft (real wells: ~1660)
    eval_ft: tuple = (2500.0, 9500.0)  # eval-zone length to keep, ft (real wells: ~4990)
    md_flip_prob: float = 0.5
    level_flip_prob: float = 0.5
    level_shift_ft: float = 32.0       # tvt_shift ~ U(-x, x)
    md_phase: bool = True              # md_phase ~ U(0, COLW)
    specaug_prob: float = 0.5
    specaug_spans: tuple = (1, 3)      # number of masked spans, inclusive range
    specaug_cols: tuple = (1, 21)      # span width in 32 ft columns, inclusive range
    synth_prob: float = 0.0            # see synth_well(); 0 disables it entirely
    synth_mix: tuple = (0.2, 0.8)      # trajectory mixup lambda range
    move_cap_ft: float = 20.0          # widest move the 21-class vocabulary can express
    dev_cap_ft: float = 120.0          # widest label allowed, level shift included (window 128)

    @property
    def min_pre_ft(self) -> float:
        return float(self.pre_ft[0])


def lateral_span(tvt: np.ndarray, rows_per_col: int, move_cap_ft: float) -> tuple[int, int]:
    """The longest run of rows whose column-to-column TVT moves fit the vocabulary.

    A well's known prefix is not all lateral: it contains the curve landing into the
    lateral, where TVT sweeps ~700 ft and moves reach ~40 ft per 32 ft column.  The move
    vocabulary tops out at +-20 ft and the window at +-128 ft, so a re-anchored sample
    taken there could not be expressed on the grid at all -- the label would be clipped
    and the supervision would be a lie.  Real evaluation zones never enter it (their
    whole TVT range is ~28 ft), so excluding it also keeps training on-distribution.

    Returned as a span rather than a start because MD flip reverses the well, which moves
    the build section from the front to the back.  Derived from the data, not assumed.
    """
    rpc = max(1, rows_per_col)
    good = (np.abs(np.diff(tvt[::rpc].astype(np.float64))) <= move_cap_ft).astype(np.int8)
    if good.size == 0 or not good.any():
        return 0, len(tvt)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], good, [0]))))
    starts, ends = edges[::2], edges[1::2]
    j = int(np.argmax(ends - starts))
    return int(starts[j] * rpc), int(min(len(tvt), (ends[j] + 1) * rpc))


def _flip_md(w: dict) -> dict:
    """Traverse the well backwards.  MD is rebuilt increasing; the samples reverse."""
    md = w["md"]
    return dict(w, md=(md[0] + (md[-1] - md[::-1])).astype(md.dtype),
                z=w["z"][::-1].copy(), tvt=w["tvt"][::-1].copy(), gr=w["gr"][::-1].copy())


def _flip_level(w: dict, c: float) -> dict:
    """Mirror the depth axis about ``c``: TVT, the typewell TVT axis and Z together.

    Reflecting TVT and the typewell about the *same* constant leaves f(TVT) intact, so
    the GR the model has to match is unchanged; reflecting Z as well keeps the identity
    dTVT = dz_layer - dz (and therefore the dip channel and the dz_layer head) consistent.
    """
    twt = (2.0 * c - w["tw_tvt"])[::-1].copy()
    twg = w["tw_gr"][::-1].copy()
    return dict(w, tvt=(2.0 * c - w["tvt"]).astype(w["tvt"].dtype),
                z=(2.0 * float(w["z"][0]) - w["z"]).astype(w["z"].dtype),
                tw_tvt=twt, tw_gr=twg)


def _specaug(gr: np.ndarray, rng: np.random.Generator, cfg: AugCfg,
             dmd: float = 1.0) -> np.ndarray:
    """Blank spans of GR.  Missing GR is a real, common condition in this dataset."""
    gr = gr.copy()
    n = len(gr)
    for _ in range(int(rng.integers(cfg.specaug_spans[0], cfg.specaug_spans[1] + 1))):
        cols = int(rng.integers(cfg.specaug_cols[0], cfg.specaug_cols[1] + 1))
        width = max(1, int(cols * gd.COLW / dmd))       # columns -> rows, via MD spacing
        if width >= n:
            continue
        s = int(rng.integers(0, n - width))
        gr[s:s + width] = np.nan
    return gr


def _window(tvt: np.ndarray, dmd: float, rng: np.random.Generator, cfg: AugCfg,
            ps_col: int) -> tuple[int, int, int] | None:
    """Choose (start, i0, end) row indices: context start, prediction start, right edge.

    Lengths are drawn in feet and converted with the well's own MD spacing.  Two hard
    constraints make the sample representable on the grid, and both are checked rather
    than assumed: the whole window lies inside the lateral span, and the label never
    leaves the +-128 ft window (``dev_cap_ft``).  Returns None when this well cannot
    supply such a window, and the caller falls back to the real one.
    """
    rpc = max(1, int(round(gd.COLW / dmd)))
    lo, hi = lateral_span(tvt, rpc, cfg.move_cap_ft)
    min_pre = max(int(cfg.min_pre_ft / dmd), ps_col * rpc + 1)
    min_eval = max(int(cfg.eval_ft[0] / dmd), 32 * rpc)
    lo_i0, hi_i0 = lo + min_pre, hi - min_eval
    if lo_i0 > hi_i0:
        return None

    i0 = int(rng.integers(lo_i0, hi_i0 + 1))
    # the level shift moves the anchor afterwards, so it comes out of the same budget
    cap = max(8.0, cfg.dev_cap_ft - cfg.level_shift_ft)
    dev = np.abs(tvt[i0:hi].astype(np.float64) - float(tvt[i0 - 1])) > cap
    max_len = int(np.argmax(dev)) if dev.any() else hi - i0
    ev_len = min(int(rng.uniform(*cfg.eval_ft) / dmd), int(gd.H * gd.COLW / dmd), max_len)
    if ev_len < min_eval:
        return None
    pre_len = max(int(rng.uniform(*cfg.pre_ft) / dmd), min_pre)
    return max(lo, i0 - pre_len), i0, i0 + ev_len


def augment(w: dict, rng: np.random.Generator, cfg: AugCfg,
            ps_col: int = 16) -> tuple[dict, float, float]:
    """Apply the real-well augmentation stack.  Returns (well, tvt_shift, md_phase)."""
    orig = w
    n = len(w["md"])
    fin = np.isfinite(w["tvt_input"])
    i0_real = int(np.argmax(~fin)) if (~fin).any() else n
    dmd = float(np.median(np.diff(w["md"][:min(n, 512)]))) or 1.0

    if cfg.md_flip_prob and rng.random() < cfg.md_flip_prob:
        w = _flip_md(w)

    win = _window(w["tvt"], dmd, rng, cfg, ps_col) \
        if (cfg.crop_prob and rng.random() < cfg.crop_prob) else None
    if win is None:
        # No croppable window (e.g. a flip put the build section where the eval zone
        # would go).  Fall back to the well's own, always representable, real window.
        w, win = orig, (0, i0_real, n)
    start, i0, end = win

    sl = slice(start, end)
    w = {k: (v[sl] if k in ("md", "z", "tvt", "gr", "tvt_input") else v) for k, v in w.items()}
    i0 = i0 - start

    if cfg.level_flip_prob and rng.random() < cfg.level_flip_prob:
        w = _flip_level(w, float(w["tvt"][max(0, i0 - 1)]))

    ti = np.where(np.arange(len(w["md"])) < i0, w["tvt"], np.nan).astype(np.float32)
    w = dict(w, tvt_input=ti)

    if cfg.specaug_prob and rng.random() < cfg.specaug_prob:
        w = dict(w, gr=_specaug(w["gr"], rng, cfg, dmd))

    shift = float(rng.uniform(-cfg.level_shift_ft, cfg.level_shift_ft)) if cfg.level_shift_ft else 0.0
    phase = float(rng.uniform(0.0, gd.COLW)) if cfg.md_phase else 0.0
    return w, shift, phase


# ------------------------------------------------------------------------- synthesis


def synth_well(w: dict, other: dict, rng: np.random.Generator,
               cfg: AugCfg) -> dict | None:
    """A partially faithful version of the writeup's synthetic wells.

    Implements steps 3-5 of the pipeline -- mix two real TVT trajectories, read the GR
    back out of a typewell along the mixed path, and paste a real well's residual on top:

        TVT_mix = k*TVT_a + (1-k)*TVT_b        (trajectory mixup)
        GR      = f(TVT_mix) + r               (f = typewell profile, r = real residual)

    Steps 1-2 -- consolidating the 773 typewells into master series and re-skinning a
    trajectory into a different system at the same quantile of its drilling band -- are
    NOT implemented, so the vertical material stays the well's own typewell and the
    diversity is lower than the author's.  Off by default (``synth_prob=0``).
    """
    n = min(len(w["md"]), len(other["md"]))
    twt, twg = w["tw_tvt"].astype(np.float64), w["tw_gr"].astype(np.float64)
    if n < 1024 or len(twt) < 8:
        return None

    lam = float(rng.uniform(*cfg.synth_mix))
    a, b = w["tvt"][:n].astype(np.float64), other["tvt"][:n].astype(np.float64)
    tvt = lam * a + (1.0 - lam) * (b - b[0] + a[0])
    if tvt.min() <= twt[0] or tvt.max() >= twt[-1]:
        return None            # would need GR the typewell does not cover; reject, don't clip

    r = w["gr"][:n].astype(np.float64) - np.interp(np.clip(a, twt[0], twt[-1]), twt, twg)
    gr = np.interp(tvt, twt, twg) + np.where(np.isfinite(r), r, 0.0)
    gr[~np.isfinite(w["gr"][:n])] = np.nan

    # z_layer is the internal representation, exactly as the writeup keeps it: the source
    # well's real formation surface is held fixed and the trajectory z is *derived* as
    # z = z_layer - TVT.  That is what makes dTVT = dz_layer - dz true by construction,
    # so the dip channel and the dz_layer head stay consistent with the new path.
    z_layer = a + w["z"][:n].astype(np.float64)
    return dict(md=w["md"][:n].copy(), z=(z_layer - tvt).astype(np.float32),
                tvt=tvt.astype(np.float32), gr=gr.astype(np.float32),
                tvt_input=w["tvt_input"][:n].copy(),
                tw_tvt=w["tw_tvt"], tw_gr=w["tw_gr"])


# ------------------------------------------------------------------------- item build


def build_item(w: dict, ps_col: int, with_label: bool, tvt_shift: float = 0.0,
               md_phase: float = 0.0, pre_ps_label: bool = False,
               colw_in: float | None = None) -> dict:
    """One model input, in the exact tuple layout ``gr2tvt_model.items_to_x`` expects."""
    c = gd.build_compact(w, with_label=with_label, colw_in=colw_in, pre_ps_cols=ps_col,
                         pre_ps_label=pre_ps_label, tvt_shift=tvt_shift,
                         pre_ps_gr=True, z_dip=True, md_phase=md_phase)
    item = (c["t_n"], c["h_n"], c["h_valid"], c["t_cover"], c["y"].astype(np.float32),
            c["meta"]["ncol_tot"], c["y_bnd"], c["h_cnt"], c["h_rows"], c["k_lv"],
            c["k_flag"], c["s_n"], c["s_cover"], c["d_n"])
    return dict(item=item, meta=c["meta"])


class AnchorWellDataset(Dataset):
    """Real wells, augmented.  ``__getitem__`` returns {"item", "meta", "name"}.

    The collate function is a no-op list: ``items_to_x`` does its own ``np.stack``, so
    the batch is handed to the author's code in the shape it already expects.
    """

    def __init__(self, wells: dict[str, dict], names: list[str], aug: AugCfg | None,
                 ps_col: int = 16, seed: int = 0, epoch_len: int | None = None,
                 pre_ps_label: bool = False):
        self.wells, self.names, self.aug = wells, list(names), aug
        self.ps_col, self.seed, self.pre_ps_label = ps_col, seed, pre_ps_label
        self.epoch_len = int(epoch_len) if epoch_len else len(self.names)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_len if self.aug is not None else len(self.names)

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng((self.seed * 1_000_003 + self.epoch) * 1_000_003 + idx)

    def __getitem__(self, idx: int) -> dict:
        if self.aug is None:
            name = self.names[idx]
            out = build_item(self.wells[name], self.ps_col, True, pre_ps_label=self.pre_ps_label)
            return dict(out, name=name)

        rng = self._rng(idx)
        for attempt in range(8):
            name = self.names[int(rng.integers(len(self.names)))] if idx >= len(self.names) \
                else self.names[idx]
            w = self.wells[name]
            if self.aug.synth_prob and rng.random() < self.aug.synth_prob:
                other = self.wells[self.names[int(rng.integers(len(self.names)))]]
                s = synth_well(w, other, rng, self.aug)
                if s is not None:
                    w = s
            try:
                aw, shift, phase = augment(w, rng, self.aug, self.ps_col)
                out = build_item(aw, self.ps_col, True, tvt_shift=shift, md_phase=phase,
                                 pre_ps_label=self.pre_ps_label)
            except (ValueError, IndexError):
                rng = np.random.default_rng(int(rng.integers(1 << 62)))
                continue
            if np.isfinite(out["item"][6]).sum() >= 8:      # y_bnd: enough supervision
                return dict(out, name=name)
            rng = np.random.default_rng(int(rng.integers(1 << 62)))
        name = self.names[idx % len(self.names)]
        return dict(build_item(self.wells[name], self.ps_col, True,
                               pre_ps_label=self.pre_ps_label), name=name)


def collate(batch: list[dict]) -> list[dict]:
    return batch


def batch_tensors(batch: list[dict], device, model):
    """(x, y_bnd, d_n) for a batch, built through the author's ``items_to_x``."""
    from gr2tvt_model import items_to_x
    items = [b["item"] for b in batch]
    x, _ = items_to_x(items, device, model)
    y_bnd = torch.tensor(np.stack([it[6] for it in items]), device=device)
    d_n = torch.tensor(np.stack([it[13] for it in items]), device=device)
    return x, y_bnd, d_n
