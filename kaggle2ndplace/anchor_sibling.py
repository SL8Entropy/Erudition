#!/usr/bin/env python3
"""Sibling-lateral reference GR: a better profile to match the well against.

Every model here works by comparing the lateral's gamma ray against a reference profile
of "what GR reads at each depth", and sliding until the patterns line up.  That reference
is the supplied typewell, which has two weaknesses: it is a different, lower-resolution
tool than the lateral (the competition host said so directly), and it may simply not
represent this well's rock, which is the signature of the catastrophic wells.

Siblings are the fix.  The 773 typewells consolidate into ~54-60 master systems -- members
agree to 0.000 GR where they overlap, so they are one log cropped differently -- and wells
sharing a system are drilling the same rock.  For a sibling that is a *training* well the
true TVT is known at every row, so its **lateral** GR can be binned by depth into a
profile measured by a horizontal well, at lateral resolution, in the same rock.

3rd place weighted typewell against sibling 0.2 / 0.8 in their HMM and called reference-GR
choice their single biggest private-leaderboard lever.

Leakage: the bank is built only from wells passed to ``fit`` (training wells), and
``profile`` always drops the queried well itself, so a training well is never its own
sibling.
"""

from __future__ import annotations

import numpy as np

DEFAULT_GRID_FT = 1.0
DEFAULT_MIN_OVERLAP_FT = 100.0
DEFAULT_CORR = 0.9


def _union_find(n, pairs):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def master_systems(wells, names, grid_ft=DEFAULT_GRID_FT, min_overlap_ft=DEFAULT_MIN_OVERLAP_FT,
                   corr_thr=DEFAULT_CORR):
    """Group wells whose typewells are crops of one series, by dense pairwise correlation.

    One matmul over a shared depth grid: standardise each typewell on its own support,
    correlate every pair over the depths they share, join pairs that overlap enough and
    agree enough, and take connected components.
    """
    lo = min(float(wells[n]["tw_tvt"][0]) for n in names)
    hi = max(float(wells[n]["tw_tvt"][-1]) for n in names)
    grid = np.arange(lo, hi + grid_ft, grid_ft)
    A = np.full((len(names), len(grid)), np.nan, dtype=np.float32)
    for i, n in enumerate(names):
        t, g = wells[n]["tw_tvt"].astype(np.float64), wells[n]["tw_gr"].astype(np.float64)
        inside = (grid >= t[0]) & (grid <= t[-1])
        A[i, inside] = np.interp(grid[inside], t, g)
    valid = np.isfinite(A)
    Z = np.where(valid, A, 0.0).astype(np.float64)
    cnt = valid.astype(np.float64)
    n_ov = cnt @ cnt.T                                        # overlapping bins per pair
    s1 = Z @ cnt.T                                            # sums restricted to the overlap
    s2 = (Z ** 2) @ cnt.T
    s12 = Z @ Z.T
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = s12 / n_ov - (s1 / n_ov) * (s1.T / n_ov)
        va = s2 / n_ov - (s1 / n_ov) ** 2
        corr = cov / np.sqrt(np.maximum(va, 1e-12) * np.maximum(va.T, 1e-12))
    ok = (n_ov >= min_overlap_ft / grid_ft) & (corr >= corr_thr)
    iu, ju = np.triu_indices(len(names), 1)
    pairs = [(int(i), int(j)) for i, j in zip(iu, ju) if ok[i, j]]
    roots = _union_find(len(names), pairs)
    return {n: int(r) for n, r in zip(names, roots)}


class SiblingBank:
    """Per-system depth -> GR profile, built from training wells' laterals."""

    def __init__(self, bin_ft: float = 1.0):
        self.bin_ft = float(bin_ft)
        self.groups: dict[str, int] = {}
        self.by_well: dict[str, tuple] = {}     # well -> (tvt_centres, gr, count) of its own lateral
        self.systems: dict[int, list] = {}

    def fit(self, wells, train_names, all_names=None, log=print):
        """Group over ``all_names``, but bin laterals from ``train_names`` only.

        Grouping is allowed to see every well: it reads only the typewell GR, which is an
        input at inference time.  The depth->GR profile is the part that needs the answer,
        so it is built from training wells alone.
        """
        all_names = list(all_names) if all_names is not None else list(train_names)
        self.groups = master_systems(wells, all_names)
        sizes = {}
        for n in train_names:
            w = wells[n]
            tvt, gr = w["tvt"].astype(np.float64), w["gr"].astype(np.float64)
            ok = np.isfinite(tvt) & np.isfinite(gr)
            if ok.sum() < 64:
                continue
            idx = np.floor(tvt[ok] / self.bin_ft).astype(np.int64)
            lo = idx.min()
            cnt = np.bincount(idx - lo)
            tot = np.bincount(idx - lo, weights=gr[ok])
            have = cnt > 0
            centres = (np.arange(len(cnt))[have] + lo + 0.5) * self.bin_ft
            self.by_well[n] = (centres, tot[have] / cnt[have], cnt[have].astype(np.float64))
            g = self.groups[n]
            self.systems.setdefault(g, []).append(n)
            sizes[g] = sizes.get(g, 0) + 1
        counts = np.array(sorted(sizes.values(), reverse=True))
        n_sys_all = len(set(self.groups.values()))
        log(f"sibling bank: {n_sys_all} systems over {len(all_names)} wells; "
            f"{len(self.by_well)} training laterals binned into {len(self.systems)} of them; "
            f"members median {int(np.median(counts))}, max {int(counts.max())}, "
            f"{int((counts == 1).sum())} singleton systems")
        return self

    def profile(self, name, tvt_query, min_wells=1):
        """Depth->GR at ``tvt_query`` from this well's siblings, plus a coverage mask.

        Counts are accumulated across siblings so a depth logged by several wells outweighs
        one logged by a single well.  The well itself is always dropped, so a training well
        is never its own sibling.  Returns (gr, covered_mask, n_siblings).
        """
        tvt_query = np.asarray(tvt_query, dtype=np.float64)
        members = [m for m in self.systems.get(self.groups.get(name, -1), []) if m != name]
        if len(members) < min_wells:
            return np.full(len(tvt_query), np.nan), np.zeros(len(tvt_query), bool), 0
        num = np.zeros(len(tvt_query))
        den = np.zeros(len(tvt_query))
        for m in members:
            c, g, k = self.by_well[m]
            inside = (tvt_query >= c[0]) & (tvt_query <= c[-1])
            if not inside.any():
                continue
            num[inside] += np.interp(tvt_query[inside], c, g * k)
            den[inside] += np.interp(tvt_query[inside], c, k)
        cov = den > 0
        out = np.full(len(tvt_query), np.nan)
        out[cov] = num[cov] / den[cov]
        return out, cov, len(members)

    def spread(self, name, tvt_query, min_wells=3):
        """Disagreement *between* siblings at each depth, in GR units.

        The mean sibling profile answers "what does this rock read here"; the spread answers
        "how much do wells in this rock disagree here", which is a per-depth statement about
        how much evidence a GR match at that depth is worth.  3rd place's sibling tables
        carried a sigma alongside the bias for the same reason.

        Note their warning, which we do not intend to repeat: sharpening a particle filter's
        likelihood with this sigma failed on 5 of 5 profiles, because a broad likelihood was
        doing implicit robustness work.  It is used here only as an input channel, never as
        a likelihood.
        """
        tvt_query = np.asarray(tvt_query, dtype=np.float64)
        members = [m for m in self.systems.get(self.groups.get(name, -1), []) if m != name]
        if len(members) < min_wells:
            return np.zeros(len(tvt_query)), np.zeros(len(tvt_query), bool)
        n = np.zeros(len(tvt_query)); s1 = np.zeros(len(tvt_query)); s2 = np.zeros(len(tvt_query))
        for m in members:
            c, g, _ = self.by_well[m]
            inside = (tvt_query >= c[0]) & (tvt_query <= c[-1])
            if not inside.any():
                continue
            v = np.interp(tvt_query[inside], c, g)      # each sibling counts once, not per row
            n[inside] += 1.0; s1[inside] += v; s2[inside] += v * v
        cov = n >= min_wells
        out = np.zeros(len(tvt_query))
        with np.errstate(invalid="ignore"):
            out[cov] = np.sqrt(np.maximum(s2[cov] / n[cov] - (s1[cov] / n[cov]) ** 2, 0.0))
        return out, cov

    def n_siblings(self, name):
        return max(len([m for m in self.systems.get(self.groups.get(name, -1), []) if m != name]), 0)


def apply_to_wells(wells, train_names, all_names, weight, bin_ft=1.0, log=print):
    """Replace every well's typewell GR with a typewell/sibling blend, in place.

    This is the cheapest possible injection point for the CNNs: ``gr2tvt_data.build_compact``
    reads ``tw_gr`` to fill the typewell channel, so blending here changes what the model
    matches against without touching the author's modules, the channel count or the
    checkpoint layout.  Flip augmentation reverses ``tw_gr`` alongside ``tw_tvt``, so the
    blend survives it.

    Depths no sibling logged keep the original typewell, and the well itself is never one of
    its own siblings.  Returns the bank.
    """
    bank = SiblingBank(bin_ft=bin_ft).fit(wells, train_names, all_names=all_names, log=log)
    touched = rows = 0
    for n in all_names:
        w = wells[n]
        tvt = w["tw_tvt"].astype(np.float64)
        sib, cov, ns = bank.profile(n, tvt)
        if ns == 0 or not cov.any():
            continue
        g = w["tw_gr"].astype(np.float32).copy()
        g[cov] = ((1.0 - weight) * g[cov] + weight * sib[cov]).astype(np.float32)
        w["tw_gr"] = g
        touched += 1
        rows += int(cov.sum())
    log(f"sibling reference applied at weight {weight:.2f} to {touched}/{len(all_names)} wells "
        f"({rows / max(touched, 1):.0f} typewell samples blended per well)")
    return bank


def attach_spread(wells, train_names, all_names, bank=None, bin_ft=1.0, log=print):
    """Store each well's sibling-disagreement profile on the well dict, for the sigma channel.

    Kept as (tvt, sigma) on the well rather than resolved to levels here, because the level
    axis depends on the sample's anchor and tvt_shift, which are not known until build time.
    """
    if bank is None:
        bank = SiblingBank(bin_ft=bin_ft).fit(wells, train_names, all_names=all_names, log=log)
    got = 0
    for n in all_names:
        w = wells[n]
        tvt = w["tw_tvt"].astype(np.float64)
        sig, cov = bank.spread(n, tvt)
        if not cov.any():
            continue
        w["sib_sd_tvt"] = tvt.astype(np.float32)
        w["sib_sd"] = np.where(cov, sig, 0.0).astype(np.float32)
        got += 1
    log(f"sibling spread attached to {got}/{len(all_names)} wells "
        f"(median disagreement {np.median([np.median(wells[n]['sib_sd'][wells[n]['sib_sd'] > 0]) for n in all_names if 'sib_sd' in wells[n]]):.1f} GR)")
    return bank
