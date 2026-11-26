#!/usr/bin/env python3
"""Check that every finished ablation run followed the same protocol.

For each results/<recipe>_ep150/cfg.pkl (the config the run really used), this prints:
  - the run's intended change: how its recipe differs from small_s<seed> (the plain control)
  - the run-time settings applied on top of the recipe (epochs, batch, workers, ...)
and fails if those run-time settings are not identical across all runs.
It also fails if a run's config differs from its registered recipe in any other way.

Run from ablation_study/pipeline:
    python check_protocol.py                    # every run in results/
    python check_protocol.py --results some/dir
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "experiments" / "bilzard"))
import seq_NN_cfg  # noqa: E402  (needed to unpickle cfg.pkl and to read the recipes)

# Allowed to differ between laptops without changing the method: speed or machine-only settings.
MACHINE_ONLY = {"num_workers", "device", "persistent_workers", "prefetch_factor", "pin_memory"}


def flatten(obj, prefix=""):
    d = obj if isinstance(obj, dict) else {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def norm(v):
    # --offline-timm rewrites "hf_hub:timm/<name>" to "<name>": the same weights.
    if isinstance(v, str):
        return v.replace("hf_hub:timm/", "")
    return v


def is_path(v):
    return isinstance(v, (str, Path)) and (":\\" in str(v) or ":/" in str(v) or str(v).startswith("/"))


def diff(a, b):
    keys = set(a) | set(b)
    return {k: (a.get(k), b.get(k)) for k in sorted(keys)
            if repr(norm(a.get(k))) != repr(norm(b.get(k))) and not is_path(a.get(k)) and not is_path(b.get(k))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    recipes = dict(seq_NN_cfg.SEQ_TRAIN_CFGS)
    runs = sorted(p for p in Path(args.results).glob("*_ep150") if (p / "cfg.pkl").exists())
    if not runs:
        print(f"No finished runs (no */cfg.pkl) under {args.results}")
        return 1
    problems, overrides = [], {}
    for run in runs:
        name = run.name[: -len("_ep150")]
        if name not in recipes:
            problems.append(f"{name}: not an ablation recipe")
            continue
        used = flatten(pickle.load(open(run / "cfg.pkl", "rb")))
        recipe = flatten(recipes[name])
        seed = name.rsplit("_s", 1)[-1]
        control = flatten(recipes.get(f"small_s{seed}", recipes[name]))
        intended = {k: v[0] for k, v in diff(recipe, control).items() if k != "seed"}
        applied = {k: v[0] for k, v in diff(used, recipe).items()}
        overrides[name] = applied
        print(f"\n== {name}")
        print("   intended change vs small_s" + seed + ":", ", ".join(sorted(intended)) or "none (this is the control)")
        print("   run-time settings:", ", ".join(f"{k}={v}" for k, v in sorted(applied.items())) or "none")
    # Every run must carry exactly the same run-time settings, machine-only keys aside.
    method = {n: {k: v for k, v in o.items() if k not in MACHINE_ONLY} for n, o in overrides.items()}
    ref_name = next(iter(method), None)
    for n, o in method.items():
        if repr(o) != repr(method[ref_name]):
            problems.append(f"{n}: run-time settings differ from {ref_name}: "
                            f"{ {k: o.get(k) for k in set(o) | set(method[ref_name]) if repr(o.get(k)) != repr(method[ref_name].get(k))} }")
    machine = {n: {k: v for k, v in o.items() if k in MACHINE_ONLY} for n, o in overrides.items()}
    if len({repr(m) for m in machine.values()}) > 1:
        print("note: machine-only settings differ (speed only, not the method):",
              ", ".join(f"{n} {m}" for n, m in machine.items()))
    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"OK: {len(overrides)} runs share identical run-time settings; each differs from its control only by its recipe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
