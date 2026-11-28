#!/usr/bin/env python3
"""Network speed, size and GPU memory of trained runs: teacher vs distilled student.

Times the U-Net forward pass for one well (batch 1, the 16 x 345 x 400 input the pipeline builds),
in fp32 and in the pipeline's own setting (bf16 autocast + channels_last). The runs are measured
in alternating rounds so laptop power-state drift (~30% between sessions) hits all of them alike;
each number is the median over rounds. Input building, the geo prior and post-processing are not
timed: they are the same CPU work for every model.

Run from ablation_study/pipeline with an idle GPU:
    python bench_speed.py x_slope_s1 cnx_fastvit_kd_xslope_s1 [more runs ...] > results/_bench_speed.txt
"""
import argparse
import os
import pathlib
import pickle
import statistics
import sys
from pathlib import Path

import torch

sys.modules.setdefault("pathlib._local", pathlib)  # cfg.pkl written by Python 3.13+
sys.path.insert(0, str(Path(__file__).parent / "experiments" / "bilzard"))
import seq_NN_cfg  # noqa: E402,F401  (needed to unpickle cfg.pkl / models.pkl)
from seq_NN_train import make_model  # noqa: E402


def load_run(name):
    d = Path("results") / f"{name}_ep150"
    with open(d / "cfg.pkl", "rb") as f:
        cfg = pickle.load(f)
    with open(d / "models.pkl", "rb") as f:
        payload = pickle.load(f)
    model = payload[sorted(payload)[0]] if isinstance(payload, dict) else payload
    return cfg, model.eval(), (d / "models.pkl").stat().st_size


def dummy_batch(cfg, device):
    n_ch = len(cfg.unet_static_channels)
    bins, tw = int(cfg.num_bins), int(cfg.typewell_len)
    return (torch.randn(1, n_ch, bins, tw, device=device),
            torch.randn(1, 2, tw, device=device),
            torch.full((1, bins), float(cfg.downsample), device=device),
            torch.ones(1, bins, dtype=torch.bool, device=device))


def time_ms(model, batch, bf16, iters, warmup):
    x, aux, cnt, mask = batch
    if bf16:
        x = x.contiguous(memory_format=torch.channels_last)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        for _ in range(warmup):
            model(x, aux, bin_count=cnt, row_mask=mask)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # after warm-up: excludes cuDNN's algorithm-search workspace
        times = []
        for _ in range(iters):
            start.record()
            model(x, aux, bin_count=cnt, row_mask=mask)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run names without _ep150, e.g. x_slope_s1")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    entries = []  # (label, model, cfg, file_bytes)
    for name in a.runs:
        cfg, model, size = load_run(name)
        entries.append((name, model, cfg, size))  # models.pkl is saved already reparameterised
        if any(type(m).__name__ == "RepMixer" for m in model.modules()):
            # Same architecture before reparameterisation; timing does not depend on the weights.
            entries.append((f"{name} before reparam", make_model(cfg).eval(), cfg, size))

    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"batch 1 | {a.rounds} rounds x {a.iters} iters, median ms per well")
    rows = {e[0]: {"fp32": [], "bf16": []} for e in entries}
    mem = {}
    for r in range(a.rounds):
        for label, model, cfg, _ in entries:
            model.to(dev)
            batch = dummy_batch(cfg, dev)
            for mode in ("fp32", "bf16"):
                if mode == "bf16":
                    model.to(memory_format=torch.channels_last)
                rows[label][mode].append(time_ms(model, batch, mode == "bf16", a.iters, a.warmup))
                if r == 0:
                    mem[(label, mode)] = torch.cuda.max_memory_allocated() / 2**20
            model.to(memory_format=torch.contiguous_format).cpu()
            torch.cuda.empty_cache()

    print(f"{'model':44s} {'params':>8s} {'file MB':>8s} {'fp32 ms':>8s} {'bf16 ms':>8s} {'peak MiB fp32/bf16':>19s}")
    for label, model, cfg, size in entries:
        n_par = sum(t.numel() for t in model.parameters()) / 1e6
        f32, b16 = (statistics.median(rows[label][m]) for m in ("fp32", "bf16"))
        print(f"{label:44s} {n_par:7.1f}M {size / 2**20:8.1f} {f32:8.2f} {b16:8.2f} "
              f"{mem[(label, 'fp32')]:9.0f}/{mem[(label, 'bf16')]:.0f}")
    print("per-round ms (fp32 | bf16):")
    for label in rows:
        print(f"  {label:42s} " + " ".join(f"{v:.1f}" for v in rows[label]["fp32"]) + " | "
              + " ".join(f"{v:.1f}" for v in rows[label]["bf16"]))


if __name__ == "__main__":
    main()
