"""Build the first training batches (simulation on, fixed seeds, 0 workers) from one fork.
usage: gen_items.py <source_dir> <cfg_name|-> <out.npz> [decoy_prob]"""
import sys, random
from copy import deepcopy
from pathlib import Path
SOL = Path(__file__).resolve().parents[1]   # kaggle_1st_place/solution
sys.path.insert(0, str(SOL))
import numpy as np, torch
import seq_NN_main_reproduce as repro

def flatten(obj, prefix, out):
    if torch.is_tensor(obj):
        out[prefix] = obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray) and obj.dtype != object:
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items(): flatten(v, f"{prefix}.{k}", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj): flatten(v, f"{prefix}[{i}]", out)

def main():
    src, name, outp = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    decoy = float(sys.argv[4]) if len(sys.argv) > 4 else None
    settings, _ = repro.load_settings(None)
    cfg_module, main_module = repro.import_snapshot(src)
    tr = main_module.seq_NN_train
    if name == "-":
        _, _, cfg = repro.select_recipe("0801_V2", cfg_module)
    else:
        cfg = deepcopy(dict(cfg_module.SEQ_TRAIN_CFGS)[name])
    if decoy is not None:
        cfg.sim_cfg["z_shift"]["decoy_prob"] = decoy
    repro.apply_settings_paths(cfg, settings, Path(outp).parent / "gen_items_out")
    n = str(cfg.model_cfg.get("unet_timm_model_name", ""))
    if n.startswith("hf_hub:timm/"): cfg.model_cfg["unet_timm_model_name"] = n[len("hf_hub:timm/"):]
    cfg.num_workers = 0
    cfg.refresh()
    wells = main_module.discover_well_ids(cfg.train_path)[:12]
    geo_cfg = tr.resolve_geo_prior_cfg(cfg)
    geo, _ = tr.make_geo_prior_for_wells(cfg.train_path, cfg.train_path, wells, wells, geo_cfg,
                                         exclude_query_from_support=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    loader = tr.make_loader(path=cfg.train_path, well_ids=wells, cfg=cfg, training=True, simulation=True,
                            shuffle=False, batch_size=2, seed=0, geo_prior=geo)
    out = {}
    for b, batch in enumerate(loader):
        flatten(batch, f"b{b}", out)
    np.savez_compressed(outp, **out)
    print("saved", len(out), "arrays")

if __name__ == "__main__":
    main()
