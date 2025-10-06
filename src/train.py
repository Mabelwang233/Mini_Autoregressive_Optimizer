import argparse, glob, os, csv
from pathlib import Path
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .data_hf import make_loaders
from .model_hf import build_model
from .utils import (
    set_seed, get_optimizer, bpb_from_loss, ppl_from_loss,
    grad_global_norm, param_global_norm, eval_sharpness_on_val
)

def load_cfg(path):  # yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def evaluate(model, loader, device):
    """Returns (avg_loss, perplexity, bpb) over the loader."""
    model.eval()
    tot_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(input_ids=x, labels=y)
            loss = out.loss
            tot_loss += loss.item()
            n_batches += 1
    avg_loss = tot_loss / max(1, n_batches)
    return avg_loss, ppl_from_loss(avg_loss), bpb_from_loss(avg_loss)

def train_one(cfg, opt_name: str):
    set_seed(cfg['run']['seed'])
    device = cfg['run']['device'] if torch.cuda.is_available() else 'cpu'
    out_dir = Path(cfg['run']['out_dir']) / f"{opt_name}_seed{cfg['run']['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(
        tokenizer_name=cfg['hf']['tokenizer_name'],
        data_root=cfg['hf']['data_root'],
        seq_len=cfg['train']['seq_len'],
        batch_size=cfg['train']['batch_size'],
        dataset_size=cfg['train']['dataset_size'],
        device=device,
    )

    model = build_model(cfg['hf']['model_name'], device)
    opt = get_optimizer(opt_name, model.parameters(), cfg['train']['lr'], cfg['train']['weight_decay'])

    best, best_path = float('inf'), out_dir / 'best.pt'
    step, total_steps = 0, cfg['run']['steps']
    pbar = tqdm(total=total_steps, desc=opt_name)

    header = ["step","train_loss","val_loss","val_ppl","val_bpb",
              "gap_loss","grad_norm","param_norm"]
    rows = [header]

    itr = iter(train_loader)
    while step < total_steps:
        try:
            x, y = next(itr)
        except StopIteration:
            itr = iter(train_loader)
            x, y = next(itr)

        model.train()
        out = model(input_ids=x, labels=y)
        train_loss = out.loss

        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        train_loss.backward()

        g_norm = grad_global_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
        opt.step()

        step += 1
        if step % cfg['run']['log_every'] == 0:
            val_loss, val_ppl, val_bpb = evaluate(model, val_loader, device)
            gap_loss = val_loss - float(train_loss.item())
            p_norm = param_global_norm(model)
            rows.append([
                step, f"{float(train_loss.item()):.4f}",
                f"{val_loss:.4f}", f"{val_ppl:.4f}", f"{val_bpb:.4f}",
                f"{gap_loss:.4f}", f"{g_norm:.4f}", f"{p_norm:.4f}"
            ])
            pbar.set_postfix(train=f"L{train_loss.item():.3f}",
                             val=f"L{val_loss:.3f}/PPL{val_ppl:.2f}")

        if step % cfg['run']['eval_every'] == 0:
            val_loss, _, _ = evaluate(model, val_loader, device)
            if val_loss < best:
                best = val_loss
                torch.save({'model': model.state_dict(), 'cfg': cfg}, best_path)

        pbar.update(1)

    # sharpness proxy summary
    sh_mean, sh_std = eval_sharpness_on_val(model, val_loader, batches=2, eps=1e-2)
    with open(out_dir / "summary.txt","w") as f:
        f.write(f"best_val_loss={best:.6f}\nsharpness_delta_mean={sh_mean:.6f}\nsharpness_delta_std={sh_std:.6f}\n")

    # write CSV
    with open(out_dir / "log.csv", "w", newline="") as f:
        writer = csv.writer(f); writer.writerows(rows)

    return out_dir

def plot_convergence(cfg, runs):
    # loss
    plt.figure(figsize=(7,5), dpi=150)
    for run in runs:
        path = Path(run) / "log.csv"
        steps, vals = [], []
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                steps.append(int(row["step"]))
                vals.append(float(row["val_loss"]))
        label = run.name.split("_")[0]
        plt.plot(steps, vals, label=label)
    plt.xlabel("step"); plt.ylabel("val CE loss"); plt.legend(); plt.title("Convergence (loss)")
    out_png_loss = Path(cfg['run']['out_dir']) / "convergence_loss.png"
    plt.savefig(out_png_loss, bbox_inches="tight")

    # perplexity
    plt.figure(figsize=(7,5), dpi=150)
    for run in runs:
        path = Path(run) / "log.csv"
        steps, ppls = [], []
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                steps.append(int(row["step"]))
                ppls.append(float(row["val_ppl"]))
        label = run.name.split("_")[0]
        plt.plot(steps, ppls, label=label)
    plt.xlabel("step"); plt.ylabel("val perplexity"); plt.legend(); plt.title("Convergence (perplexity)")
    out_png_ppl = Path(cfg['run']['out_dir']) / "convergence_ppl.png"
    plt.savefig(out_png_ppl, bbox_inches="tight")

    # gap (loss)
    plt.figure(figsize=(7,5), dpi=150)
    for run in runs:
        path = Path(run) / "log.csv"
        steps, gaps = [], []
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                steps.append(int(row["step"]))
                gaps.append(float(row["gap_loss"]))
        label = run.name.split("_")[0]
        plt.plot(steps, gaps, label=label)
    plt.xlabel("step"); plt.ylabel("gap (val_loss - train_loss)"); plt.legend(); plt.title("Generalization gap (loss)")
    out_png_gap = Path(cfg['run']['out_dir']) / "convergence_gap_loss.png"
    plt.savefig(out_png_gap, bbox_inches="tight")

def run_convergence(cfg):
    runs = []
    for name in ["sgd","adam","lion"]:
        runs.append(train_one(cfg, name))
    plot_convergence(cfg, runs)
    print("Saved convergence plots in", cfg['run']['out_dir'])
    return runs

def run_many(cfg, seeds=(1337, 2027, 3407), opts=("sgd","adam","lion")):
    results = {}  # {opt: [(val_loss, val_ppl), ...]}
    for opt in opts:
        results[opt] = []
        for sd in seeds:
            cfg_ = {k:(v.copy() if isinstance(v, dict) else v) for k,v in cfg.items()}
            cfg_["run"]["seed"] = sd
            out = train_one(cfg_, opt)
            with open(Path(out)/"log.csv") as f:
                rows = list(csv.DictReader(f))
            final = rows[-1]
            results[opt].append((float(final["val_loss"]), float(final["val_ppl"])))
    # summarize
    summary_rows = [["optimizer","val_loss_mean","val_loss_std","val_ppl_mean","val_ppl_std","n_runs"]]
    for opt, arr in results.items():
        vl = np.array([x[0] for x in arr]); vp = np.array([x[1] for x in arr])
        summary_rows.append([opt, f"{vl.mean():.4f}", f"{vl.std(ddof=1):.4f}",
                             f"{vp.mean():.4f}", f"{vp.std(ddof=1):.4f}", len(arr)])
    out_csv = Path(cfg['run']['out_dir']) / "multi_seed_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f); writer.writerows(summary_rows)
    print("Wrote", out_csv)
    return results

# ----- CLI -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, default='configs/default.yaml')
    ap.add_argument('--task', type=str,
                    choices=['convergence','landscape','multiseed'],
                    required=True)
    args, overrides = ap.parse_known_args()

    cfg = load_cfg(args.cfg)

    # allow CLI overrides like train.batch_size=16
    for ov in overrides:
        if '=' not in ov: continue
        key, val = ov.split('=', 1)
        target = cfg
        keys = key.split('.')
        for k in keys[:-1]:
            target = target[k]
        leaf = keys[-1]
        # parse scalars
        lv = val
        if val.lower() in ('true','false'):
            lv = val.lower() == 'true'
        else:
            try:
                lv = int(val)
            except ValueError:
                try:
                    lv = float(val)
                except ValueError:
                    pass
        target[leaf] = lv

    if args.task == 'convergence':
        run_convergence(cfg)
    elif args.task == 'multiseed':
        run_many(cfg)
    else:
        run_landscape(cfg)

if __name__ == '__main__':
    main()
