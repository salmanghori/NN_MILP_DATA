#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import random
import joblib

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ---------------------- Feature schema helpers ----------------------
def feature_schema_for(feat_per_agent: int):
    """
    Return (CONT_COLS_IN_AGENT, FLAG_COLS_IN_AGENT) for supported layouts.
    Supports 8 (legacy), 10 (cross-only margin), 12 (cross + same margins).
    """
    if feat_per_agent == 12:
        # [pos, vel, flow, dist_c,
        #  gap_same_min, has_same,
        #  cross_sum_min, has_cross,
        #  cross_margin, cross_is_safe,
        #  gap_margin,  gap_is_safe]
        cont = [0, 1, 3, 4, 6, 8, 10]
        flags = [2, 5, 7, 9, 11]
    elif feat_per_agent == 10:
        cont = [0, 1, 3, 4, 6, 8]
        flags = [2, 5, 7, 9]
    elif feat_per_agent == 8:
        cont = [0, 1, 3, 4, 6]
        flags = [2, 5, 7]
    else:
        raise ValueError(f"Unsupported feat_per_agent={feat_per_agent}. Expected one of {8,10,12}.")
    return cont, flags


# ---------------------- Dataset wrapper ----------------------
class XYDataset(Dataset):
    def __init__(self, X, mask, Y, ymask):
        self.X = X
        self.mask = mask
        self.Y = Y
        self.ymask = ymask

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x  = self.X[idx]
        y  = self.Y[idx]
        ym = self.ymask[idx]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(ym)


# ---------------------- Model ----------------------
class FiLMLayer(nn.Module):
    def __init__(self, global_dim, feature_dim):
        super().__init__()
        self.processor = nn.Sequential(
            nn.Linear(global_dim, 2*feature_dim)
        )
        # init close to identity: gamma≈0, beta≈0
        with torch.no_grad():
            for m in self.processor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x, g):
        gamma, beta = torch.chunk(self.processor(g), 2, dim=-1)
        gamma = gamma.unsqueeze(1); beta = beta.unsqueeze(1)
        return (x * (1.0 + gamma)) + beta

class PerSlotMLP(nn.Module):
    def __init__(self, max_agents, feat_per_agent, include_global=True, global_dim=0,
                 hidden=512, depth=7, dropout=0.08):
        super().__init__()
        self.max_agents = max_agents
        self.feat_per_agent = feat_per_agent
        self.include_global = bool(include_global) and (global_dim > 0)
        self.global_dim = int(global_dim) if self.include_global else 0

        self.film_layer = FiLMLayer(self.global_dim, feat_per_agent) if self.include_global else None

        # HYBRID: concatenates raw globals after modulation
        in_dim = feat_per_agent + (self.global_dim if self.include_global else 0)

        layers, last = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden), nn.ReLU()]
            if dropout and dropout > 0.0:
                layers += [nn.Dropout(dropout)]
            last = hidden
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        B = X.shape[0]
        core_len = self.max_agents * self.feat_per_agent
        slots = X[:, :core_len].view(B, self.max_agents, self.feat_per_agent)

        if self.include_global and self.film_layer is not None:
            g = X[:, core_len:]  # [B, G]
            mod = self.film_layer(slots, g)  # [B, A, F]
            g_exp = g.unsqueeze(1).expand(B, self.max_agents, self.global_dim)  # [B, A, G]
            inp = torch.cat([mod, g_exp], dim=-1)  # [B, A, F+G]
        else:
            inp = slots

        return self.net(inp).squeeze(-1)



# ---------------------- Utils: stratified split by file ----------------------
def _build_file_index(meta):
    """
    Return:
      files: sorted unique file names
      file2rows: {file -> [row indices]}
      file2pair: {file -> (lambda, R)}
    """
    files = [m['file'] for m in meta]
    uniq = sorted(set(files))
    file2rows = {f: [] for f in uniq}
    file2pair = {}
    for i, f in enumerate(files):
        file2rows[f].append(i)
        if f not in file2pair:
            lam = float(meta[i]['arrival_rate'])
            R   = float(meta[i]['R'])
            file2pair[f] = (lam, R)
    return uniq, file2rows, file2pair


def _choose_one(files, file2pair, predicate, rng):
    cand = [f for f in files if predicate(file2pair[f])]
    rng.shuffle(cand)
    return cand[0] if cand else None


def stratified_split_by_file(
    meta,
    test_size=3,
    val_size=4,
    hard_R=(100.0, 160.0),
    easy_R=(140.0, 200.0),
    seed=42,
    override_test_files=None,
    override_val_files=None,
):
    """
    File-level split that guarantees val/test coverage of key scenarios.
    If overrides are provided, they are used verbatim (must match meta['file'] strings).
    """
    rng = random.Random(seed)
    all_files, file2rows, file2pair = _build_file_index(meta)
    pool = set(all_files)

    def is_lam(x): return lambda t: abs(t[0] - x) < 1e-9
    def in_R(Rset): return lambda t: t[1] in Rset

    # TEST
    if override_test_files:
        test_files = [f for f in override_test_files if f in pool]
        for f in test_files:
            pool.discard(f)
    else:
        test_files = []
        hh = _choose_one(list(pool), file2pair, lambda t: is_lam(0.9)(t) and in_R(set(hard_R))(t), rng)
        if hh: test_files.append(hh); pool.discard(hh)
        he = _choose_one(list(pool), file2pair, lambda t: is_lam(0.9)(t) and in_R(set(easy_R))(t), rng)
        if he: test_files.append(he); pool.discard(he)
        ee = _choose_one(list(pool), file2pair, lambda t: is_lam(0.5)(t) and in_R(set(easy_R))(t), rng)
        if ee: test_files.append(ee); pool.discard(ee)
        rest = list(pool)
        rng.shuffle(rest)
        for f in rest:
            if len(test_files) >= test_size:
                break
            test_files.append(f); pool.discard(f)

    # VAL
    if override_val_files:
        val_files = [f for f in override_val_files if f in pool]
        for f in val_files:
            pool.discard(f)
    else:
        val_files = []
        hh2 = _choose_one(list(pool), file2pair, lambda t: is_lam(0.9)(t) and in_R(set(hard_R))(t), rng)
        if hh2: val_files.append(hh2); pool.discard(hh2)
        lam7 = _choose_one(list(pool), file2pair, is_lam(0.7), rng)
        if lam7: val_files.append(lam7); pool.discard(lam7)
        e_h = _choose_one(list(pool), file2pair, lambda t: is_lam(0.5)(t) and in_R(set(hard_R))(t), rng)
        if e_h: val_files.append(e_h); pool.discard(e_h)
        rest = list(pool)
        rng.shuffle(rest)
        for f in rest:
            if len(val_files) >= val_size:
                break
            val_files.append(f); pool.discard(f)

    train_files = sorted(pool)
    val_files   = sorted(val_files)
    test_files  = sorted(test_files)

    # Build row index arrays
    idx_train, idx_val, idx_test = [], [], []
    for f in train_files:
        idx_train.extend(file2rows[f])
    for f in val_files:
        idx_val.extend(file2rows[f])
    for f in test_files:
        idx_test.extend(file2rows[f])

    return (np.array(idx_train, dtype=np.int64),
            np.array(idx_val,   dtype=np.int64),
            np.array(idx_test,  dtype=np.int64),
            set(train_files), set(val_files), set(test_files), file2pair)


def print_bucket_table(meta, idx_train, idx_val, idx_test):
    rows = {
        'train': set(idx_train.tolist()),
        'val':   set(idx_val.tolist()),
        'test':  set(idx_test.tolist())
    }
    from collections import defaultdict
    counts = {k: defaultdict(int) for k in rows.keys()}
    for split_name, idxset in rows.items():
        for i in idxset:
            lam = float(meta[i]['arrival_rate'])
            R   = float(meta[i]['R'])
            counts[split_name][(lam, R)] += 1

    keys = sorted({k for d in counts.values() for k in d.keys()})
    print("\n[buckets] rows per (Λ,R) by split:")
    print(f"{'Λ':>10} {'R':>10} {'train':>12} {'val':>12} {'test':>12}")
    for lam, R in keys:
        t = counts['train'].get((lam, R), 0)
        v = counts['val'].get((lam,  R), 0)
        s = counts['test'].get((lam, R), 0)
        print(f"{lam:10.3f} {R:10.1f} {t:12d} {v:12d} {s:12d}")


# ---------------------- Other utils ----------------------
def compute_continuous_feature_mask(max_agents, feat_per_agent, include_globals, global_dim):
    cont_cols, flag_cols = feature_schema_for(feat_per_agent)
    feat_len = max_agents * feat_per_agent + (global_dim if include_globals else 0)
    cont_mask = np.zeros((feat_len,), dtype=bool)
    for s in range(max_agents):
        base = s * feat_per_agent
        for c in cont_cols:
            cont_mask[base + c] = True
        for f in flag_cols:
            cont_mask[base + f] = False
    if include_globals and global_dim > 0:
        cont_mask[-global_dim:] = True
    return cont_mask


def fit_and_apply_scaler_inplace(X, cont_mask, train_idx):
    scaler = StandardScaler()
    X_train_cont = X[train_idx][:, cont_mask]
    scaler.fit(X_train_cont)
    X[:, cont_mask] = scaler.transform(X[:, cont_mask])
    return scaler


def masked_mse(pred, target, ymask):
    diff2 = (pred - target) ** 2
    masked = diff2 * ymask
    denom = torch.clamp(ymask.sum(), min=1.0)
    return masked.sum() / denom


def masked_huber(pred, target, ymask, delta=1.0):
    """
    Smooth L1 (Huber) on valid slots.
    delta: transition point between L2 and L1 (a.k.a. beta in PyTorch SmoothL1).
    """
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    huber = 0.5 * (quadratic ** 2) / delta + (abs_diff - quadratic)
    masked = huber * ymask
    denom = torch.clamp(ymask.sum(), min=1.0)
    return masked.sum() / denom


def current_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg.get('lr', None)
    return None


# ---------------------- Group metrics (per-Λ / per-R) ----------------------
@torch.no_grad()
def evaluate_groups(model, device, X_all, Y_all, ymask_all, meta_all, row_indices, batch_size=32768, loss_fn='huber', delta=1.0):
    """
    Compute masked MSE/MAE per arrival_rate and per R on a subset defined by row_indices.
    """
    model.eval()
    preds_list, y_list, m_list = [], [], []
    for start in range(0, len(row_indices), batch_size):
        sl = row_indices[start:start+batch_size]
        xb = torch.from_numpy(X_all[sl]).to(device, non_blocking=True)
        pb = model(xb).cpu().numpy()
        preds_list.append(pb)
        y_list.append(Y_all[sl])
        m_list.append(ymask_all[sl])
    P = np.vstack(preds_list)
    Y = np.vstack(y_list)
    M = np.vstack(m_list)

    def masked_mae_np(p, y, m):
        num = np.sum(np.abs(p - y) * m)
        den = np.sum(m) + 1e-9
        return float(num / den)

    def masked_mse_np(p, y, m):
        num = np.sum(((p - y) ** 2) * m)
        den = np.sum(m) + 1e-9
        return float(num / den)

    lambdas, radii = {}, {}
    for i, gi in enumerate(row_indices):
        lam = float(meta_all[gi]['arrival_rate'])
        R   = float(meta_all[gi]['R'])
        lambdas.setdefault(lam, []).append(i)
        radii.setdefault(R, []).append(i)

    results = {'by_lambda': [], 'by_R': []}
    for lam, idxs in sorted(lambdas.items()):
        p, y, m = P[idxs], Y[idxs], M[idxs]
        results['by_lambda'].append({
            'lambda': lam,
            'count_rows': len(idxs),
            'masked_MSE': masked_mse_np(p, y, m),
            'masked_MAE': masked_mae_np(p, y, m)
        })
    for R, idxs in sorted(radii.items()):
        p, y, m = P[idxs], Y[idxs], M[idxs]
        results['by_R'].append({
            'R': R,
            'count_rows': len(idxs),
            'masked_MSE': masked_mse_np(p, y, m),
            'masked_MAE': masked_mae_np(p, y, m)
        })
    return results


# ---------------------- Training loop ----------------------
def train_one_file(
    npz_path,
    batch_size=16384, epochs=80, lr=1e-3, hidden=512, depth=7, dropout=0.08,
    seed=42, num_workers=4, weight_decay=1e-4,
    loss='huber', huber_delta=0.2,
    lr_factor=0.5, lr_patience=2, min_lr=1e-6, early_patience=10,
    grad_clip=1.0,
    test_size=3, val_size=4,
    hard_R_str="100.0,160.0", easy_R_str="140.0,200.0",
    test_files_override=None, val_files_override=None,
):
    print(f"\n[load] {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    # Required arrays
    X      = data['X'].astype(np.float32)
    mask   = data['mask'].astype(np.float32)      # slots where an agent exists at t
    Y      = data['Y'].astype(np.float32)
    y_mask = data['y_mask'].astype(np.float32)    # slots where agent exists at t and t+1
    meta   = list(data['meta'])

    # Meta info for shapes
    max_agents       = int(data['max_agents'][0]) if 'max_agents' in data else None
    feat_per_agent   = int(data['feat_per_agent'][0]) if 'feat_per_agent' in data else 12
    include_global   = bool(int(data['include_global'][0])) if 'include_global' in data else True
    global_dim       = int(data['global_dim'][0]) if 'global_dim' in data else 2

    assert max_agents is not None, "max_agents not found in NPZ."
    _ = feature_schema_for(feat_per_agent)  # validate schema

    print(f"[info] X={X.shape} Y={Y.shape} | max_agents={max_agents} feat_per_agent={feat_per_agent} "
          f"| include_globals={include_global} global_dim={global_dim}")

    # Sanity checks on masks
    assert np.all((mask == 0) | (mask == 1)), "mask must be binary {0,1}"
    assert np.all((y_mask == 0) | (y_mask == 1)), "y_mask must be binary {0,1}"
    assert np.all(y_mask <= mask), "y_mask must be 0 wherever mask is 0 at time t"
    assert y_mask.sum() > 0, "No valid targets (y_mask.sum()==0)."

    # Stratified split by file
    tf_override = [s.strip() for s in test_files_override.split(',')] if (test_files_override and len(test_files_override.strip())>0) else None
    vf_override = [s.strip() for s in val_files_override.split(',')] if (val_files_override and len(val_files_override.strip())>0) else None

    hard_R = tuple(float(r.strip()) for r in hard_R_str.split(',') if r.strip())
    easy_R = tuple(float(r.strip()) for r in easy_R_str.split(',') if r.strip())

    idx_train, idx_val, idx_test, train_files, val_files, test_files, file2pair = stratified_split_by_file(
        meta,
        test_size=test_size,
        val_size=val_size,
        hard_R=hard_R,
        easy_R=easy_R,
        seed=seed,
        override_test_files=tf_override,
        override_val_files=vf_override,
    )

    print(f"[split] files: train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print(f"[split] rows : train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    def list_pairs(files_set, name):
        lst = sorted(list(files_set))
        pairs = [file2pair[f] for f in lst]
        print(f"[{name}] files ({len(lst)}): {lst}")
        print(f"[{name}] (Λ,R): {[ (float(l), float(r)) for (l,r) in pairs ]}")

    list_pairs(train_files, "train")
    list_pairs(val_files,   "val")
    list_pairs(test_files,  "test")

    print_bucket_table(meta, idx_train, idx_val, idx_test)

    # Coverage stats
    total_slots = y_mask.shape[0] * y_mask.shape[1]
    overall_cov = float(y_mask.sum() / total_slots)
    def cov_pair(rows):
        rows = np.asarray(rows, dtype=np.int64)
        ym = y_mask[rows]
        mk = mask[rows]
        overall = float(ym.sum() / ym.size) if ym.size > 0 else 0.0
        real = float(ym.sum() / np.clip(mk.sum(), 1, None)) if mk.sum() > 0 else 0.0
        return overall, real

    train_overall, train_real = cov_pair(idx_train)
    val_overall,   val_real   = cov_pair(idx_val)
    test_overall,  test_real  = cov_pair(idx_test)

    print(f"[coverage] overall valid targets ratio = {overall_cov:.4f}")
    print(f"[coverage] train overall={train_overall:.4f} real={train_real:.4f} | "
          f"val overall={val_overall:.4f} real={val_real:.4f} | "
          f"test overall={test_overall:.4f} real={test_real:.4f}")

    # Zero baseline on train/val (masked MSE)
    def masked_mse_np(p, y, m):
        num = np.sum(((p - y) ** 2) * m)
        den = np.sum(m) + 1e-9
        return float(num / den)

    if len(idx_train) > 0 and len(idx_val) > 0:
        zero_train = masked_mse_np(np.zeros_like(Y[idx_train]), Y[idx_train], y_mask[idx_train])
        zero_val   = masked_mse_np(np.zeros_like(Y[idx_val]),   Y[idx_val],   y_mask[idx_val])
        print(f"[baseline] zero-accel masked_MSE: train={zero_train:.6f}  val={zero_val:.6f}")
    else:
        print("[baseline] Skipped zero-accel baseline (empty train/val).")

    # Normalize ONLY continuous features (fit on train)
    cont_mask = compute_continuous_feature_mask(max_agents, feat_per_agent, include_global, global_dim)
    scaler = fit_and_apply_scaler_inplace(X, cont_mask, idx_train if len(idx_train)>0 else np.arange(X.shape[0]))

    # Datasets/DataLoaders
    pin = torch.cuda.is_available()
    ds_train = XYDataset(X[idx_train], mask[idx_train], Y[idx_train], y_mask[idx_train])
    ds_val   = XYDataset(X[idx_val],   mask[idx_val],   Y[idx_val],   y_mask[idx_val])
    ds_test  = XYDataset(X[idx_test],  mask[idx_test],  Y[idx_test],  y_mask[idx_test])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin, drop_last=False)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin, drop_last=False)

    # Model/opt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerSlotMLP(
        max_agents=max_agents,
        feat_per_agent=feat_per_agent,
        include_global=include_global,
        global_dim=global_dim,
        hidden=hidden,
        depth=depth,
        dropout=dropout
    ).to(device)
    print(f"[model] Using globals in MLP: {model.include_global} (global_dim={model.global_dim})")


    torch.backends.cudnn.benchmark = True
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=lr_factor, patience=lr_patience,
        cooldown=0, min_lr=min_lr, verbose=True
    )

    use_huber = (loss.lower() == 'huber')

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_mse, total_mae, denom = 0.0, 0.0, 0.0
        t0 = time.time()
        with torch.set_grad_enabled(train):
            for xb, yb, ymb in loader:
                xb   = xb.to(device, non_blocking=True)
                yb   = yb.to(device, non_blocking=True)
                ymb  = ymb.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    pred = model(xb)
                    if use_huber:
                        loss_val = masked_huber(pred, yb, ymb, delta=huber_delta)
                    else:
                        loss_val = masked_mse(pred, yb, ymb)

                    mse_batch = ((pred - yb) ** 2 * ymb).sum()
                    mae_batch = ((pred - yb).abs() * ymb).sum()

                if train:
                    scaler_amp.scale(loss_val).backward()
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler_amp.step(opt)
                    scaler_amp.update()

                total_mse += mse_batch.item()
                total_mae += mae_batch.item()
                denom     += float(ymb.sum().item())

        elapsed = time.time() - t0
        return (total_mse / max(denom, 1.0), total_mae / max(denom, 1.0), elapsed)

    print(f"[train] device={device} | file={os.path.basename(npz_path)}")
    best_val = float('inf')
    best_ep  = -1
    epochs_no_improve = 0
    basename = os.path.splitext(os.path.basename(npz_path))[0]
    out_dir = os.path.dirname(npz_path)

    if len(idx_val) == 0 or len(idx_train) == 0:
        print("[warn] Empty train or val split; skipping training/eval.")
        return

    for ep in range(1, epochs + 1):
        train_mse, train_mae, t_train = run_epoch(dl_train, train=True)
        val_mse,   val_mae,   t_val   = run_epoch(dl_val,   train=False)

        scheduler.step(val_mse)

        print(f"Epoch {ep:02d} | train MSE={train_mse:.6f} MAE={train_mae:.6f} | "
              f"val MSE={val_mse:.6f} MAE={val_mae:.6f} | lr={current_lr(opt):.2e} | time={t_train:.1f}s")

        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_ep = ep
            epochs_no_improve = 0
            save_path = os.path.join(out_dir, f"best_model_{basename}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'max_agents': max_agents,
                'feat_per_agent': feat_per_agent,
                'include_globals': include_global,
                'global_dim': global_dim,
                'cont_mask': cont_mask,
                'loss': loss,
                'huber_delta': huber_delta,
                'hidden': hidden,
                'depth': depth,
                'dropout': dropout,
            }, save_path)
            joblib.dump(scaler, os.path.join(out_dir, f"scaler_{basename}.pkl"))
            np.savez_compressed(
                os.path.join(out_dir, f"indices_{basename}.npz"),
                idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
                train_files=np.array(list(train_files)),
                val_files=np.array(list(val_files)),
                test_files=np.array(list(test_files))
            )
            print(f"[save] best (val MSE={best_val:.6f} @ ep {ep}) -> {save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_patience:
            print(f"[early-stop] no val improvement for {early_patience} epoch(s) (best @ ep {best_ep}).")
            break

    if best_ep == -1:
        save_path = os.path.join(out_dir, f"best_model_{basename}.pt")
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        best_ep = 0
        best_val = float('nan')

    # --- Test using best checkpoint ---
    best_ckpt = os.path.join(out_dir, f"best_model_{basename}.pt")
    print(f"[load-best] Loaded best checkpoint for test: {best_ckpt} (best val MSE={best_val:.6f} @ ep {best_ep})")
    state = torch.load(best_ckpt, map_location=device)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)

    ds_test = XYDataset(X[idx_test], mask[idx_test], Y[idx_test], y_mask[idx_test])
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin)
    model.eval()
    total_mse, total_mae, denom = 0.0, 0.0, 0.0
    with torch.no_grad():
        for xb, yb, ymb in dl_test:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ymb = ymb.to(device, non_blocking=True)
            pb = model(xb)
            total_mse += ((pb - yb) ** 2 * ymb).sum().item()
            total_mae += ((pb - yb).abs() * ymb).sum().item()
            denom     += float(ymb.sum().item())
    print(f"[test] masked_MSE={total_mse/max(denom,1.0):.6f}  masked_MAE={total_mae/max(denom,1.0):.6f}")

    group_stats = evaluate_groups(
        model, device, X, Y, y_mask, meta, idx_test,
        batch_size=max(batch_size, 32768), loss_fn=loss, delta=huber_delta
    )
    print("\n[breakdown] By arrival_rate (Λ):")
    for g in group_stats['by_lambda']:
        print(f"  Λ={g['lambda']:.3f}: rows={g['count_rows']:6d} | MSE={g['masked_MSE']:.6f} MAE={g['masked_MAE']:.6f}")

    print("\n[breakdown] By radius (R):")
    for g in group_stats['by_R']:
        print(f"  R={g['R']:.1f}: rows={g['count_rows']:6d} | MSE={g['masked_MSE']:.6f} MAE={g['masked_MAE']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to NPZ (e.g., inputs_targets_dataset_XY_ALL.npz)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Loss options
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"])
    parser.add_argument("--huber-delta", type=float, default=0.2)

    # Scheduler / early stop
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")

    # Stability
    parser.add_argument("--grad-clip", type=float, default=1.0, help="clip grad norm to this value; <=0 disables")

    # Stratified split
    parser.add_argument("--test-size", type=int, default=3, help="Target num files in test set")
    parser.add_argument("--val-size", type=int, default=4, help="Target num files in val set")
    parser.add_argument("--hard-R", type=str, default="100.0,160.0", help="Comma-sep list of 'hard' R values")
    parser.add_argument("--easy-R", type=str, default="140.0,200.0", help="Comma-sep list of 'easy' R values")
    parser.add_argument("--test-files-override", type=str, default=None,
                        help="Comma-sep list of exact file *labels* as stored in meta['file']")
    parser.add_argument("--val-files-override", type=str, default=None,
                        help="Comma-sep list of exact file *labels* as stored in meta['file']")

    args = parser.parse_args()

    # Repro
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prefer CPUs Slurm gave us; fall back to CLI
    num_workers_env = int(os.environ.get("SLURM_CPUS_PER_TASK", args.num_workers))

    train_one_file(
        npz_path=args.data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        seed=args.seed,
        num_workers=num_workers_env,
        weight_decay=args.weight_decay,
        loss=args.loss,
        huber_delta=args.huber_delta,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
        early_patience=args.patience,
        grad_clip=args.grad_clip,
        test_size=args.test_size,
        val_size=args.val_size,
        hard_R_str=args.hard_R,
        easy_R_str=args.easy_R,
        test_files_override=args.test_files_override,
        val_files_override=args.val_files_override,
    )


if __name__ == "__main__":
    main()                   