# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# from __future__ import annotations
# import os
# import re
# import json
# import math
# import argparse
# from collections import OrderedDict, Counter, defaultdict
# from pathlib import Path
# from typing import Any, Dict, List, Tuple
#
# import numpy as np
#
# # ===================== CONFIG: DATA BUILD =====================
# # Datasets (arrival-rate Λ) -> root folders
# DATASETS = OrderedDict([
#     (r"$\Lambda=0.9 $", r"C:\Users\ghoriss\Documents\IBEX\Longduration"),
#     (r"$\Lambda=0.7 $", r"C:\Users\ghoriss\Documents\IBEX\Longduration_1_low"),
#     (r"$\Lambda=0.5 $", r"C:\Users\ghoriss\Documents\IBEX\Longduration_2_low"),
# ])
#
# # Radii subfolders to scan under each dataset root (in this exact order)
# RADII_FOLDERS = ["R_40", "R_60", "R_80", "R_100", "R_120", "R_140", "R_160", "R_180", "R_200"]
#
# # Filenames to match (case-insensitive)
# FILENAME_PATTERN = re.compile(r"vehicles_positions_id_R_\d+.*\.json$", re.IGNORECASE)
#
# # ===================== FEATURES / TARGETS =====================
# # Per-agent slot feature layout (12):
# # [pos, vel, flow, dist, same_gap, has_same, cross_sum, has_cross, cross_mgn, cross_safe, gap_mgn, gap_safe]
# FEATURES_PER_SLOT = 12
# GLOBAL_FEATURES_DIM = 2  # [R, arrival_rate] appended once per sample
#
# # Safety semantics
# D_SAFE_M = 3.0  # same-lane headway threshold
# S_DIST_M = 4.0  # cross-flow sum-of-distances-to-center threshold
#
# # Target from t -> t+1
# DT = 0.1
# ACCEL_MIN, ACCEL_MAX = -5.0, 5.0
#
# # Validation toggle
# DISABLE_VALIDATION = False
#
# # ===================== HELPERS (old-format JSON) =====================
# def load_json(filepath: str):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return json.load(f)
#
# def is_inside_by_structure(rec) -> bool:
#     """Old schema: inside-zone records use nested singleton lists for numeric fields."""
#     # e.g., rec = [[x],[y],[vx],[vy], d, vid] or similar
#     return isinstance(rec, list) and len(rec) >= 6 and isinstance(rec[0], list)
#
# def normalize_record(rec):
#     """
#     Returns unified (x, y, vx, vy, flow, vid) for both inside/outside records.
#     Old logs store a 'd' field indicating lane: 'x'/0 for x-lane, 'y'/1 for y-lane.
#     """
#     if is_inside_by_structure(rec):
#         x, y = rec[0][0], rec[1][0]
#         vx, vy = rec[2][0], rec[3][0]
#         d, vid = rec[4], rec[5]
#     else:
#         # outside-zone shape (not used here except consistency)
#         x, y, vx, vy, d, vid = rec
#     # infer flow (0=x-lane, 1=y-lane)
#     flow = 0
#     if isinstance(d, str):
#         flow = 0 if d.lower().startswith('x') else 1
#     else:
#         flow = 0 if int(d) == 0 else 1
#     return float(x), float(y), float(vx), float(vy), int(flow), int(vid)
#
# def along_lane_pos_vel(x, y, vx, vy, flow):
#     """Return (pos_along_lane, vel_along_lane) for the agent's lane axis."""
#     return (x, vx) if flow == 0 else (y, vy)
#
# def find_all_json_files(root_dir: str):
#     """Yield (radius_folder_name, full_path) in the specified RADII order for a given dataset root."""
#     for folder_name in RADII_FOLDERS:
#         radius_path = os.path.join(root_dir, folder_name)
#         if not os.path.exists(radius_path):
#             print(f"[warn] Missing folder: {radius_path}")
#             continue
#         for root, _, files in os.walk(radius_path):
#             for filename in files:
#                 if FILENAME_PATTERN.match(filename):
#                     yield folder_name, os.path.join(root, filename)
#
# def parse_lambda_from_label(lbl: str) -> float:
#     """Extract Λ from labels like '$\\Lambda=0.9 $'. Fallback: first float."""
#     try:
#         s = lbl.replace("\\", "").replace("$", "").replace("Lambda", "Λ").replace(" ", "")
#         if "Λ=" in s:
#             return float(s.split("Λ=")[1])
#     except Exception:
#         pass
#     m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lbl)
#     return float(m.group(0)) if m else -1.0
#
# def parse_radius_from_folder(folder_name: str) -> float:
#     # "R_120" -> 120.0
#     try:
#         return float(folder_name.split('_')[1])
#     except Exception:
#         return -1.0
#
# # ===================== FIRST PASS: determine N_SLOTS =====================
# def first_pass_max_inside(DATASETS: OrderedDict) -> Tuple[int, int, Counter]:
#     """
#     Scan all datasets to compute:
#       - max number of inside-zone agents in any timestep (N_SLOTS)
#       - total number of matched JSON files
#       - histogram of concurrent inside agents
#     """
#     max_agents = 0
#     count_files = 0
#     hist = Counter()
#     for _, root_dir in DATASETS.items():
#         for _, fp in find_all_json_files(root_dir):
#             count_files += 1
#             data = load_json(fp)  # old schema: top-level is a list[timestep] of records
#             if not isinstance(data, list):
#                 continue
#             for t_list in data:
#                 if not t_list:
#                     hist[0] += 1
#                     continue
#                 inside = [rec for rec in t_list if is_inside_by_structure(rec)]
#                 n = len(inside)
#                 hist[n] += 1
#                 if n > max_agents:
#                     max_agents = n
#     return max_agents, count_files, hist
#
# def percentile_from_hist(hist: Counter, p: float) -> int:
#     """p-th percentile of a discrete histogram {value:count}."""
#     items = sorted(hist.items())
#     total = sum(c for _, c in items)
#     if total == 0:
#         return 0
#     threshold = p * total
#     cum = 0
#     for val, cnt in items:
#         cum += cnt
#         if cum >= threshold:
#             return val
#     return items[-1][0]
#
# # ===================== NEIGHBOR FEATURES & SAFETY =====================
# def same_and_cross_features(pos: np.ndarray, mask: np.ndarray, flows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Given aligned slot arrays (length S):
#       pos   : along-lane position (signed)
#       mask  : 1 if real agent, else 0
#       flows : 0 for x-lane, 1 for y-lane for real agents
#     Returns (same_gap_min, has_same, cross_sum_min, has_cross), each length S.
#     """
#     S = pos.shape[0]
#     abs_pos = np.abs(pos).astype(np.float32)
#
#     gap_same_min       = np.zeros((S,), dtype=np.float32)
#     cross_sum_min      = np.zeros((S,), dtype=np.float32)
#     has_same_neighbor  = np.zeros((S,), dtype=np.float32)
#     has_cross_neighbor = np.zeros((S,), dtype=np.float32)
#
#     idx_x = [i for i in range(S) if mask[i] > 0.5 and int(flows[i]) == 0]
#     idx_y = [i for i in range(S) if mask[i] > 0.5 and int(flows[i]) == 1]
#
#     # same-lane nearest neighbors in 1D (adjacent after sorting by pos)
#     def fill_same(indices: List[int]):
#         n = len(indices)
#         if n <= 1:
#             return
#         arr = sorted(indices, key=lambda i: pos[i])
#         for k in range(1, n):
#             i_prev, i_curr = arr[k-1], arr[k]
#             d = abs(pos[i_curr] - pos[i_prev])
#             # update both sides
#             if has_same_neighbor[i_curr] == 0 or d < gap_same_min[i_curr]:
#                 gap_same_min[i_curr] = d
#                 has_same_neighbor[i_curr] = 1.0
#             if has_same_neighbor[i_prev] == 0 or d < gap_same_min[i_prev]:
#                 gap_same_min[i_prev] = d
#                 has_same_neighbor[i_prev] = 1.0
#
#     fill_same(idx_x)
#     fill_same(idx_y)
#
#     # cross-lane min of |pos_i| + |pos_j|
#     def fill_cross(curr: List[int], other: List[int]):
#         if not other:
#             return
#         min_other_abs = float(np.min(abs_pos[other]))
#         for i in curr:
#             cross_sum_min[i] = abs_pos[i] + min_other_abs
#             has_cross_neighbor[i] = 1.0
#
#     fill_cross(idx_x, idx_y)
#     fill_cross(idx_y, idx_x)
#
#     return gap_same_min, has_same_neighbor, cross_sum_min, has_cross_neighbor
#
# # ===================== VALIDATION =====================
# def validate_sample(X_row: np.ndarray, Y_row: np.ndarray, mask: np.ndarray, y_mask: np.ndarray, features_per_slot: int) -> None:
#     if DISABLE_VALIDATION:
#         return
#     try:
#         S = mask.shape[0]
#         expect_feat_len = S * features_per_slot
#         X_slots = X_row[:expect_feat_len].reshape(S, features_per_slot)
#         valid = mask.astype(bool)
#
#         # y_mask ⊆ mask
#         if not np.all(y_mask <= mask):
#             print("[warn] y_mask has 1s where mask has 0s (should be subset).")
#
#         # Y bounds on present agents
#         if valid.any():
#             yv = Y_row[:S][valid]
#             if (yv < ACCEL_MIN - 1e-6).any() or (yv > ACCEL_MAX + 1e-6).any():
#                 print(f"[warn] Y out of bounds: [{yv.min():.3g}, {yv.max():.3g}]")
#
#         # padded slots must be zero
#         inv = ~valid
#         if inv.any() and np.where(np.any(np.abs(X_slots[inv]) > 1e-6, axis=1))[0].size > 0:
#             print("[warn] non-zero features in padded slots")
#
#         # boolean columns must be {0,1}
#         bool_cols = [5, 7, 9, 11]
#         if np.any(~np.isin(X_slots[:, bool_cols], [0.0, 1.0])):
#             print("[warn] non-binary values in boolean feature columns")
#
#         # dist ≈ |pos|
#         pos = X_slots[:, 0]
#         dist = X_slots[:, 3]
#         if np.where(np.abs(dist - np.abs(pos)) > 1e-5)[0].size > 0:
#             print("[warn] dist != abs(pos) found")
#
#         # safety flags consistent with margins if neighbor exists
#         has_cross = X_slots[:, 7]
#         cross_m  = X_slots[:, 8]
#         cross_s  = X_slots[:, 9]
#         if np.where((has_cross == 1.0) & ((cross_m > 0) != (cross_s == 1.0)))[0].size > 0:
#             print("[warn] cross_safe inconsistent with cross_mgn/has_cross")
#
#         has_same = X_slots[:, 5]
#         gap_m    = X_slots[:, 10]
#         gap_s    = X_slots[:, 11]
#         if np.where((has_same == 1.0) & ((gap_m > 0) != (gap_s == 1.0)))[0].size > 0:
#             print("[warn] gap_safe inconsistent with gap_mgn/has_same")
#
#     except Exception as e:
#         print(f"[warn] validation error ignored: {e}")
#
# # ===================== BUILD ONE DATASET (old schema) =====================
# def build_for_one_dataset(root_dir: str, arrival_rate: float, N_SLOTS: int):
#     """
#     Build arrays for a single Λ-root, using old JSON schema.
#     Returns: X, Y, mask, y_mask, meta_rows
#     """
#     X_list, Y_list, M_list, YM_list = [], [], [], []
#     meta_rows: List[Dict[str, Any]] = []
#
#     # coverage / audits
#     cov_total_slots = 0
#     cov_real_slots  = 0
#     cov_valid_slots = 0
#     rows_skipped_no_targets = 0
#     last_frame_rows = 0
#     overflow_drops  = 0
#     accel_nan_inf   = 0
#     feat_nan_inf    = 0
#     flow_flip_count = 0
#
#     per_file_cov = defaultdict(lambda: [0.0, 0.0])  # valid / real
#     per_R_cov    = defaultdict(lambda: [0.0, 0.0])
#     per_L_cov    = defaultdict(lambda: [0.0, 0.0])
#
#     for radius_folder, fp in find_all_json_files(root_dir):
#         R_val = parse_radius_from_folder(radius_folder)
#         data = load_json(fp)
#         if not isinstance(data, list):
#             print(f"[warn] {fp} top-level JSON not a list; skipping")
#             continue
#         T = len(data)
#         if T <= 1:
#             continue
#
#         # stable slot mapping within file
#         slot_map: Dict[int, int] = {}   # vid -> slot index
#         free_slots = list(range(N_SLOTS))
#
#         for t_idx in range(T):
#             if t_idx == T - 1:
#                 last_frame_rows += 1
#                 continue
#
#             t_list  = data[t_idx]
#             t1_list = data[t_idx + 1]
#             if not t_list:
#                 continue
#
#             inside_t  = [rec for rec in t_list  if is_inside_by_structure(rec)]
#             inside_t1 = [rec for rec in t1_list if is_inside_by_structure(rec)]
#             if not inside_t:
#                 continue
#
#             # normalize current inside agents
#             agents_now = []
#             vids_now = set()
#             for rec in inside_t:
#                 x, y, vx, vy, flow, vid = normalize_record(rec)
#                 flow = 0 if flow == 0 else 1
#                 agents_now.append({'vid': vid, 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'flow': flow})
#                 vids_now.add(vid)
#
#             # lookup next-step by vid
#             next_map: Dict[int, Dict] = {}
#             for rec in inside_t1:
#                 x, y, vx, vy, flow, vid = normalize_record(rec)
#                 flow = 0 if flow == 0 else 1
#                 next_map[vid] = {'vid': vid, 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'flow': flow}
#
#             # free slots for departed agents
#             missing = [vid for vid in list(slot_map.keys()) if vid not in vids_now]
#             if missing:
#                 import bisect
#                 for vid in missing:
#                     s = slot_map.pop(vid)
#                     bisect.insort(free_slots, s)
#
#             # allocate arrays
#             feats = np.zeros((N_SLOTS, FEATURES_PER_SLOT), dtype=np.float32)
#             mask  = np.zeros((N_SLOTS,), dtype=np.float32)
#             flows = np.full((N_SLOTS,), -1, dtype=np.int64)
#             yrow  = np.zeros((N_SLOTS,), dtype=np.float32)
#             ymask = np.zeros((N_SLOTS,), dtype=np.float32)
#
#             # fill base cols: pos, vel, flow, dist
#             for a in agents_now:
#                 vid = a['vid']
#                 # slot assignment (stable)
#                 if vid in slot_map:
#                     s = slot_map[vid]
#                 else:
#                     if not free_slots:
#                         overflow_drops += 1
#                         continue
#                     s = free_slots.pop(0)
#                     slot_map[vid] = s
#
#                 pos_along, vel_along = along_lane_pos_vel(a['x'], a['y'], a['vx'], a['vy'], a['flow'])
#                 dist_c = abs(pos_along)  # 1D distance spec you’re using for X
#
#                 feats[s, 0:4] = [pos_along, vel_along, float(a['flow']), dist_c]
#                 flows[s] = a['flow']
#                 mask[s] = 1.0
#
#             # neighbors & safety (only for real slots)
#             if mask.sum() > 0:
#                 pos = feats[:, 0]
#                 gap_same_min, has_same, cross_sum_min, has_cross = same_and_cross_features(pos, mask, flows)
#
#                 feats[:, 4] = gap_same_min
#                 feats[:, 5] = has_same
#                 feats[:, 6] = cross_sum_min
#                 feats[:, 7] = has_cross
#
#                 # cross safety margins / flags
#                 cross_mgn  = np.zeros((N_SLOTS,), dtype=np.float32)
#                 cross_safe = np.zeros((N_SLOTS,), dtype=np.float32)
#                 valid_c = has_cross > 0.5
#                 cross_mgn[valid_c]  = cross_sum_min[valid_c] - S_DIST_M
#                 cross_safe[valid_c] = (cross_mgn[valid_c] > 0.0).astype(np.float32)
#                 feats[:, 8] = cross_mgn
#                 feats[:, 9] = cross_safe
#
#                 # same-lane safety margins / flags
#                 gap_mgn  = np.zeros((N_SLOTS,), dtype=np.float32)
#                 gap_safe = np.zeros((N_SLOTS,), dtype=np.float32)
#                 valid_g = has_same > 0.5
#                 gap_mgn[valid_g]  = gap_same_min[valid_g] - D_SAFE_M
#                 gap_safe[valid_g] = (gap_mgn[valid_g] > 0.0).astype(np.float32)
#                 feats[:, 10] = gap_mgn
#                 feats[:, 11] = gap_safe
#
#             # targets: accel if persists to t+1
#             real_slots = int(mask.sum())
#             cov_total_slots += N_SLOTS
#             cov_real_slots  += real_slots
#
#             for s in range(N_SLOTS):
#                 if mask[s] < 0.5:
#                     continue
#                 # find vid by reverse lookup (slot_map is vid->slot)
#                 # build temporary reverse map once for speed if needed
#                 # small S so direct loop is fine
#                 vid_s = None
#                 for vid, ss in slot_map.items():
#                     if ss == s:
#                         vid_s = vid
#                         break
#                 if vid_s is None:
#                     continue
#
#                 if vid_s in next_map:
#                     flow_now = int(feats[s, 2])
#                     b = next_map[vid_s]
#                     flow_next = int(b['flow'])
#                     if flow_now != flow_next:
#                         flow_flip_count += 1
#                     v_now = feats[s, 1]
#                     _, v_next = along_lane_pos_vel(b['x'], b['y'], b['vx'], b['vy'], b['flow'])
#                     acc = (v_next - v_now) / DT
#                     if not np.isfinite(acc):
#                         accel_nan_inf += 1
#                         acc = 0.0
#                     acc = max(ACCEL_MIN, min(ACCEL_MAX, float(acc)))
#                     yrow[s]  = acc
#                     ymask[s] = 1.0
#
#             if ymask.sum() == 0:
#                 rows_skipped_no_targets += 1
#                 continue
#
#             cov_valid_slots += int(ymask.sum())
#
#             # flatten + append global features [R, Λ]
#             x_vec = feats.flatten()
#             if not np.isfinite(x_vec).all():
#                 feat_nan_inf += int(np.size(x_vec) - np.isfinite(x_vec).sum())
#                 x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=0.0, neginf=0.0)
#             x_vec = np.hstack([x_vec, np.array([R_val, arrival_rate], dtype=np.float32)])
#
#             X_list.append(x_vec)
#             Y_list.append(yrow)
#             M_list.append(mask)
#             YM_list.append(ymask)
#             meta_rows.append({
#                 "file": fp,
#                 "frame_idx": t_idx,
#                 "R": R_val,
#                 "arrival_rate": arrival_rate,
#                 "dataset_label": f"Λ={arrival_rate}",
#                 "radius_folder": radius_folder
#             })
#
#             # coverage buckets
#             per_file_cov[fp][0] += float(ymask.sum())
#             per_file_cov[fp][1] += float(mask.sum())
#             per_R_cov[R_val][0] += float(ymask.sum())
#             per_R_cov[R_val][1] += float(mask.sum())
#             per_L_cov[arrival_rate][0] += float(ymask.sum())
#             per_L_cov[arrival_rate][1] += float(mask.sum())
#
#     # stack
#     if X_list:
#         X = np.vstack(X_list).astype(np.float32)
#         Y = np.vstack(Y_list).astype(np.float32)
#         mask = np.vstack(M_list).astype(np.float32)
#         y_mask = np.vstack(YM_list).astype(np.float32)
#     else:
#         X = np.zeros((0, N_SLOTS * FEATURES_PER_SLOT + GLOBAL_FEATURES_DIM), dtype=np.float32)
#         Y = np.zeros((0, N_SLOTS), dtype=np.float32)
#         mask = np.zeros((0, N_SLOTS), dtype=np.float32)
#         y_mask = np.zeros((0, N_SLOTS), dtype=np.float32)
#
#     # coverage prints
#     total_slots = X.shape[0] * N_SLOTS
#     overall_cov = (y_mask.sum() / total_slots) if total_slots > 0 else 0.0
#     real_cov    = (y_mask.sum() / np.clip(mask.sum(), 1, None)) if mask.size > 0 else 0.0
#
#     print("[coverage] overall valid targets ratio = {:.4f}".format(overall_cov))
#     print("[coverage] valid / real-agent slots    = {:.4f}  (valid={} / real={})"
#           .format(real_cov, int(y_mask.sum()), int(mask.sum())))
#     print("[coverage] counters: total_slots={} real_slots={} valid_slots={}"
#           .format(cov_total_slots, cov_real_slots, cov_valid_slots))
#     print("[info] rows skipped (no targets)       =", rows_skipped_no_targets)
#     print("[info] last-frame rows (no t+1)        =", last_frame_rows)
#     print("[warn] slot overflows (agent dropped)   =", overflow_drops)
#     print("[warn] flow flips between t and t+1     =", flow_flip_count)
#     print("[audit] feature NaN/Inf replaced        =", feat_nan_inf)
#     print("[audit] target  NaN/Inf replaced        =", accel_nan_inf)
#
#     if mask.size > 0:
#         bad = np.logical_and(mask < 0.5, y_mask > 0.5).sum()
#         if bad > 0:
#             print(f"[ERROR] y_mask has {int(bad)} entries where mask==0; check builder logic!")
#
#     # show a few file coverages
#     print("\n[coverage/by-file] (first 10)")
#     for i, (f, (v, r)) in enumerate(per_file_cov.items()):
#         if i >= 10: break
#         ratio = (v / r) if r > 0 else 0.0
#         short = os.path.basename(f)
#         print(f"  {short:40s} real={int(r):8d} valid={int(v):8d} ratio={ratio:.3f}")
#
#     if per_L_cov:
#         print("\n[coverage/by-Λ]")
#         for lam in sorted(per_L_cov.keys()):
#             v, r = per_L_cov[lam]
#             print(f"  Λ={lam:.3f}: real={int(r):8d} valid={int(v):8d} ratio={(v/r) if r>0 else 0.0:.3f}")
#
#     if per_R_cov:
#         print("\n[coverage/by-R]")
#         for Rv in sorted(per_R_cov.keys()):
#             v, r = per_R_cov[Rv]
#             print(f"  R={Rv:5.1f}: real={int(r):8d} valid={int(v):8d} ratio={(v/r) if r>0 else 0.0:.3f}")
#
#     return X, Y, mask, y_mask, meta_rows
#
# # ===================== MAIN =====================
# def main():
#     ap = argparse.ArgumentParser(description="Build combined NPZ from OLD-format logs (list-of-frames).")
#     ap.add_argument("--out", type=str, default="inputs_targets_dataset_XY_ALL.npz",
#                     help="Output NPZ filename")
#     args = ap.parse_args()
#
#     # Pass 1: discover N_SLOTS
#     N_SLOTS, nfiles, hist = first_pass_max_inside(DATASETS)
#     if N_SLOTS <= 0:
#         print("[fatal] No inside-zone agents found across datasets. Check JSON format/paths.")
#         return
#     print(f"[info] scanned {nfiles} files; max inside agents per snapshot = {N_SLOTS}")
#     p95 = percentile_from_hist(hist, 0.95)
#     mean_occ = sum(k*v for k, v in hist.items()) / max(1, sum(hist.values()))
#     print("[padding] concurrent-agents histogram (top bins):")
#     for k, c in sorted(hist.items())[:10]:
#         print(f"  n={k:2d}: count={c}")
#     print(f"[padding] mean ≈ {mean_occ:.2f}, 95th percentile ≈ {p95}")
#     print(f"[note] using N_SLOTS = max = {N_SLOTS} (no truncation)")
#
#     all_X, all_Y, all_mask, all_y_mask = [], [], [], []
#     all_meta: List[Dict[str, Any]] = []
#     total_files_read = 0
#
#     # Pass 2: build per-Λ and concatenate
#     for ds_label, root_dir in DATASETS.items():
#         lam = parse_lambda_from_label(ds_label)
#         print(f"\n[build] Dataset {ds_label} (Λ={lam}) @ {root_dir}")
#         X, Y, mask, y_mask, meta_rows = build_for_one_dataset(root_dir, lam, N_SLOTS)
#
#         all_X.append(X); all_Y.append(Y); all_mask.append(mask); all_y_mask.append(y_mask)
#         all_meta.extend(meta_rows)
#         # Count files actually read via first pass; here we just note completion step
#         total_files_read += len({m['file'] for m in meta_rows})
#
#     # Concatenate
#     if all_X:
#         X_all     = np.vstack(all_X).astype(np.float32)
#         Y_all     = np.vstack(all_Y).astype(np.float32)
#         mask_all  = np.vstack(all_mask).astype(np.float32)
#         ymask_all = np.vstack(all_y_mask).astype(np.float32)
#     else:
#         X_all     = np.zeros((0, N_SLOTS * FEATURES_PER_SLOT + GLOBAL_FEATURES_DIM), dtype=np.float32)
#         Y_all     = np.zeros((0, N_SLOTS), dtype=np.float32)
#         mask_all  = np.zeros((0, N_SLOTS), dtype=np.float32)
#         ymask_all = np.zeros((0, N_SLOTS), dtype=np.float32)
#
#     meta_array = np.array(all_meta, dtype=object)
#
#     # Save ONE combined file
#     np.savez_compressed(
#         args.out,
#         X=X_all, Y=Y_all, mask=mask_all, y_mask=ymask_all,
#         meta=meta_array,
#         features_per_slot=np.array(FEATURES_PER_SLOT, dtype=np.int32),
#         n_slots=np.array(N_SLOTS, dtype=np.int32),
#         globals_desc=np.array(["R", "arrival_rate"], dtype=object),
#         d_safe=np.array(D_SAFE_M, dtype=np.float32),
#         s_dist=np.array(S_DIST_M, dtype=np.float32),
#         dt=np.array(DT, dtype=np.float32),
#         accel_min=np.array(ACCEL_MIN, dtype=np.float32),
#         accel_max=np.array(ACCEL_MAX, dtype=np.float32),
#     )
#
#     # Summary
#     expect_x_dim = N_SLOTS * FEATURES_PER_SLOT + GLOBAL_FEATURES_DIM
#     print("--------------------------------------------------")
#     print(f"[done] wrote {args.out}")
#     print(f"[stats] files read (with frames): {total_files_read}")
#     print(f"[stats] samples total:            {X_all.shape[0]}")
#     print(f"[stats] X shape:                  {X_all.shape} (expect last dim = {expect_x_dim})")
#     print(f"[stats] Y/mask shapes:            {Y_all.shape}, {mask_all.shape}, {ymask_all.shape}")
#     print(f"[stats] meta shape:               {meta_array.shape}")
#     if X_all.shape[0] > 0 and X_all.shape[1] != expect_x_dim:
#         print("[warn] X last-dim != expected; check FEATURES_PER_SLOT / GLOBAL_FEATURES_DIM / N_SLOTS")
#
#     # Quick invariant spot-check on a few rows
#     if X_all.shape[0] > 0:
#         print("\n[sanity] validating first 3 samples…")
#         for i in range(min(3, X_all.shape[0])):
#             validate_sample(X_all[i], Y_all[i], mask_all[i], ymask_all[i], FEATURES_PER_SLOT)
#         print("[sanity] done.")
#
# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import re
import json
import math
import argparse
from collections import OrderedDict, Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np

# ===================== CONFIG: DATA BUILD =====================
DATASETS = OrderedDict([
    (r"$\Lambda=0.9 $", r"C:\Users\ghoriss\Documents\IBEX\Longduration"),
    (r"$\Lambda=0.7 $", r"C:\Users\ghoriss\Documents\IBEX\Longduration_1_low"),
    (r"$\Lambda=0.5 $", r"C:\Users\ghoriss\Documents\IBEX\Longduration_2_low"),
])

RADII_FOLDERS = ["R_40", "R_60", "R_80", "R_100", "R_120", "R_140", "R_160", "R_180", "R_200"]

FILENAME_PATTERN = re.compile(r"vehicles_positions_id_R_\d+.*\.json$", re.IGNORECASE)

# ===================== FEATURES / TARGETS =====================
# Per-agent slot features (12):
# [pos, vel, flow, dist, same_gap, has_same, cross_sum, has_cross, cross_mgn, cross_safe, gap_mgn, gap_safe]
FEATURES_PER_SLOT = 12
GLOBAL_FEATURES_DIM = 2  # [R, arrival_rate] appended once per sample

# Safety semantics
D_SAFE_M = 3.0   # same-lane headway threshold
S_DIST_M = 4.0   # cross-flow sum-of-distances-to-center threshold

# Target from t -> t+1
DT = 0.1
ACCEL_MIN, ACCEL_MAX = -5.0, 5.0

# Validation toggle
DISABLE_VALIDATION = False

# ===================== HELPERS (old-format JSON) =====================
def load_json(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_inside_by_structure(rec) -> bool:
    """Old schema: inside-zone records use nested singleton lists for numeric fields."""
    return isinstance(rec, list) and len(rec) >= 6 and isinstance(rec[0], list)

def normalize_record(rec):
    """
    Returns unified (x, y, vx, vy, flow, vid) for both inside/outside records.
    Old logs store a 'd' field indicating lane: 'x'/0 for x-lane, 'y'/1 for y-lane.
    """
    if is_inside_by_structure(rec):
        x, y = rec[0][0], rec[1][0]
        vx, vy = rec[2][0], rec[3][0]
        d, vid = rec[4], rec[5]
    else:
        x, y, vx, vy, d, vid = rec
    if isinstance(d, str):
        flow = 0 if d.lower().startswith('x') else 1
    else:
        flow = 0 if int(d) == 0 else 1
    return float(x), float(y), float(vx), float(vy), int(flow), int(vid)

def along_lane_pos_vel(x, y, vx, vy, flow):
    """Return (pos_along_lane, vel_along_lane) for the agent's lane axis."""
    return (x, vx) if flow == 0 else (y, vy)

def find_all_json_files(root_dir: str):
    """Yield (radius_folder_name, full_path) in the specified RADII order for a given dataset root."""
    for folder_name in RADII_FOLDERS:
        radius_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(radius_path):
            print(f"[warn] Missing folder: {radius_path}")
            continue
        for root, _, files in os.walk(radius_path):
            for filename in files:
                if FILENAME_PATTERN.match(filename):
                    yield folder_name, os.path.join(root, filename)

def parse_lambda_from_label(lbl: str) -> float:
    """Extract Λ from labels like '$\\Lambda=0.9 $'. Fallback: first float."""
    try:
        s = lbl.replace("\\", "").replace("$", "").replace("Lambda", "Λ").replace(" ", "")
        if "Λ=" in s:
            return float(s.split("Λ=")[1])
    except Exception:
        pass
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lbl)
    return float(m.group(0)) if m else -1.0

def parse_radius_from_folder(folder_name: str) -> float:
    # "R_120" -> 120.0
    try:
        return float(folder_name.split('_')[1])
    except Exception:
        return -1.0

# ===================== FIRST PASS: determine N_SLOTS =====================
def first_pass_max_inside(DATASETS: OrderedDict) -> Tuple[int, int, Counter]:
    """
    Scan all datasets to compute:
      - max number of inside-zone agents in any timestep (N_SLOTS)
      - total number of matched JSON files
      - histogram of concurrent inside agents
    """
    max_agents = 0
    count_files = 0
    hist = Counter()
    for _, root_dir in DATASETS.items():
        for _, fp in find_all_json_files(root_dir):
            count_files += 1
            data = load_json(fp)
            if not isinstance(data, list):
                continue
            for t_list in data:
                if not t_list:
                    hist[0] += 1
                    continue
                inside = [rec for rec in t_list if is_inside_by_structure(rec)]
                n = len(inside)
                hist[n] += 1
                if n > max_agents:
                    max_agents = n
    return max_agents, count_files, hist

def percentile_from_hist(hist: Counter, p: float) -> int:
    """p-th percentile of a discrete histogram {value:count}."""
    items = sorted(hist.items())
    total = sum(c for _, c in items)
    if total == 0:
        return 0
    threshold = p * total
    cum = 0
    for val, cnt in items:
        cum += cnt
        if cum >= threshold:
            return val
    return items[-1][0]

# ===================== NEIGHBOR FEATURES & SAFETY =====================
def same_and_cross_features(pos: np.ndarray, mask: np.ndarray, flows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given aligned slot arrays (length S):
      pos   : along-lane position (signed)
      mask  : 1 if real agent, else 0
      flows : 0 for x-lane, 1 for y-lane for real agents
    Returns (same_gap_min, has_same, cross_sum_min, has_cross), each length S.
    """
    S = pos.shape[0]
    abs_pos = np.abs(pos).astype(np.float32)

    gap_same_min       = np.zeros((S,), dtype=np.float32)
    cross_sum_min      = np.zeros((S,), dtype=np.float32)
    has_same_neighbor  = np.zeros((S,), dtype=np.float32)
    has_cross_neighbor = np.zeros((S,), dtype=np.float32)

    idx_x = [i for i in range(S) if mask[i] > 0.5 and int(flows[i]) == 0]
    idx_y = [i for i in range(S) if mask[i] > 0.5 and int(flows[i]) == 1]

    # same-lane nearest neighbors in 1D (adjacent after sorting by pos)
    def fill_same(indices: List[int]):
        n = len(indices)
        if n <= 1:
            return
        arr = sorted(indices, key=lambda i: pos[i])
        for k in range(1, n):
            i_prev, i_curr = arr[k-1], arr[k]
            d = abs(pos[i_curr] - pos[i_prev])
            # update both sides
            if has_same_neighbor[i_curr] == 0 or d < gap_same_min[i_curr]:
                gap_same_min[i_curr] = d
                has_same_neighbor[i_curr] = 1.0
            if has_same_neighbor[i_prev] == 0 or d < gap_same_min[i_prev]:
                gap_same_min[i_prev] = d
                has_same_neighbor[i_prev] = 1.0

    fill_same(idx_x)
    fill_same(idx_y)

    # cross-lane min of |pos_i| + |pos_j|
    def fill_cross(curr: List[int], other: List[int]):
        if not other:
            return
        min_other_abs = float(np.min(abs_pos[other]))
        for i in curr:
            cross_sum_min[i] = abs_pos[i] + min_other_abs
            has_cross_neighbor[i] = 1.0

    fill_cross(idx_x, idx_y)
    fill_cross(idx_y, idx_x)

    return gap_same_min, has_same_neighbor, cross_sum_min, has_cross_neighbor

# ===================== VALIDATION =====================
def validate_sample(X_row: np.ndarray, Y_row: np.ndarray, mask: np.ndarray, y_mask: np.ndarray, features_per_slot: int) -> None:
    if DISABLE_VALIDATION:
        return
    try:
        S = mask.shape[0]
        expect_feat_len = S * features_per_slot
        X_slots = X_row[:expect_feat_len].reshape(S, features_per_slot)
        valid = mask.astype(bool)

        # y_mask ⊆ mask
        if not np.all(y_mask <= mask):
            print("[warn] y_mask has 1s where mask has 0s (should be subset).")

        # Y bounds on present agents
        if valid.any():
            yv = Y_row[:S][valid]
            if (yv < ACCEL_MIN - 1e-6).any() or (yv > ACCEL_MAX + 1e-6).any():
                print(f"[warn] Y out of bounds: [{yv.min():.3g}, {yv.max():.3g}]")

        # padded slots must be zero
        inv = ~valid
        if inv.any() and np.where(np.any(np.abs(X_slots[inv]) > 1e-6, axis=1))[0].size > 0:
            print("[warn] non-zero features in padded slots")

        # boolean columns must be {0,1}
        bool_cols = [5, 7, 9, 11]
        if np.any(~np.isin(X_slots[:, bool_cols], [0.0, 1.0])):
            print("[warn] non-binary values in boolean feature columns")

        # dist ≈ |pos|
        pos = X_slots[:, 0]
        dist = X_slots[:, 3]
        if np.where(np.abs(dist - np.abs(pos)) > 1e-5)[0].size > 0:
            print("[warn] dist != abs(pos) found")

        # safety flags consistent with margins if neighbor exists
        has_cross = X_slots[:, 7]
        cross_m  = X_slots[:, 8]
        cross_s  = X_slots[:, 9]
        if np.where((has_cross == 1.0) & ((cross_m > 0) != (cross_s == 1.0)))[0].size > 0:
            print("[warn] cross_safe inconsistent with cross_mgn/has_cross")

        has_same = X_slots[:, 5]
        gap_m    = X_slots[:, 10]
        gap_s    = X_slots[:, 11]
        if np.where((has_same == 1.0) & ((gap_m > 0) != (gap_s == 1.0)))[0].size > 0:
            print("[warn] gap_safe inconsistent with gap_mgn/has_same")

    except Exception as e:
        print(f"[warn] validation error ignored: {e}")

# ===================== BUILD ONE DATASET (old schema) =====================
def build_for_one_dataset(root_dir: str, arrival_rate: float, N_SLOTS: int):
    """
    Build arrays for a single Λ-root, using old JSON schema.
    Returns: X, Y, mask, y_mask, meta_rows
    """
    X_list, Y_list, M_list, YM_list = [], [], [], []
    meta_rows: List[Dict[str, Any]] = []

    cov_total_slots = 0
    cov_real_slots  = 0
    cov_valid_slots = 0
    rows_skipped_no_targets = 0
    last_frame_rows = 0
    overflow_drops  = 0
    accel_nan_inf   = 0
    feat_nan_inf    = 0
    flow_flip_count = 0

    per_file_cov = defaultdict(lambda: [0.0, 0.0])  # valid / real
    per_R_cov    = defaultdict(lambda: [0.0, 0.0])
    per_L_cov    = defaultdict(lambda: [0.0, 0.0])

    for radius_folder, fp in find_all_json_files(root_dir):
        R_val = parse_radius_from_folder(radius_folder)
        data = load_json(fp)
        if not isinstance(data, list):
            print(f"[warn] {fp} top-level JSON not a list; skipping")
            continue
        T = len(data)
        if T <= 1:
            continue

        # stable slot mapping within file
        slot_map: Dict[int, int] = {}   # vid -> slot index
        free_slots = list(range(N_SLOTS))

        for t_idx in range(T):
            if t_idx == T - 1:
                last_frame_rows += 1
                continue

            t_list  = data[t_idx]
            t1_list = data[t_idx + 1]
            if not t_list:
                continue

            inside_t  = [rec for rec in t_list  if is_inside_by_structure(rec)]
            inside_t1 = [rec for rec in t1_list if is_inside_by_structure(rec)]
            if not inside_t:
                continue

            # normalize current inside agents
            agents_now = []
            vids_now = set()
            for rec in inside_t:
                x, y, vx, vy, flow, vid = normalize_record(rec)
                flow = 0 if flow == 0 else 1
                agents_now.append({'vid': vid, 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'flow': flow})
                vids_now.add(vid)

            # lookup next-step by vid
            next_map: Dict[int, Dict] = {}
            for rec in inside_t1:
                x, y, vx, vy, flow, vid = normalize_record(rec)
                flow = 0 if flow == 0 else 1
                next_map[vid] = {'vid': vid, 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'flow': flow}

            # free slots for departed agents
            missing = [vid for vid in list(slot_map.keys()) if vid not in vids_now]
            if missing:
                import bisect
                for vid in missing:
                    s = slot_map.pop(vid)
                    bisect.insort(free_slots, s)

            # allocate arrays
            feats = np.zeros((N_SLOTS, FEATURES_PER_SLOT), dtype=np.float32)
            mask  = np.zeros((N_SLOTS,), dtype=np.float32)
            flows = np.full((N_SLOTS,), -1, dtype=np.int64)
            yrow  = np.zeros((N_SLOTS,), dtype=np.float32)
            ymask = np.zeros((N_SLOTS,), dtype=np.float32)

            # fill base cols: pos, vel, flow, dist
            for a in agents_now:
                vid = a['vid']
                # slot assignment (stable)
                if vid in slot_map:
                    s = slot_map[vid]
                else:
                    if not free_slots:
                        overflow_drops += 1
                        continue
                    s = free_slots.pop(0)
                    slot_map[vid] = s

                pos_along, vel_along = along_lane_pos_vel(a['x'], a['y'], a['vx'], a['vy'], a['flow'])
                dist_c = abs(pos_along)  # 1D distance spec for this dataset

                feats[s, 0:4] = [pos_along, vel_along, float(a['flow']), dist_c]
                flows[s] = a['flow']
                mask[s] = 1.0

            # neighbors & safety (only for real slots)
            if mask.sum() > 0:
                pos = feats[:, 0]
                gap_same_min, has_same, cross_sum_min, has_cross = same_and_cross_features(pos, mask, flows)

                feats[:, 4] = gap_same_min
                feats[:, 5] = has_same
                feats[:, 6] = cross_sum_min
                feats[:, 7] = has_cross

                # cross safety margins / flags
                cross_mgn  = np.zeros((N_SLOTS,), dtype=np.float32)
                cross_safe = np.zeros((N_SLOTS,), dtype=np.float32)
                valid_c = has_cross > 0.5
                cross_mgn[valid_c]  = cross_sum_min[valid_c] - S_DIST_M
                cross_safe[valid_c] = (cross_mgn[valid_c] > 0.0).astype(np.float32)
                feats[:, 8] = cross_mgn
                feats[:, 9] = cross_safe

                # same-lane safety margins / flags
                gap_mgn  = np.zeros((N_SLOTS,), dtype=np.float32)
                gap_safe = np.zeros((N_SLOTS,), dtype=np.float32)
                valid_g = has_same > 0.5
                gap_mgn[valid_g]  = gap_same_min[valid_g] - D_SAFE_M
                gap_safe[valid_g] = (gap_mgn[valid_g] > 0.0).astype(np.float32)
                feats[:, 10] = gap_mgn
                feats[:, 11] = gap_safe

            # targets: accel if persists to t+1
            real_slots = int(mask.sum())
            cov_total_slots += N_SLOTS
            cov_real_slots  += real_slots

            # reverse slot_map once (small S, trivial cost)
            rev_map = {s: vid for vid, s in slot_map.items()}

            for s in range(N_SLOTS):
                if mask[s] < 0.5:
                    continue
                vid_s = rev_map.get(s, None)
                if vid_s is None:
                    continue
                if vid_s in next_map:
                    flow_now = int(feats[s, 2])
                    b = next_map[vid_s]
                    flow_next = int(b['flow'])
                    if flow_now != flow_next:
                        flow_flip_count += 1
                    v_now = feats[s, 1]
                    _, v_next = along_lane_pos_vel(b['x'], b['y'], b['vx'], b['vy'], b['flow'])
                    acc = (v_next - v_now) / DT
                    if not np.isfinite(acc):
                        accel_nan_inf += 1
                        acc = 0.0
                    acc = max(ACCEL_MIN, min(ACCEL_MAX, float(acc)))
                    yrow[s]  = acc
                    ymask[s] = 1.0

            if ymask.sum() == 0:
                rows_skipped_no_targets += 1
                continue

            cov_valid_slots += int(ymask.sum())

            # flatten + append global features [R, Λ]
            x_vec = feats.flatten()
            if not np.isfinite(x_vec).all():
                feat_nan_inf += int(np.size(x_vec) - np.isfinite(x_vec).sum())
                x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=0.0, neginf=0.0)
            x_vec = np.hstack([x_vec, np.array([R_val, arrival_rate], dtype=np.float32)])

            X_list.append(x_vec)
            Y_list.append(yrow)
            M_list.append(mask)
            YM_list.append(ymask)
            meta_rows.append({
                "file": fp,
                "frame_idx": t_idx,
                "R": R_val,
                "arrival_rate": arrival_rate,
                "dataset_label": f"Λ={arrival_rate}",
                "radius_folder": radius_folder
            })

            # coverage buckets
            per_file_cov[fp][0] += float(ymask.sum())
            per_file_cov[fp][1] += float(mask.sum())
            per_R_cov[R_val][0] += float(ymask.sum())
            per_R_cov[R_val][1] += float(mask.sum())
            per_L_cov[arrival_rate][0] += float(ymask.sum())
            per_L_cov[arrival_rate][1] += float(mask.sum())

    # stack
    if X_list:
        X = np.vstack(X_list).astype(np.float32)
        Y = np.vstack(Y_list).astype(np.float32)
        mask = np.vstack(M_list).astype(np.float32)
        y_mask = np.vstack(YM_list).astype(np.float32)
    else:
        X = np.zeros((0, N_SLOTS * FEATURES_PER_SLOT + GLOBAL_FEATURES_DIM), dtype=np.float32)
        Y = np.zeros((0, N_SLOTS), dtype=np.float32)
        mask = np.zeros((0, N_SLOTS), dtype=np.float32)
        y_mask = np.zeros((0, N_SLOTS), dtype=np.float32)

    # coverage prints
    total_slots = X.shape[0] * N_SLOTS
    overall_cov = (y_mask.sum() / total_slots) if total_slots > 0 else 0.0
    real_cov    = (y_mask.sum() / np.clip(mask.sum(), 1, None)) if mask.size > 0 else 0.0

    print("[coverage] overall valid targets ratio = {:.4f}".format(overall_cov))
    print("[coverage] valid / real-agent slots    = {:.4f}  (valid={} / real={})"
          .format(real_cov, int(y_mask.sum()), int(mask.sum())))
    print("[coverage] counters: total_slots={} real_slots={} valid_slots={}"
          .format(cov_total_slots, cov_real_slots, cov_valid_slots))
    print("[info] rows skipped (no targets)       =", rows_skipped_no_targets)
    print("[info] last-frame rows (no t+1)        =", last_frame_rows)
    print("[warn] slot overflows (agent dropped)   =", overflow_drops)
    print("[warn] flow flips between t and t+1     =", flow_flip_count)
    print("[audit] feature NaN/Inf replaced        =", feat_nan_inf)
    print("[audit] target  NaN/Inf replaced        =", accel_nan_inf)

    if mask.size > 0:
        bad = np.logical_and(mask < 0.5, y_mask > 0.5).sum()
        if bad > 0:
            print(f"[ERROR] y_mask has {int(bad)} entries where mask==0; check builder logic!")

    # show a few file coverages
    print("\n[coverage/by-file] (first 10)")
    for i, (f, (v, r)) in enumerate(per_file_cov.items()):
        if i >= 10: break
        ratio = (v / r) if r > 0 else 0.0
        short = os.path.basename(f)
        print(f"  {short:40s} real={int(r):8d} valid={int(v):8d} ratio={ratio:.3f}")

    if per_L_cov:
        print("\n[coverage/by-Λ]")
        for lam in sorted(per_L_cov.keys()):
            v, r = per_L_cov[lam]
            print(f"  Λ={lam:.3f}: real={int(r):8d} valid={int(v):8d} ratio={(v/r) if r>0 else 0.0:.3f}")

    if per_R_cov:
        print("\n[coverage/by-R]")
        for Rv in sorted(per_R_cov.keys()):
            v, r = per_R_cov[Rv]
            print(f"  R={Rv:5.1f}: real={int(r):8d} valid={int(v):8d} ratio={(v/r) if r>0 else 0.0:.3f}")

    return X, Y, mask, y_mask, meta_rows

# ===================== STORAGE HELPERS (compact + shards) =====================
def cast_for_storage(X: np.ndarray, Y: np.ndarray, mask: np.ndarray, y_mask: np.ndarray):
    """Shrink arrays for disk: X,Y -> float16; masks -> uint8."""
    Xc = X.astype(np.float16, copy=False)
    Yc = Y.astype(np.float16, copy=False)
    Mc = mask.astype(np.uint8, copy=False)
    YMc = y_mask.astype(np.uint8, copy=False)
    return Xc, Yc, Mc, YMc

def chunks(total: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        yield start, end

def save_npz_single(path: str,
                    X, Y, mask, y_mask, meta_array,
                    features_per_slot: int, n_slots: int):
    np.savez_compressed(
        path,
        X=X, Y=Y, mask=mask, y_mask=y_mask,
        meta=meta_array,
        features_per_slot=np.array(features_per_slot, dtype=np.int32),
        n_slots=np.array(n_slots, dtype=np.int32),
        globals_desc=np.array(["R", "arrival_rate"], dtype=object),
        d_safe=np.array(D_SAFE_M, dtype=np.float32),
        s_dist=np.array(S_DIST_M, dtype=np.float32),
        dt=np.array(DT, dtype=np.float32),
        accel_min=np.array(ACCEL_MIN, dtype=np.float32),
        accel_max=np.array(ACCEL_MAX, dtype=np.float32),
    )
    print(f"[done] wrote {path}")

def save_npz_sharded(base_stem: str,
                     X, Y, mask, y_mask, meta_array,
                     features_per_slot: int, n_slots: int,
                     rows_per_shard: int = 100_000) -> List[str]:
    paths = []
    N = X.shape[0]
    ndigits = max(3, int(math.log10(max(1, math.ceil(N / rows_per_shard)))) + 1)
    if N == 0:
        out = f"{base_stem}.part{0:0{ndigits}d}.npz"
        save_npz_single(out, X, Y, mask, y_mask, meta_array, features_per_slot, n_slots)
        return [out]
    for s, e in chunks(N, rows_per_shard):
        out = f"{base_stem}.part{s:0{ndigits}d}-{e-1:0{ndigits}d}.npz"
        save_npz_single(out, X[s:e], Y[s:e], mask[s:e], y_mask[s:e], meta_array[s:e], features_per_slot, n_slots)
        paths.append(out)
    return paths

# ===================== MAIN =====================
def main():
    ap = argparse.ArgumentParser(description="Build combined NPZ from OLD-format logs (list-of-frames).")
    ap.add_argument("--out", type=str, default="inputs_targets_dataset_XY_ALL.npz", help="Output NPZ filename or stem")
    ap.add_argument("--out-dir", type=str, default=".", help="Directory for outputs")
    ap.add_argument("--rows-per-shard", type=int, default=100_000, help="Rows per shard when not using --single")
    ap.add_argument("--single", action="store_true", help="Write a single NPZ instead of sharded parts (requires enough disk space)")
    args = ap.parse_args()

    # Pass 1: discover N_SLOTS
    N_SLOTS, nfiles, hist = first_pass_max_inside(DATASETS)
    if N_SLOTS <= 0:
        print("[fatal] No inside-zone agents found across datasets. Check JSON format/paths.")
        return
    print(f"[info] scanned {nfiles} files; max inside agents per snapshot = {N_SLOTS}")
    p95 = percentile_from_hist(hist, 0.95)
    mean_occ = sum(k*v for k, v in hist.items()) / max(1, sum(hist.values()))
    print("[padding] concurrent-agents histogram (top bins):")
    for k, c in sorted(hist.items())[:10]:
        print(f"  n={k:2d}: count={c}")
    print(f"[padding] mean ≈ {mean_occ:.2f}, 95th percentile ≈ {p95}")
    print(f"[note] using N_SLOTS = max = {N_SLOTS} (no truncation)")

    all_X, all_Y, all_mask, all_y_mask = [], [], [], []
    all_meta: List[Dict[str, Any]] = []
    total_files_read = 0

    # Pass 2: build per-Λ and concatenate
    for ds_label, root_dir in DATASETS.items():
        lam = parse_lambda_from_label(ds_label)
        print(f"\n[build] Dataset {ds_label} (Λ={lam}) @ {root_dir}")
        X, Y, mask, y_mask, meta_rows = build_for_one_dataset(root_dir, lam, N_SLOTS)

        all_X.append(X); all_Y.append(Y); all_mask.append(mask); all_y_mask.append(y_mask)
        all_meta.extend(meta_rows)
        total_files_read += len({m['file'] for m in meta_rows})

    # Concatenate
    if all_X:
        X_all     = np.vstack(all_X).astype(np.float32)
        Y_all     = np.vstack(all_Y).astype(np.float32)
        mask_all  = np.vstack(all_mask).astype(np.float32)
        ymask_all = np.vstack(all_y_mask).astype(np.float32)
    else:
        X_all     = np.zeros((0, N_SLOTS * FEATURES_PER_SLOT + GLOBAL_FEATURES_DIM), dtype=np.float32)
        Y_all     = np.zeros((0, N_SLOTS), dtype=np.float32)
        mask_all  = np.zeros((0, N_SLOTS), dtype=np.float32)
        ymask_all = np.zeros((0, N_SLOTS), dtype=np.float32)

    meta_array = np.array(all_meta, dtype=object)

    # Cast for compact storage
    Xc, Yc, Mc, YMc = cast_for_storage(X_all, Y_all, mask_all, ymask_all)

    # Save (single or sharded)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If user provided ".npz", use stem; else treat as stem already
    out_name = Path(args.out)
    stem = out_name.stem if out_name.suffix.lower() == ".npz" else out_name.name
    out_stem_full = str(out_dir / stem)

    expect_x_dim = N_SLOTS * FEATURES_PER_SLOT + GLOBAL_FEATURES_DIM
    print("--------------------------------------------------")
    print(f"[done build] files read (with frames): {total_files_read}")
    print(f"[stats] samples total:            {X_all.shape[0]}")
    print(f"[stats] X shape:                  {X_all.shape} (expect last dim = {expect_x_dim})")
    print(f"[stats] Y/mask shapes:            {Y_all.shape}, {mask_all.shape}, {ymask_all.shape}")
    print(f"[stats] meta shape:               {meta_array.shape}")
    if X_all.shape[0] > 0 and X_all.shape[1] != expect_x_dim:
        print("[warn] X last-dim != expected; check FEATURES_PER_SLOT / GLOBAL_FEATURES_DIM / N_SLOTS")
    if X_all.shape[0] != meta_array.shape[0]:
        print("[warn] X row count != meta row count; data is misaligned!")

    try:
        if args.single:
            # Single file (may fail if disk space is low)
            out_path = f"{out_stem_full}.npz"
            save_npz_single(out_path, Xc, Yc, Mc, YMc, meta_array, FEATURES_PER_SLOT, N_SLOTS)
        else:
            # Safer default: sharded outputs
            parts = save_npz_sharded(out_stem_full, Xc, Yc, Mc, YMc, meta_array,
                                     FEATURES_PER_SLOT, N_SLOTS, rows_per_shard=args.rows_per_shard)
            print(f"[done] wrote {len(parts)} shard(s).")
    except OSError as e:
        print(f"[error] write failed: {e}")
        print("       Tip: use a larger --out-dir, reduce --rows-per-shard, or add --single on a drive with space.")

    # Quick invariant spot-check on a few rows
    if X_all.shape[0] > 0:
        print("\n[sanity] validating first 3 samples…")
        for i in range(min(3, X_all.shape[0])):
            validate_sample(X_all[i], Y_all[i], mask_all[i], ymask_all[i], FEATURES_PER_SLOT)
        print("[sanity] done.")

if __name__ == "__main__":
    main()
