# # ===============================================================
# # Build XY intersection dataset with cross- and same-lane safety margins/flags
# # ===============================================================
# import os
# import re
# import json
# import math
# import numpy as np
# from collections import OrderedDict
# from typing import Dict, Tuple, List
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
# # ---------- PER-AGENT FEATURE LAYOUT (12) ----------
# # [pos_along, vel_along, flow_id, dist_to_center,
# #  gap_same_min, has_same,
# #  cross_sum_min, has_cross,
# #  cross_margin, cross_is_safe,
# #  gap_margin,  gap_is_safe]
# FEAT_PER_AGENT = 12
#
# # Global features appended once per sample: [R, ARRIVAL_RATE]
# INCLUDE_GLOBAL_FEATURES = True
# GLOBAL_DIM = 2 if INCLUDE_GLOBAL_FEATURES else 0
#
# # Targets: per-agent along-lane acceleration from t -> t+1
# DT = 0.1              # seconds (your sim step)
# T_MAX_PER_FILE = None # optional cap; set None to use full file
#
# # Safety constants (used for features & metadata)
# D_SAFE_SAME   = 3.0   # meters, same-lane minimum headway (USED)
# S_DIST_CROSS  = 4.0   # meters, cross-lane sum-of-distances-to-center threshold (USED)
# # ===============================================================
#
#
# # ------------------ Helpers (Data Build) ------------------
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
#
# def parse_radius_from_folder(folder_name: str) -> float:
#     # "R_120" -> 120.0
#     try:
#         return float(folder_name.split('_')[1])
#     except Exception:
#         return -1.0
#
#
# def is_inside_by_structure(rec) -> bool:
#     """Inside-zone records use nested singleton lists for numeric fields."""
#     return isinstance(rec[0], list)
#
#
# def normalize_record(rec):
#     """
#     Returns unified (x, y, vx, vy, flow_id (0 for x, 1 for y), vid).
#     Works for both flat (outside) and nested (inside) records.
#     """
#     if is_inside_by_structure(rec):
#         x, y = rec[0][0], rec[1][0]
#         vx, vy = rec[2][0], rec[3][0]
#         d, vid = rec[4], rec[5]
#     else:
#         x, y, vx, vy, d, vid = rec
#
#     flow = 0 if (isinstance(d, str) and str(d).lower().startswith('x')) or d == 0 else 1
#     return float(x), float(y), float(vx), float(vy), flow, int(vid)
#
#
# def along_lane_pos_vel(x, y, vx, vy, flow):
#     """Return (pos_along_lane, vel_along_lane) for the agent's lane axis."""
#     if flow == 0:   # x-lane (east/west)
#         return x, vx
#     else:           # y-lane (north/south)
#         return y, vy
#
#
# def load_json(filepath: str):
#     with open(filepath, 'r') as f:
#         return json.load(f)
#
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
#
# # --------------- First pass: padding size ---------------
# def first_pass_max_agents_across_datasets() -> Tuple[int, int]:
#     """
#     Scan all DATASETS and radii to compute:
#       - max number of inside-zone agents in any timestep
#       - total number of matched JSON files
#     """
#     max_agents = 0
#     count_files = 0
#     for _, root_dir in DATASETS.items():
#         for _, fp in find_all_json_files(root_dir):
#             count_files += 1
#             data = load_json(fp)  # outer list: timesteps
#             T = len(data)
#             T_use = min(T, T_MAX_PER_FILE) if T_MAX_PER_FILE is not None else T
#             for ti in range(T_use):
#                 t_list = data[ti]
#                 if not t_list:
#                     continue
#                 inside = [rec for rec in t_list if is_inside_by_structure(rec)]
#                 if not inside:
#                     continue
#                 max_agents = max(max_agents, len(inside))
#     return max_agents, count_files
#
#
# # ---------- Neighborhood features + safety margins/flags ----------
# def compute_same_lane_min_gap_and_cross_min(
#     feats: np.ndarray, mask: np.ndarray, flows: np.ndarray, max_agents: int
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Inputs per timestep (aligned by slots):
#       feats: [max_agents, >=4] where cols 0..3 are [pos, vel, flow, dist_c]
#       mask : [max_agents] (1 if real agent)
#       flows: [max_agents] (0 or 1 for x- or y-lane)
#
#     Returns (float32 vectors, length=max_agents):
#       gap_same_min      : min_{j in same lane, j!=i} |pos_i - pos_j| (0.0 if none)
#       has_same_neighbor : 1.0 if same-lane neighbor exists, else 0.0
#       cross_sum_min     : min_{j in cross lane} (|pos_i| + |pos_j|) (0.0 if none)
#       has_cross_neighbor: 1.0 if cross-lane neighbor exists, else 0.0
#     """
#     pos = feats[:, 0]
#     abs_pos = np.abs(pos).astype(np.float32)
#
#     gap_same_min       = np.zeros((max_agents,), dtype=np.float32)
#     cross_sum_min      = np.zeros((max_agents,), dtype=np.float32)
#     has_same_neighbor  = np.zeros((max_agents,), dtype=np.float32)
#     has_cross_neighbor = np.zeros((max_agents,), dtype=np.float32)
#
#     idx_x = [i for i in range(max_agents) if mask[i] > 0.5 and flows[i] == 0]
#     idx_y = [i for i in range(max_agents) if mask[i] > 0.5 and flows[i] == 1]
#
#     # same-lane nearest neighbors in 1D
#     def fill_same_lane_min(indices: List[int]):
#         n = len(indices)
#         if n <= 1:
#             return
#         arr = sorted(indices, key=lambda i: pos[i])
#         for k in range(1, n):
#             i_prev, i_curr = arr[k - 1], arr[k]
#             d = abs(pos[i_curr] - pos[i_prev])
#             # Update both neighbors
#             if has_same_neighbor[i_curr] == 0 or d < gap_same_min[i_curr]:
#                 gap_same_min[i_curr] = d
#                 has_same_neighbor[i_curr] = 1.0
#             if has_same_neighbor[i_prev] == 0 or d < gap_same_min[i_prev]:
#                 gap_same_min[i_prev] = d
#                 has_same_neighbor[i_prev] = 1.0
#
#     fill_same_lane_min(idx_x)
#     fill_same_lane_min(idx_y)
#
#     # cross-lane min of |pos_i| + |pos_j|
#     def fill_cross_min(curr_indices: List[int], other_indices: List[int]):
#         if not other_indices:
#             return
#         min_other_abs = float(np.min(abs_pos[other_indices]))
#         for i in curr_indices:
#             cross_sum_min[i] = abs_pos[i] + min_other_abs
#             has_cross_neighbor[i] = 1.0
#
#     fill_cross_min(idx_x, idx_y)
#     fill_cross_min(idx_y, idx_x)
#
#     return gap_same_min, has_same_neighbor, cross_sum_min, has_cross_neighbor
#
#
# # --------------- Build dataset for a single Λ ---------------
# def build_for_one_dataset(root_dir: str, arrival_rate: float, max_agents: int):
#     """
#     Build:
#       - X       : [num_samples, max_agents*FEAT_PER_AGENT + GLOBAL_DIM]
#       - mask    : [num_samples, max_agents] (1=real agent in X at time t, 0=padding)
#       - ids     : [num_samples, max_agents] (agent global IDs, -1 for padding)
#       - Y       : [num_samples, max_agents] (accel along lane for agents that persist to t+1, else 0)
#       - y_mask  : [num_samples, max_agents] (1=agent exists at t and t+1; 0=otherwise)
#       - meta    : list of dicts with file, radius, t_index, num_agents, etc.
#     """
#     X_list, M_list, ID_list, Y_list, YM_list, meta = [], [], [], [], [], []
#
#     for folder_name, fp in find_all_json_files(root_dir):
#         R_val = parse_radius_from_folder(folder_name)
#         data = load_json(fp)
#         T = len(data)
#         T_use = min(T, T_MAX_PER_FILE) if T_MAX_PER_FILE is not None else T
#
#         # Maintain stable slot assignment for this file
#         slot_map: Dict[int, int] = {}       # vid -> slot index
#         free_slots = list(range(max_agents))
#
#         for t_idx in range(T_use - 1):  # need t and t+1
#             t_list   = data[t_idx]
#             t1_list  = data[t_idx + 1]
#
#             if not t_list:
#                 continue
#
#             # Inside-zone agents by structure
#             inside_t   = [rec for rec in t_list  if is_inside_by_structure(rec)]
#             inside_t1  = [rec for rec in t1_list if is_inside_by_structure(rec)]
#
#             if not inside_t:
#                 continue  # nothing to learn at this step
#
#             # Normalize current inside agents
#             agents_now = []
#             vids_now = set()
#             for rec in inside_t:
#                 x, y, vx, vy, flow, vid = normalize_record(rec)
#                 agents_now.append({'vid': vid, 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'flow': flow})
#                 vids_now.add(vid)
#
#             # Build a lookup for t+1 (by vid)
#             next_map: Dict[int, Dict] = {}
#             for rec in inside_t1:
#                 x, y, vx, vy, flow, vid = normalize_record(rec)
#                 next_map[vid] = {'vid': vid, 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'flow': flow}
#
#             # Release slots for agents that left the zone at this step
#             missing_vids = [vid for vid in list(slot_map.keys()) if vid not in vids_now]
#             if missing_vids:
#                 import bisect
#                 for vid in missing_vids:
#                     freed = slot_map.pop(vid)
#                     bisect.insort(free_slots, freed)
#
#             # Prepare arrays for this timestep
#             feats = np.zeros((max_agents, FEAT_PER_AGENT), dtype=np.float32)
#             ids   = np.full((max_agents,), -1, dtype=np.int64)
#             mask  = np.zeros((max_agents,), dtype=np.float32)
#             flows = np.full((max_agents,), -1, dtype=np.int64)  # 0 or 1 for real slots
#             yrow  = np.zeros((max_agents,), dtype=np.float32)
#             ymask = np.zeros((max_agents,), dtype=np.float32)
#
#             # Fill base 4 cols and record flow per slot
#             for a in agents_now:
#                 vid = a['vid']
#                 # keep existing slot or assign new
#                 if vid in slot_map:
#                     s = slot_map[vid]
#                 else:
#                     if not free_slots:
#                         # overflow: increase max_agents if you ever see this
#                         continue
#                     s = free_slots.pop(0)
#                     slot_map[vid] = s
#
#                 pos_along, vel_along = along_lane_pos_vel(a['x'], a['y'], a['vx'], a['vy'], a['flow'])
#                 dist_c = math.hypot(a['x'], a['y'])
#
#                 # cols: [pos, vel, flow, dist_c, (neighbor/safety features will be filled next)]
#                 feats[s, 0:4] = [pos_along, vel_along, float(a['flow']), dist_c]
#                 flows[s] = a['flow']
#                 ids[s] = vid
#                 mask[s] = 1.0
#
#             # Compute lane-structural features with validity flags + safety margins/flags
#             if mask.sum() > 0:
#                 gap_same_min, has_same, cross_min, has_cross = \
#                     compute_same_lane_min_gap_and_cross_min(feats, mask, flows, max_agents)
#
#                 # Base neighbor features
#                 feats[:, 4] = gap_same_min
#                 feats[:, 5] = has_same
#                 feats[:, 6] = cross_min
#                 feats[:, 7] = has_cross
#
#                 # --- Cross safety: margin & flag ---
#                 cross_margin   = np.zeros((max_agents,), dtype=np.float32)
#                 cross_is_safe  = np.zeros((max_agents,), dtype=np.float32)
#                 valid_cross = has_cross > 0.5
#                 cross_margin[valid_cross]  = cross_min[valid_cross] - S_DIST_CROSS
#                 cross_is_safe[valid_cross] = (cross_margin[valid_cross] >= 0.0).astype(np.float32)
#                 feats[:, 8] = cross_margin
#                 feats[:, 9] = cross_is_safe
#
#                 # --- Same-lane safety: margin & flag ---
#                 gap_margin  = np.zeros((max_agents,), dtype=np.float32)
#                 gap_is_safe = np.zeros((max_agents,), dtype=np.float32)
#                 valid_same = has_same > 0.5
#                 gap_margin[valid_same]  = gap_same_min[valid_same] - D_SAFE_SAME
#                 gap_is_safe[valid_same] = (gap_margin[valid_same] >= 0.0).astype(np.float32)
#                 feats[:, 10] = gap_margin
#                 feats[:, 11] = gap_is_safe
#
#             # Targets: accel if the agent persists to t+1
#             for s in range(max_agents):
#                 if mask[s] < 0.5:
#                     continue
#                 vid = int(ids[s])
#                 if vid in next_map:
#                     a_now_vel = feats[s, 1]
#                     b = next_map[vid]
#                     _, vel_next = along_lane_pos_vel(b['x'], b['y'], b['vx'], b['vy'], b['flow'])
#                     yrow[s]  = (vel_next - a_now_vel) / DT
#                     ymask[s] = 1.0
#
#             # Skip rows where no valid targets exist
#             if ymask.sum() == 0:
#                 continue
#
#             # Flatten and append global features [R, ARRIVAL_RATE]
#             x_vec = feats.flatten()
#             if INCLUDE_GLOBAL_FEATURES:
#                 x_vec = np.hstack([x_vec, np.array([R_val, arrival_rate], dtype=np.float32)])
#
#             # Store
#             X_list.append(x_vec)
#             M_list.append(mask)
#             ID_list.append(ids)
#             Y_list.append(yrow)
#             YM_list.append(ymask)
#             meta.append({
#                 "file": fp,
#                 "radius_folder": folder_name,
#                 "R": R_val,
#                 "arrival_rate": arrival_rate,
#                 "t_index": t_idx,
#                 "num_agents": int(mask.sum()),
#                 "num_targets": int(ymask.sum()),
#                 "d_safe_same": D_SAFE_SAME,
#                 "s_dist_cross": S_DIST_CROSS
#             })
#
#     # Stack to arrays
#     if X_list:
#         X     = np.vstack(X_list).astype(np.float32)
#         mask  = np.vstack(M_list).astype(np.float32)
#         IDs   = np.vstack(ID_list).astype(np.int64)
#         Y     = np.vstack(Y_list).astype(np.float32)
#         ymask = np.vstack(YM_list).astype(np.float32)
#     else:
#         X     = np.zeros((0, max_agents*FEAT_PER_AGENT + GLOBAL_DIM), dtype=np.float32)
#         mask  = np.zeros((0, max_agents), dtype=np.float32)
#         IDs   = np.zeros((0, max_agents), dtype=np.int64)
#         Y     = np.zeros((0, max_agents), dtype=np.float32)
#         ymask = np.zeros((0, max_agents), dtype=np.float32)
#
#     return X, mask, IDs, Y, ymask, meta
#
#
# # ------------------ Build & Save ------------------
# def build_and_save_all():
#     max_agents, nfiles = first_pass_max_agents_across_datasets()
#     if max_agents == 0:
#         raise SystemExit("No inside-zone agents found. Check your files / structure rule.")
#     print(f"[info] scanned {nfiles} file(s) across all datasets; max inside agents per snapshot = {max_agents}\n")
#
#     combined = {"X": [], "mask": [], "ids": [], "Y": [], "y_mask": [], "meta": []}
#     feat_dim_total = max_agents * FEAT_PER_AGENT + GLOBAL_DIM
#
#     for ds_label, root_dir in DATASETS.items():
#         lam = parse_lambda_from_label(ds_label)
#         print(f"[info] Building dataset for {ds_label} (Λ={lam}) at root={root_dir}")
#         X, mask, IDs, Y, ymask, meta = build_for_one_dataset(root_dir, lam, max_agents)
#
#         out_dir = os.path.join(root_dir, "processed")
#         os.makedirs(out_dir, exist_ok=True)
#         out_npz = os.path.join(out_dir, "inputs_targets_dataset_XY_new.npz")
#
#         np.savez_compressed(
#             out_npz,
#             X=X, mask=mask, ids=IDs, Y=Y, y_mask=ymask,
#             meta=np.array(meta, dtype=object),
#             feat_dim=np.array([feat_dim_total], dtype=np.int32),
#             feat_per_agent=np.array([FEAT_PER_AGENT], dtype=np.int32),
#             max_agents=np.array([max_agents], dtype=np.int32),
#             include_global=np.array([int(INCLUDE_GLOBAL_FEATURES)], dtype=np.int32),
#             global_dim=np.array([GLOBAL_DIM], dtype=np.int32),
#             dt=np.array([DT], dtype=np.float32),
#             d_safe_same=np.array([D_SAFE_SAME], dtype=np.float32),
#             s_dist_cross=np.array([S_DIST_CROSS], dtype=np.float32)
#         )
#
#         print(f"[done] Saved per-Λ file: {out_npz}")
#         print(f"        X shape    : {X.shape}  (features per sample = {feat_dim_total})")
#         print(f"        Y shape    : {Y.shape}  (per-agent accel targets)")
#         print(f"        mask shape : {mask.shape}")
#         print(f"        y_mask shape: {ymask.shape}")
#         print(f"        ids shape  : {IDs.shape}")
#
#         if X.shape[0] > 0:
#             gdim = GLOBAL_DIM
#             row0 = X[0]
#             g = row0[-gdim:] if (INCLUDE_GLOBAL_FEATURES and gdim > 0) else None
#             core = row0[:-gdim] if (INCLUDE_GLOBAL_FEATURES and gdim > 0) else row0
#             agents0 = core.reshape(max_agents, FEAT_PER_AGENT)
#             k0 = int(mask[0].sum())
#             kt0 = int(ymask[0].sum())
#             print("[peek] first sample:")
#             print("       num_agents (X) =", k0, " | num_targets (Y) =", kt0)
#             for j in range(min(k0, 5)):
#                 (pos, vel, flow, dist,
#                  gap_same_min, has_same,
#                  cross_min, has_cross,
#                  cross_margin, cross_is_safe,
#                  gap_margin, gap_is_safe) = agents0[j]
#                 y_val   = Y[0, j]
#                 y_valid = int(ymask[0, j])
#                 print(f"       slot[{j:02d}] id={IDs[0,j]} pos={pos:.3f} vel={vel:.3f} flow={int(flow)} "
#                       f"dist={dist:.3f} gap_min={gap_same_min:.3f} has_same={int(has_same)} "
#                       f"cross_min={cross_min:.3f} has_cross={int(has_cross)} "
#                       f"cross_margin={cross_margin:.3f} cross_is_safe={int(cross_is_safe)} "
#                       f"gap_margin={gap_margin:.3f} gap_is_safe={int(gap_is_safe)} "
#                       f"| a_t={y_val:.3f} (valid={y_valid})")
#             if g is not None:
#                 print(f"       global: R={g[0]:.3f}, arrival_rate={g[1]:.3f}")
#         print()
#
#         combined["X"].append(X); combined["mask"].append(mask); combined["ids"].append(IDs)
#         combined["Y"].append(Y); combined["y_mask"].append(ymask); combined["meta"].extend(meta)
#
#     # Save combined under the first dataset's root /processed
#     first_root = next(iter(DATASETS.values()))
#     comb_out_dir = os.path.join(first_root, "processed")
#     os.makedirs(comb_out_dir, exist_ok=True)
#     comb_npz = os.path.join(comb_out_dir, "inputs_targets_dataset_XY_ALL.npz")
#
#     if len(combined["X"]) > 0:
#         X_all     = np.vstack(combined["X"]).astype(np.float32)
#         mask_all  = np.vstack(combined["mask"]).astype(np.float32)
#         ids_all   = np.vstack(combined["ids"]).astype(np.int64)
#         Y_all     = np.vstack(combined["Y"]).astype(np.float32)
#         ymask_all = np.vstack(combined["y_mask"]).astype(np.float32)
#     else:
#         X_all     = np.zeros((0, max_agents*FEAT_PER_AGENT + GLOBAL_DIM), dtype=np.float32)
#         mask_all  = np.zeros((0, max_agents), dtype=np.float32)
#         ids_all   = np.zeros((0, max_agents), dtype=np.int64)
#         Y_all     = np.zeros((0, max_agents), dtype=np.float32)
#         ymask_all = np.zeros((0, max_agents), dtype=np.float32)
#
#     np.savez_compressed(
#         comb_npz,
#         X=X_all, mask=mask_all, ids=ids_all, Y=Y_all, y_mask=ymask_all,
#         meta=np.array(combined["meta"], dtype=object),
#         feat_dim=np.array([max_agents * FEAT_PER_AGENT + GLOBAL_DIM], dtype=np.int32),
#         feat_per_agent=np.array([FEAT_PER_AGENT], dtype=np.int32),
#         max_agents=np.array([max_agents], dtype=np.int32),
#         include_global=np.array([int(INCLUDE_GLOBAL_FEATURES)], dtype=np.int32),
#         global_dim=np.array([GLOBAL_DIM], dtype=np.int32),
#         dt=np.array([DT], dtype=np.float32),
#         d_safe_same=np.array([D_SAFE_SAME], dtype=np.float32),
#         s_dist_cross=np.array([S_DIST_CROSS], dtype=np.float32)
#     )
#
#     print(f"[done] Saved COMBINED dataset: {comb_npz}")
#     print(f"       X shape    : {X_all.shape}  (features per sample = {max_agents * FEAT_PER_AGENT + GLOBAL_DIM})")
#     print(f"       Y shape    : {Y_all.shape}")
#     print(f"       mask shape : {mask_all.shape}")
#     print(f"       y_mask     : {ymask_all.shape}")
#     print(f"       ids shape  : {ids_all.shape}")
#
#     return comb_npz, max_agents
#
#
# if __name__ == "__main__":
#     # Build data & save combined NPZ (per-Λ files also saved)
#     build_and_save_all()



# ===============================================================
# Build XY intersection dataset with cross- and same-lane safety margins/flags
# + Sanity checks & coverage diagnostics
# ===============================================================
import os
import re
import json
import math
import numpy as np
from collections import OrderedDict, defaultdict, Counter
from typing import Dict, Tuple, List

# ===================== CONFIG: DATA BUILD =====================
# Datasets (arrival-rate Λ) -> root folders
DATASETS = OrderedDict([
    (r"$\Lambda=0.9 $", r"C:\Users\ghoriss\Documents\IBEX\Longduration"),
    (r"$\Lambda=0.7 $", r"C:\Users\ghoriss\Documents\IBEX\Longduration_1_low"),
    (r"$\Lambda=0.5 $", r"C:\Users\ghoriss\Documents\IBEX\Longduration_2_low"),
])

# Radii subfolders to scan under each dataset root (in this exact order)
RADII_FOLDERS = ["R_40", "R_60", "R_80", "R_100", "R_120", "R_140", "R_160", "R_180", "R_200"]

# Filenames to match (case-insensitive)
FILENAME_PATTERN = re.compile(r"vehicles_positions_id_R_\d+.*\.json$", re.IGNORECASE)

# ---------- PER-AGENT FEATURE LAYOUT (12) ----------
# [pos_along, vel_along, flow_id, dist_to_center,
#  gap_same_min, has_same,
#  cross_sum_min, has_cross,
#  cross_margin, cross_is_safe,
#  gap_margin,  gap_is_safe]
FEAT_PER_AGENT = 12

# Global features appended once per sample: [R, ARRIVAL_RATE]
INCLUDE_GLOBAL_FEATURES = True
GLOBAL_DIM = 2 if INCLUDE_GLOBAL_FEATURES else 0

# Targets: per-agent along-lane acceleration from t -> t+1
DT = 0.1              # seconds (your sim step)
T_MAX_PER_FILE = None # optional cap; set None to use full file

# Safety constants (used for features & metadata)
D_SAFE_SAME   = 3.0   # meters, same-lane minimum headway (USED)
S_DIST_CROSS  = 4.0   # meters, cross-lane sum-of-distances-to-center threshold (USED)
# ===============================================================


# ------------------ Helpers (Data Build) ------------------
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


def is_inside_by_structure(rec) -> bool:
    """Inside-zone records use nested singleton lists for numeric fields."""
    return isinstance(rec[0], list)


def normalize_record(rec):
    """
    Returns unified (x, y, vx, vy, flow_id (0 for x, 1 for y), vid).
    Works for both flat (outside) and nested (inside) records.
    """
    if is_inside_by_structure(rec):
        x, y = rec[0][0], rec[1][0]
        vx, vy = rec[2][0], rec[3][0]
        d, vid = rec[4], rec[5]
    else:
        x, y, vx, vy, d, vid = rec

    flow = 0 if (isinstance(d, str) and str(d).lower().startswith('x')) or d == 0 else 1
    return float(x), float(y), float(vx), float(vy), int(flow), int(vid)


def along_lane_pos_vel(x, y, vx, vy, flow):
    """Return (pos_along_lane, vel_along_lane) for the agent's lane axis."""
    if flow == 0:   # x-lane (east/west)
        return x, vx
    else:           # y-lane (north/south)
        return y, vy


def load_json(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)


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


# --------------- First pass: padding size ---------------
def first_pass_max_agents_across_datasets() -> Tuple[int, int, dict]:
    """
    Scan all DATASETS and radii to compute:
      - max number of inside-zone agents in any timestep
      - total number of matched JSON files
      - histogram of concurrent inside agents (for padding diagnostics)
    """
    max_agents = 0
    count_files = 0
    hist = Counter()
    for _, root_dir in DATASETS.items():
        for _, fp in find_all_json_files(root_dir):
            count_files += 1
            data = load_json(fp)  # outer list: timesteps
            T = len(data)
            T_use = min(T, T_MAX_PER_FILE) if T_MAX_PER_FILE is not None else T
            for ti in range(T_use):
                t_list = data[ti]
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
    """Return the p-th percentile of a discrete count histogram."""
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


# ---------- Neighborhood features + safety margins/flags ----------
def compute_same_lane_min_gap_and_cross_min(
    feats: np.ndarray, mask: np.ndarray, flows: np.ndarray, max_agents: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs per timestep (aligned by slots):
      feats: [max_agents, >=4] where cols 0..3 are [pos, vel, flow, dist_c]
      mask : [max_agents] (1 if real agent)
      flows: [max_agents] (0 or 1 for real slots)

    Returns (float32 vectors, length=max_agents):
      gap_same_min      : min_{j in same lane, j!=i} |pos_i - pos_j| (0.0 if none)
      has_same_neighbor : 1.0 if same-lane neighbor exists, else 0.0
      cross_sum_min     : min_{j in cross lane} (|pos_i| + |pos_j|) (0.0 if none)
      has_cross_neighbor: 1.0 if cross-lane neighbor exists, else 0.0
    """
    pos = feats[:, 0]
    abs_pos = np.abs(pos).astype(np.float32)

    gap_same_min       = np.zeros((max_agents,), dtype=np.float32)
    cross_sum_min      = np.zeros((max_agents,), dtype=np.float32)
    has_same_neighbor  = np.zeros((max_agents,), dtype=np.float32)
    has_cross_neighbor = np.zeros((max_agents,), dtype=np.float32)

    idx_x = [i for i in range(max_agents) if mask[i] > 0.5 and flows[i] == 0]
    idx_y = [i for i in range(max_agents) if mask[i] > 0.5 and flows[i] == 1]

    # same-lane nearest neighbors in 1D
    def fill_same_lane_min(indices: List[int]):
        n = len(indices)
        if n <= 1:
            return
        arr = sorted(indices, key=lambda i: pos[i])
        for k in range(1, n):
            i_prev, i_curr = arr[k - 1], arr[k]
            d = abs(pos[i_curr] - pos[i_prev])
            # Update both neighbors
            if has_same_neighbor[i_curr] == 0 or d < gap_same_min[i_curr]:
                gap_same_min[i_curr] = d
                has_same_neighbor[i_curr] = 1.0
            if has_same_neighbor[i_prev] == 0 or d < gap_same_min[i_prev]:
                gap_same_min[i_prev] = d
                has_same_neighbor[i_prev] = 1.0

    fill_same_lane_min(idx_x)
    fill_same_lane_min(idx_y)

    # cross-lane min of |pos_i| + |pos_j|
    def fill_cross_min(curr_indices: List[int], other_indices: List[int]):
        if not other_indices:
            return
        min_other_abs = float(np.min(abs_pos[other_indices]))
        for i in curr_indices:
            cross_sum_min[i] = abs_pos[i] + min_other_abs
            has_cross_neighbor[i] = 1.0

    fill_cross_min(idx_x, idx_y)
    fill_cross_min(idx_y, idx_x)

    return gap_same_min, has_same_neighbor, cross_sum_min, has_cross_neighbor


# --------------- Build dataset for a single Λ ---------------
def build_for_one_dataset(root_dir: str, arrival_rate: float, max_agents: int):
    """
    Build:
      - X       : [num_samples, max_agents*FEAT_PER_AGENT + GLOBAL_DIM]
      - mask    : [num_samples, max_agents] (1=real agent in X at time t, 0=padding)
      - ids     : [num_samples, max_agents] (agent global IDs, -1 for padding)
      - Y       : [num_samples, max_agents] (accel along lane for agents that persist to t+1, else 0)
      - y_mask  : [num_samples, max_agents] (1=agent exists at t and t+1; 0=otherwise)
      - meta    : list of dicts with file, radius, t_index, num_agents, etc.
    """
    assert DT > 0, "DT must be > 0"

    X_list, M_list, ID_list, Y_list, YM_list, meta = [], [], [], [], [], []

    # Diagnostics accumulators
    cov_total_slots = 0
    cov_real_slots  = 0
    cov_valid_slots = 0
    rows_skipped_no_targets = 0
    last_frame_rows          = 0
    slot_overflow_count      = 0
    flow_flip_count          = 0
    per_file_cov_real = defaultdict(lambda: [0.0, 0.0])  # [valid_real, real]
    per_R_cov_real    = defaultdict(lambda: [0.0, 0.0])
    per_L_cov_real    = defaultdict(lambda: [0.0, 0.0])
    accel_nan_inf     = 0
    feat_nan_inf      = 0

    for folder_name, fp in find_all_json_files(root_dir):
        R_val = parse_radius_from_folder(folder_name)
        data = load_json(fp)
        if not isinstance(data, list):
            print(f"[warn] {fp} top-level JSON not a list; skipping")
            continue

        T = len(data)
        T_use = min(T, T_MAX_PER_FILE) if T_MAX_PER_FILE is not None else T
        if T_use <= 1:
            continue  # cannot form targets

        # Maintain stable slot assignment for this file
        slot_map: Dict[int, int] = {}       # vid -> slot index
        free_slots = list(range(max_agents))

        for t_idx in range(T_use):
            if t_idx == T_use - 1:
                # last usable row has no t+1; count and skip
                last_frame_rows += 1
                continue

            t_list   = data[t_idx]
            t1_list  = data[t_idx + 1]

            if not t_list:
                continue

            # Inside-zone agents by structure
            inside_t   = [rec for rec in t_list  if is_inside_by_structure(rec)]
            inside_t1  = [rec for rec in t1_list if is_inside_by_structure(rec)]

            if not inside_t:
                continue  # nothing to learn at this step

            # Normalize current inside agents
            agents_now = []
            vids_now = set()
            for rec in inside_t:
                x, y, vx, vy, flow, vid = normalize_record(rec)
                # sanity: flow is 0/1
                if flow not in (0, 1):
                    print(f"[warn] flow not in {{0,1}} for vid={vid} at {fp}@t={t_idx}; got {flow}")
                    flow = 0 if flow <= 0 else 1
                agents_now.append({'vid': vid, 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'flow': flow})
                vids_now.add(vid)

            # Build a lookup for t+1 (by vid)
            next_map: Dict[int, Dict] = {}
            for rec in inside_t1:
                x, y, vx, vy, flow, vid = normalize_record(rec)
                if flow not in (0, 1):
                    flow = 0 if flow <= 0 else 1
                next_map[vid] = {'vid': vid, 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'flow': flow}

            # Release slots for agents that left the zone at this step
            missing_vids = [vid for vid in list(slot_map.keys()) if vid not in vids_now]
            if missing_vids:
                import bisect
                for vid in missing_vids:
                    freed = slot_map.pop(vid)
                    bisect.insort(free_slots, freed)

            # Prepare arrays for this timestep
            feats = np.zeros((max_agents, FEAT_PER_AGENT), dtype=np.float32)
            ids   = np.full((max_agents,), -1, dtype=np.int64)
            mask  = np.zeros((max_agents,), dtype=np.float32)
            flows = np.full((max_agents,), -1, dtype=np.int64)  # 0 or 1 for real slots
            yrow  = np.zeros((max_agents,), dtype=np.float32)
            ymask = np.zeros((max_agents,), dtype=np.float32)

            # Fill base 4 cols and record flow per slot
            for a in agents_now:
                vid = a['vid']
                # keep existing slot or assign new
                if vid in slot_map:
                    s = slot_map[vid]
                else:
                    if not free_slots:
                        slot_overflow_count += 1
                        # overflow: drop this agent (cannot assign slot)
                        continue
                    s = free_slots.pop(0)
                    slot_map[vid] = s

                pos_along, vel_along = along_lane_pos_vel(a['x'], a['y'], a['vx'], a['vy'], a['flow'])
                dist_c = math.hypot(a['x'], a['y'])

                # cols: [pos, vel, flow, dist_c, (neighbor/safety features will be filled next)]
                feats[s, 0:4] = [pos_along, vel_along, float(a['flow']), dist_c]
                flows[s] = a['flow']
                ids[s] = vid
                mask[s] = 1.0

            # Compute lane-structural features with validity flags + safety margins/flags
            if mask.sum() > 0:
                gap_same_min, has_same, cross_min, has_cross = \
                    compute_same_lane_min_gap_and_cross_min(feats, mask, flows, max_agents)

                # Base neighbor features
                feats[:, 4] = gap_same_min
                feats[:, 5] = has_same
                feats[:, 6] = cross_min
                feats[:, 7] = has_cross

                # --- Cross safety: margin & flag ---
                cross_margin   = np.zeros((max_agents,), dtype=np.float32)
                cross_is_safe  = np.zeros((max_agents,), dtype=np.float32)
                valid_cross = has_cross > 0.5
                cross_margin[valid_cross]  = cross_min[valid_cross] - S_DIST_CROSS
                cross_is_safe[valid_cross] = (cross_margin[valid_cross] >= 0.0).astype(np.float32)
                feats[:, 8] = cross_margin
                feats[:, 9] = cross_is_safe

                # --- Same-lane safety: margin & flag ---
                gap_margin  = np.zeros((max_agents,), dtype=np.float32)
                gap_is_safe = np.zeros((max_agents,), dtype=np.float32)
                valid_same = has_same > 0.5
                gap_margin[valid_same]  = gap_same_min[valid_same] - D_SAFE_SAME
                gap_is_safe[valid_same] = (gap_margin[valid_same] >= 0.0).astype(np.float32)
                feats[:, 10] = gap_margin
                feats[:, 11] = gap_is_safe

            # Targets: accel if the agent persists to t+1
            real_slots = int(mask.sum())
            cov_total_slots += max_agents
            cov_real_slots  += real_slots

            for s in range(max_agents):
                if mask[s] < 0.5:
                    continue
                vid = int(ids[s])
                if vid in next_map:
                    # flow consistency check
                    flow_now = int(feats[s, 2])
                    flow_nxt = int(next_map[vid]['flow'])
                    if flow_now != flow_nxt:
                        flow_flip_count += 1  # lane assignment changed across steps

                    a_now_vel = feats[s, 1]
                    b = next_map[vid]
                    _, vel_next = along_lane_pos_vel(b['x'], b['y'], b['vx'], b['vy'], b['flow'])
                    acc = (vel_next - a_now_vel) / DT
                    if not np.isfinite(acc):
                        accel_nan_inf += 1
                        acc = 0.0
                    yrow[s]  = acc
                    ymask[s] = 1.0

            # Skip rows where no valid targets exist
            if ymask.sum() == 0:
                rows_skipped_no_targets += 1
                continue

            cov_valid_slots += int(ymask.sum())

            # Flatten and append global features [R, ARRIVAL_RATE]
            x_vec = feats.flatten()
            # feature NaN/Inf audit
            if not np.isfinite(x_vec).all():
                feat_nan_inf += int(np.size(x_vec) - np.isfinite(x_vec).sum())
                x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=0.0, neginf=0.0)

            if INCLUDE_GLOBAL_FEATURES:
                x_vec = np.hstack([x_vec, np.array([R_val, arrival_rate], dtype=np.float32)])

            # Store
            X_list.append(x_vec)
            M_list.append(mask)
            ID_list.append(ids)
            Y_list.append(yrow)
            YM_list.append(ymask)

            # Meta & coverage buckets
            meta.append({
                "file": fp,
                "radius_folder": folder_name,
                "R": R_val,
                "arrival_rate": arrival_rate,
                "t_index": t_idx,
                "num_agents": int(mask.sum()),
                "num_targets": int(ymask.sum()),
                "d_safe_same": D_SAFE_SAME,
                "s_dist_cross": S_DIST_CROSS
            })
            per_file_cov_real[fp][0] += float(ymask.sum())
            per_file_cov_real[fp][1] += float(mask.sum())
            per_R_cov_real[R_val][0] += float(ymask.sum())
            per_R_cov_real[R_val][1] += float(mask.sum())
            per_L_cov_real[arrival_rate][0] += float(ymask.sum())
            per_L_cov_real[arrival_rate][1] += float(mask.sum())

    # Stack to arrays
    if X_list:
        X     = np.vstack(X_list).astype(np.float32)
        mask  = np.vstack(M_list).astype(np.float32)
        IDs   = np.vstack(ID_list).astype(np.int64)
        Y     = np.vstack(Y_list).astype(np.float32)
        ymask = np.vstack(YM_list).astype(np.float32)
    else:
        X     = np.zeros((0, max_agents*FEAT_PER_AGENT + GLOBAL_DIM), dtype=np.float32)
        mask  = np.zeros((0, max_agents), dtype=np.float32)
        IDs   = np.zeros((0, max_agents), dtype=np.int64)
        Y     = np.zeros((0, max_agents), dtype=np.float32)
        ymask = np.zeros((0, max_agents), dtype=np.float32)

    # ---------------- Sanity checks & coverage prints ----------------
    total_slots = X.shape[0] * max_agents
    overall_cov = (ymask.sum() / total_slots) if total_slots > 0 else 0.0
    real_cov    = (ymask.sum() / np.clip(mask.sum(), 1, None)) if mask.size > 0 else 0.0

    print("[coverage] overall valid targets ratio = {:.4f}".format(overall_cov))
    print("[coverage] valid / real-agent slots    = {:.4f}  (valid={} / real={})"
          .format(real_cov, int(ymask.sum()), int(mask.sum())))
    print("[coverage] builder counters: total_slots={} real_slots={} valid_slots={}"
          .format(cov_total_slots, cov_real_slots, cov_valid_slots))
    print("[info] rows skipped (no valid targets) =", rows_skipped_no_targets)
    print("[info] last-frame rows (no t+1)        =", last_frame_rows)
    print("[warn] slot overflows (agent dropped)   =", slot_overflow_count)
    print("[warn] flow flips between t and t+1     =", flow_flip_count)
    print("[audit] feature NaN/Inf replaced        =", feat_nan_inf)
    print("[audit] target  NaN/Inf replaced        =", accel_nan_inf)

    # Consistency: y_mask must be zero where mask is zero
    if mask.size > 0:
        bad = np.logical_and(mask < 0.5, ymask > 0.5).sum()
        if bad > 0:
            print(f"[ERROR] y_mask has {int(bad)} entries where mask==0; check builder logic!")

    # Per-file coverage (first few)
    print("\n[coverage/by-file] (first 10)")
    for i, (f, (v, r)) in enumerate(per_file_cov_real.items()):
        if i >= 10: break
        ratio = (v / r) if r > 0 else 0.0
        short = os.path.basename(f)
        print(f"  {short:40s} real={int(r):8d} valid={int(v):8d} ratio={ratio:.3f}")

    # Per-Λ and per-R coverage
    if per_L_cov_real:
        print("\n[coverage/by-Λ]")
        for lam in sorted(per_L_cov_real.keys()):
            v, r = per_L_cov_real[lam]
            print(f"  Λ={lam:.3f}: real={int(r):8d} valid={int(v):8d} ratio={ (v/r) if r>0 else 0.0 :.3f}")
    if per_R_cov_real:
        print("\n[coverage/by-R]")
        for Rv in sorted(per_R_cov_real.keys()):
            v, r = per_R_cov_real[Rv]
            print(f"  R={Rv:5.1f}: real={int(r):8d} valid={int(v):8d} ratio={ (v/r) if r>0 else 0.0 :.3f}")

    return X, mask, IDs, Y, ymask, meta


# ------------------ Build & Save ------------------
def build_and_save_all():
    max_agents, nfiles, hist = first_pass_max_agents_across_datasets()
    if max_agents == 0:
        raise SystemExit("No inside-zone agents found. Check your files / structure rule.")
    print(f"[info] scanned {nfiles} file(s) across all datasets; max inside agents per snapshot = {max_agents}\n")

    # Padding diagnostics
    p95 = percentile_from_hist(hist, 0.95)
    mean_occ = sum(k*v for k, v in hist.items()) / max(1, sum(hist.values()))
    print("[padding] concurrent-agents histogram (top 10 bins):")
    for k, c in sorted(hist.items())[:10]:
        print(f"  n={k:2d}: count={c}")
    print(f"[padding] mean concurrent agents ≈ {mean_occ:.2f}, 95th percentile ≈ {p95}")
    if p95 < max_agents:
        print(f"[suggest] You could set max_agents={p95} to reduce padding, if you can accept 5% overflow.")

    combined = {"X": [], "mask": [], "ids": [], "Y": [], "y_mask": [], "meta": []}
    feat_dim_total = max_agents * FEAT_PER_AGENT + GLOBAL_DIM

    for ds_label, root_dir in DATASETS.items():
        lam = parse_lambda_from_label(ds_label)
        print(f"\n[info] Building dataset for {ds_label} (Λ={lam}) at root={root_dir}")
        X, mask, IDs, Y, ymask, meta = build_for_one_dataset(root_dir, lam, max_agents)

        out_dir = os.path.join(root_dir, "processed")
        os.makedirs(out_dir, exist_ok=True)
        out_npz = os.path.join(out_dir, "inputs_targets_dataset_XY_new.npz")

        np.savez_compressed(
            out_npz,
            X=X, mask=mask, ids=IDs, Y=Y, y_mask=ymask,
            meta=np.array(meta, dtype=object),
            feat_dim=np.array([feat_dim_total], dtype=np.int32),
            feat_per_agent=np.array([FEAT_PER_AGENT], dtype=np.int32),
            max_agents=np.array([max_agents], dtype=np.int32),
            include_global=np.array([int(INCLUDE_GLOBAL_FEATURES)], dtype=np.int32),
            global_dim=np.array([GLOBAL_DIM], dtype=np.int32),
            dt=np.array([DT], dtype=np.float32),
            d_safe_same=np.array([D_SAFE_SAME], dtype=np.float32),
            s_dist_cross=np.array([S_DIST_CROSS], dtype=np.float32)
        )

        print(f"[done] Saved per-Λ file: {out_npz}")
        print(f"        X shape    : {X.shape}  (features per sample = {feat_dim_total})")
        print(f"        Y shape    : {Y.shape}  (per-agent accel targets)")
        print(f"        mask shape : {mask.shape}")
        print(f"        y_mask shape: {ymask.shape}")
        print(f"        ids shape  : {IDs.shape}")

        if X.shape[0] > 0:
            gdim = GLOBAL_DIM
            row0 = X[0]
            g = row0[-gdim:] if (INCLUDE_GLOBAL_FEATURES and gdim > 0) else None
            core = row0[:-gdim] if (INCLUDE_GLOBAL_FEATURES and gdim > 0) else row0
            agents0 = core.reshape(max_agents, FEAT_PER_AGENT)
            k0 = int(mask[0].sum())
            kt0 = int(ymask[0].sum())
            print("[peek] first sample:")
            print("       num_agents (X) =", k0, " | num_targets (Y) =", kt0)
            for j in range(min(k0, 5)):
                (pos, vel, flow, dist,
                 gap_same_min, has_same,
                 cross_min, has_cross,
                 cross_margin, cross_is_safe,
                 gap_margin, gap_is_safe) = agents0[j]
                y_val   = Y[0, j]
                y_valid = int(ymask[0, j])
                print(f"       slot[{j:02d}] id={IDs[0,j]} pos={pos:.3f} vel={vel:.3f} flow={int(flow)} "
                      f"dist={dist:.3f} gap_min={gap_same_min:.3f} has_same={int(has_same)} "
                      f"cross_min={cross_min:.3f} has_cross={int(has_cross)} "
                      f"cross_margin={cross_margin:.3f} cross_is_safe={int(cross_is_safe)} "
                      f"gap_margin={gap_margin:.3f} gap_is_safe={int(gap_is_safe)} "
                      f"| a_t={y_val:.3f} (valid={y_valid})")
            if g is not None:
                print(f"       global: R={g[0]:.3f}, arrival_rate={g[1]:.3f}")
        print()

        combined["X"].append(X); combined["mask"].append(mask); combined["ids"].append(IDs)
        combined["Y"].append(Y); combined["y_mask"].append(ymask); combined["meta"].extend(meta)

    # Save combined under the first dataset's root /processed
    first_root = next(iter(DATASETS.values()))
    comb_out_dir = os.path.join(first_root, "processed")
    os.makedirs(comb_out_dir, exist_ok=True)
    comb_npz = os.path.join(comb_out_dir, "inputs_targets_dataset_XY_ALL.npz")

    if len(combined["X"]) > 0:
        X_all     = np.vstack(combined["X"]).astype(np.float32)
        mask_all  = np.vstack(combined["mask"]).astype(np.float32)
        ids_all   = np.vstack(combined["ids"]).astype(np.int64)
        Y_all     = np.vstack(combined["Y"]).astype(np.float32)
        ymask_all = np.vstack(combined["y_mask"]).astype(np.float32)
    else:
        X_all     = np.zeros((0, max_agents*FEAT_PER_AGENT + GLOBAL_DIM), dtype=np.float32)
        mask_all  = np.zeros((0, max_agents), dtype=np.float32)
        ids_all   = np.zeros((0, max_agents), dtype=np.int64)
        Y_all     = np.zeros((0, max_agents), dtype=np.float32)
        ymask_all = np.zeros((0, max_agents), dtype=np.float32)

    np.savez_compressed(
        comb_npz,
        X=X_all, mask=mask_all, ids=ids_all, Y=Y_all, y_mask=ymask_all,
        meta=np.array(combined["meta"], dtype=object),
        feat_dim=np.array([max_agents * FEAT_PER_AGENT + GLOBAL_DIM], dtype=np.int32),
        feat_per_agent=np.array([FEAT_PER_AGENT], dtype=np.int32),
        max_agents=np.array([max_agents], dtype=np.int32),
        include_global=np.array([int(INCLUDE_GLOBAL_FEATURES)], dtype=np.int32),
        global_dim=np.array([GLOBAL_DIM], dtype=np.int32),
        dt=np.array([DT], dtype=np.float32),
        d_safe_same=np.array([D_SAFE_SAME], dtype=np.float32),
        s_dist_cross=np.array([S_DIST_CROSS], dtype=np.float32)
    )

    print(f"[done] Saved COMBINED dataset: {comb_npz}")
    print(f"       X shape    : {X_all.shape}  (features per sample = {max_agents * FEAT_PER_AGENT + GLOBAL_DIM})")
    print(f"       Y shape    : {Y_all.shape}")
    print(f"       mask shape : {mask_all.shape}")
    print(f"       y_mask     : {ymask_all.shape}")
    print(f"       ids shape  : {ids_all.shape}")

    # Final combined coverage numbers for quick glance
    total_slots_all = X_all.shape[0] * max_agents
    overall_cov_all = (ymask_all.sum() / total_slots_all) if total_slots_all > 0 else 0.0
    real_cov_all    = (ymask_all.sum() / np.clip(mask_all.sum(), 1, None)) if mask_all.size > 0 else 0.0
    print("[coverage/COMBINED] overall={:.4f}  real={:.4f}  (valid={} / real={} / total={})"
          .format(overall_cov_all, real_cov_all, int(ymask_all.sum()), int(mask_all.sum()), int(total_slots_all)))

    return comb_npz, max_agents


if __name__ == "__main__":
    # Build data & save combined NPZ (per-Λ files also saved)
    build_and_save_all()
