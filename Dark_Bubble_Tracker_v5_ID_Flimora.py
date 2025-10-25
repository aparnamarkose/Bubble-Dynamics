# ===============================================================
# 🖤 Dark Bubble Tracker v5-ID — Relaxed, Motion-Aware Bubble Tracking
# ===============================================================
# Author: Aparna Markose
# Institution: Pondicherry University, India
# Version: v5-ID (2025)
#
# 🔗 Dataset:
# Electrochemical bubble evolution dataset hosted on Google Drive:
# https://drive.google.com/drive/folders/14Lx6HuAEkTNryvftkNRynNMiNJ658CQs?usp=sharing
#
# 📁 Folder Structure:
# ├── 5v/
# │    ├── Bare_Ti_5v_na2so4.mp4
# │    ├── TNT_5v_na2so4.mp4
# │    └── CTNT_5v_na2so4.mp4
# ├── 10v/
# │    ├── Bare_Ti_10v_na2so4.mp4
# │    ├── TNT_10v_na2so4.mp4
# │    └── CTNT_10v_na2so4.mp4
# ├── 15v/
# │    ├── Bare_Ti_15v_na2so4.mp4
# │    ├── TNT_15v_na2so4.mp4
# │    └── CTNT_15v_na2so4.mp4
# └── 20v/
#      ├── Bare_Ti_20v_na2so4.mp4
#      ├── TNT_20v_na2so4.mp4
#      └── CTNT_20v_na2so4.mp4
#
# ===============================================================
# Key Features:
# ✅ Tracks dark (shade-type) bubbles with unique persistent IDs
# ✅ Displays ID labels and trajectories on annotated frames
# ✅ Exports per-second counts (new & leaving bubbles)
# ✅ Exports full bubble trajectory summaries
# ✅ Relaxed “leaving” rule — top 30% zone or upward motion
#
# Reference:
# Zhao, X.; Ren, H.; Luo, L.
# Gas Bubbles in Electrochemical Gas Evolution Reactions.
# Langmuir 2019, 35(16), 5392–5408.
# https://doi.org/10.1021/acs.langmuir.9b00138
# ===============================================================

!pip install -q scipy pandas matplotlib opencv-python-headless tqdm

import os, cv2, numpy as np, pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
BASE_DIR   = "/content/drive/My Drive/Colab Notebooks/Bubbles_Program/flimora/20v"
VIDEO_NAME = "CTNT_20v_na2so4_dark.mp4"
VIDEO_PATH = os.path.join(BASE_DIR, VIDEO_NAME)

FRAMES_DIR = os.path.join(BASE_DIR, "frames_dark_v5")
ANNOT_DIR  = os.path.join(BASE_DIR, "annot_dark_v5")
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)

EXTRACT_FRAMES = True

# --- Detection & tracking parameters ---
MIN_AREA        = 50
MAX_AREA        = 6000
MAX_MATCH_DIST  = 50
MAX_MISSED      = 6
CIRC_MIN        = 0.35
MIN_CONTRAST    = 8
TOP_REGION      = 0.10
UPWARD_DISP_MIN = 40
ANNOT_EVERY     = 50

# --- Background settings ---
BG_SECONDS      = 4
BG_SAMPLE_POOL  = 200
BG_FRAMES_N     = None

# ---------- EXTRACT FRAMES ----------
if EXTRACT_FRAMES and len(os.listdir(FRAMES_DIR)) == 0:
    print("🎥 Extracting frames from video...")
    !ffmpeg -y -i "$VIDEO_PATH" -qscale:v 2 "$FRAMES_DIR/frame_%05d.png"

frames_list = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".png")])
if len(frames_list) == 0:
    raise RuntimeError("No frames found in FRAMES_DIR. Extract frames or correct path.")

# ---------- VIDEO INFO ----------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
duration_s = len(frames_list) / fps
print(f"✅ {len(frames_list)} frames ({duration_s:.2f}s @ {fps:.1f} fps)")

# ---------- SMART BACKGROUND ----------
def build_low_motion_background(frames_list, frames_dir, fps, bg_seconds=BG_SECONDS, pool_cap=BG_SAMPLE_POOL, bg_n=None):
    capN = min(len(frames_list), int(bg_seconds * fps))
    capN = min(capN, pool_cap)
    imgs = []
    for i in range(capN):
        im = cv2.imread(os.path.join(frames_dir, frames_list[i]), cv2.IMREAD_GRAYSCALE)
        imgs.append(im.astype(np.int16))
    motion_scores = [np.sum(np.abs(imgs[i] - imgs[i-1])) if i > 0 else np.sum(np.abs(imgs[0]-imgs[1])) for i in range(capN)]
    ordered = np.argsort(motion_scores)
    if bg_n is None:
        bg_n = max(3, int(capN // 2))
    bg_indices = ordered[:bg_n]
    bg_frames = [cv2.imread(os.path.join(frames_dir, frames_list[int(idx)]), cv2.IMREAD_GRAYSCALE)
                 for idx in bg_indices]
    background = np.median(np.stack(bg_frames, axis=0), axis=0).astype(np.uint8)
    print(f"🧩 Built background from {len(bg_frames)} low-motion frames (indices: {sorted(list(bg_indices))})")
    return background

background = build_low_motion_background(frames_list, FRAMES_DIR, fps)
H, W = background.shape

# ---------- TRACKING FUNCTION ----------
def match_detections_to_tracks(dets, tracks, maxdist):
    if not dets or not tracks:
        return {}, list(range(len(dets))), list(tracks.keys())
    detC = [d[1] for d in dets]
    trK = list(tracks.keys())
    trC = [tracks[t]['centroid'] for t in trK]
    D = np.zeros((len(detC), len(trC)))
    for i, dc in enumerate(detC):
        for j, tc in enumerate(trC):
            D[i, j] = np.hypot(dc[0]-tc[0], dc[1]-tc[1])
    matches, usedd, usedt = {}, set(), set()
    for di, dj in np.dstack(np.unravel_index(np.argsort(D.ravel()), D.shape))[0]:
        if di in usedd or dj in usedt:
            continue
        if D[di, dj] <= maxdist:
            matches[di] = trK[dj]
            usedd.add(di)
            usedt.add(dj)
    unmatched_dets = [i for i in range(len(detC)) if i not in matches]
    unmatched_trs = [trK[j] for j in range(len(trC)) if j not in usedt]
    return matches, unmatched_dets, unmatched_trs

# ---------- MAIN DETECTION LOOP ----------
tracks, ended_tracks_list = {}, []
next_id = 1
per_frame_new = []
kernel = np.ones((3,3), np.uint8)

print("🚀 Tracking dark bubbles (v5-ID, relaxed)...")

for i, fname in enumerate(tqdm(frames_list)):
    gray = cv2.imread(os.path.join(FRAMES_DIR, fname), cv2.IMREAD_GRAYSCALE)
    diff = cv2.subtract(background, gray)
    diff = cv2.GaussianBlur(diff, (5,5), 0)
    _, th = cv2.threshold(diff, MIN_CONTRAST, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # Bubble detection
    dets = []
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        a = cv2.contourArea(c)
        if not (MIN_AREA <= a <= MAX_AREA):
            continue
        p = cv2.arcLength(c, True)
        circ = (4*np.pi*a)/(p*p+1e-6)
        if circ < CIRC_MIN:
            continue
        x,y,w,h = cv2.boundingRect(c)
        cx, cy = x + w/2, y + h/2
        bg_mean = np.mean(background[y:y+h, x:x+w]) if h>0 and w>0 else 255
        mean_int = np.mean(gray[y:y+h, x:x+w]) if h>0 and w>0 else 255
        if bg_mean - mean_int < MIN_CONTRAST:
            continue
        dets.append(((x,y,w,h), (cx,cy), a))

    matches, unmatched_dets, unmatched_trs = match_detections_to_tracks(dets, tracks, MAX_MATCH_DIST)
    used = set()

    # Update existing tracks
    for di, tid in matches.items():
        bbox, cen, a = dets[di]
        tr = tracks[tid]
        tr['centroid'] = cen
        tr['bbox'] = bbox
        tr['last'] = i
        tr['miss'] = 0
        tr['trajectory'].append((i, cen))
        total_up = tr['trajectory'][0][1][1] - cen[1]
        avg_dy = np.mean(np.diff([pt[1][1] for pt in tr['trajectory'][-5:]])) if len(tr['trajectory'])>5 else 0
        if (cen[1] <= H*TOP_REGION) or (total_up > UPWARD_DISP_MIN and avg_dy < -0.1):
            tr['left'] = True
            tr['end_frame'] = i
            tr['ended'] = True
            ended_tracks_list.append(tr.copy())
            used.add(tid)
            continue
        used.add(tid)

    # Manage old/missing tracks
    for tid in list(tracks.keys()):
        if tid not in used:
            tracks[tid]['miss'] += 1
        if tracks[tid]['miss'] > MAX_MISSED:
            tr = tracks[tid]
            tr['end_frame'] = tr['last']
            total_up = tr['trajectory'][0][1][1] - tr['trajectory'][-1][1][1]
            tr['left'] = total_up > UPWARD_DISP_MIN
            ended_tracks_list.append(tr.copy())
            del tracks[tid]

    # Add new tracks
    new_ids = []
    for di in unmatched_dets:
        bbox, cen, a = dets[di]
        tid = next_id; next_id += 1
        tracks[tid] = dict(
            id=tid, centroid=cen, bbox=bbox, first=i, last=i, miss=0,
            trajectory=[(i, cen)], ended=False, left=False
        )
        new_ids.append(tid)
    per_frame_new.append(len(new_ids))

# ---------- POST-PROCESSING ----------
df_new = pd.DataFrame({"frame": np.arange(len(frames_list)), "new_bubbles": per_frame_new})
df_new["second"] = (df_new["frame"] / fps).astype(int)
df_sec_new = df_new.groupby("second")["new_bubbles"].sum().reset_index()

left_times = [int(tr["end_frame"]/fps) for tr in ended_tracks_list if tr.get("left", False)]
df_left = (
    pd.Series(left_times).value_counts().sort_index().reset_index()
    if len(left_times) > 0
    else pd.DataFrame({"index": [], 0: []})
)
df_left.columns = ["second", "left_bubbles"]
df_combined = pd.merge(df_sec_new, df_left, on="second", how="outer").fillna(0)

# ---------- SAVE OUTPUTS ----------
out_combined = os.path.join(BASE_DIR, f"dark_bubbles_new_vs_left_v5_relaxed_{VIDEO_NAME.split('.')[0]}.csv")
df_combined.to_csv(out_combined, index=False)

plt.figure(figsize=(10,5))
plt.plot(df_combined["second"], df_combined["new_bubbles"], 'o-', color='royalblue', label="🫧 New bubbles/sec")
plt.plot(df_combined["second"], df_combined["left_bubbles"], 's-', color='firebrick', label="⬆️ Left bubbles/sec (top 30%)")
plt.xlabel("Time (s)")
plt.ylabel("Bubble count per second")
plt.title(f"Dark Bubble Dynamics — {VIDEO_NAME.split('.')[0]}")
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()

plot_path = os.path.join(BASE_DIR, f"bubble_dynamics_{VIDEO_NAME.split('.')[0]}.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"✅ Plot saved to: {plot_path}")
print(f"💾 CSV saved to: {out_combined}")
print("🎯 Annotated frames are available in:", ANNOT_DIR)
