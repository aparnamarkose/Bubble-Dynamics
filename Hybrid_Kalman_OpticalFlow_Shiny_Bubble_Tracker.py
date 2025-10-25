# ===============================================================
# 💡 Hybrid Kalman + Optical Flow Bubble Tracker (Bright / Shiny Bubbles)
# ===============================================================
# Author: Aparna Markose
# Institution: Pondicherry University, India
# Version: Hybrid-v5 (2025)
#
# 🔗 Dataset:
# Electrochemical bubble evolution dataset hosted on Google Drive:
# https://drive.google.com/drive/folders/14Lx6HuAEkTNryvftkNRynNMiNJ658CQs?usp=sharing
#
# 📁 Folder Structure (Flimora Dataset):
# ├── 5v/
# │    ├── Bare_Ti_5v_na2so4_shiny.mp4
# │    ├── TNT_5v_na2so4.mp4
# │    └── CTNT_5v_na2so4.mp4
# ├── 10v/
# │    ├── Bare_Ti_10v_na2so4_shiny.mp4
# │    ├── TNT_10v_na2so4.mp4
# │    └── CTNT_10v_na2so4.mp4
# ├── 15v/
# │    ├── Bare_Ti_15v_na2so4_shiny.mp4
# │    ├── TNT_15v_na2so4.mp4
# │    └── CTNT_15v_na2so4.mp4
# └── 20v/
#      ├── Bare_Ti_20v_na2so4_shiny.mp4
#      ├── TNT_20v_na2so4.mp4
#      └── CTNT_20v_na2so4.mp4
#
# ===============================================================
# Key Features:
# ✅ Tracks shiny, high-reflection gas bubbles using hybrid Kalman + optical flow
# ✅ Detects bubbles appearing and disappearing (new vs leaving)
# ✅ Exports trajectory files (.csv) for each bubble
# ✅ Generates combined per-second bubble dynamics plots
# ✅ Calibrated for Bare Ti (high brightness) surfaces
#
# Reference:
# Zhao, X.; Ren, H.; Luo, L.
# Gas Bubbles in Electrochemical Gas Evolution Reactions.
# Langmuir 2019, 35(16), 5392–5408.
# https://doi.org/10.1021/acs.langmuir.9b00138
# ===============================================================

import os, cv2, math, numpy as np, pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------
# PATHS & INITIAL SETUP
# -------------------------
POSSIBLE = [
    "/content/drive/My Drive/Colab Notebooks/Bubbles_Program/flimora/20v/BTi_20v_na2so4_shiny.mp4",
    "/content/drive/My Drive/Colab Notebooks/Bubbles_Program/flimora/15v/BTi_15v_na2so4_shiny.mp4",
    "/content/drive/My Drive/Colab Notebooks/Bubbles_Program/flimora/10v/BTi_10v_na2so4_shiny.mp4"
]
VIDEO_PATH = next((p for p in POSSIBLE if os.path.exists(p)), None)
if VIDEO_PATH is None:
    raise FileNotFoundError("No video found. Please check your Google Drive path or dataset structure.")

OUTPUT_DIR = os.path.join(os.path.dirname(VIDEO_PATH), "hybrid_tracking_output")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "annot_frames")
TRAJ_DIR = os.path.join(OUTPUT_DIR, "trajectories")
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)

# -------------------------
# DETECTION + TRACKING PARAMETERS
# -------------------------
TOPHAT_K = np.ones((11,11), np.uint8)
BRIGHTNESS_GAIN = 2.0
ADAPTIVE_BLOCK = 35
ADAPTIVE_C = 12
MIN_AREA = 80
MAX_AREA = 25000
MAX_MATCH_DIST = 80
MAX_MISSED_FRAMES = 15
LEAVE_Y_THRESHOLD = 80
FADE_INTENSITY_DROP = 0.6
MIN_UPWARD_FLOW_TO_COUNT = 0.5
FPS_DEFAULT = 60.0

# Kalman helper
def create_kalman(x, y, vx=0.0, vy=0.0):
    kf = cv2.KalmanFilter(6, 2)
    dt = 1.0
    kf.transitionMatrix = np.array([
        [1,0,dt,0,0.5*dt*dt,0],
        [0,1,0,dt,0,0.5*dt*dt],
        [0,0,1,0,dt,0],
        [0,0,0,1,0,dt],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]
    ], np.float32)
    kf.measurementMatrix = np.zeros((2,6), np.float32)
    kf.measurementMatrix[0,0] = 1.0
    kf.measurementMatrix[1,1] = 1.0
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.statePre = np.array([[x],[y],[vx],[vy],[0.0],[0.0]], np.float32)
    kf.statePost = kf.statePre.copy()
    return kf

def id2color(tid):
    return (int((tid*37)%256), int((tid*71)%256), int((tid*97)%256))

# -------------------------
# DETECTION FUNCTION
# -------------------------
def detect_candidates(gray):
    H, W = gray.shape
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, TOPHAT_K)
    enhanced = cv2.addWeighted(gray, 1.0, tophat, BRIGHTNESS_GAIN, 0)
    binary = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   ADAPTIVE_BLOCK, -ADAPTIVE_C)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < MIN_AREA or a > MAX_AREA:
            continue
        x,y,w,h = cv2.boundingRect(c)
        cx, cy = x + w/2, y + h/2
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean_int = cv2.mean(gray, mask=mask)[0]
        dets.append({"cx":cx, "cy":cy, "bbox":(x,y,w,h), "area":a, "mean_int":mean_int, "mask":mask})
    return dets

# Optical flow helper
def compute_flow(prev_gray, next_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray,
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def median_flow_in_mask(flow, mask):
    ys = np.where(mask>0)
    if len(ys[0]) == 0:
        return 0.0
    v = flow[ys][:,1]
    return float(np.median(v))

# -------------------------
# PROCESS VIDEO
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
print(f"Processing video: {VIDEO_PATH}")
print(f"Detected {total_frames} frames @ {fps:.1f} fps")

ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame from video.")
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

tracks = {}
next_id = 1
completed_tracks = {}
new_events = []
left_events = []
per_frame_new = [0]*total_frames
per_frame_left = [0]*total_frames

for frame_idx in tqdm(range(total_frames), desc="Processing full video"):
    if frame_idx == 0:
        frame = prev_frame.copy()
    else:
        ret, frame = cap.read()
        if not ret:
            break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dets = detect_candidates(gray)
    flow = compute_flow(prev_gray, gray)

    for d in dets:
        d['median_vflow'] = median_flow_in_mask(flow, d['mask'])

    preds = {}
    for tid, tr in tracks.items():
        pred = tr['kf'].predict()
        px, py = float(pred[0]), float(pred[1])
        preds[tid] = (px, py)

    matched_det_to_tid = {}
    if len(dets)>0 and len(preds)>0:
        track_ids = list(preds.keys())
        cost_matrix = np.full((len(dets), len(track_ids)), 1e6, dtype=np.float32)
        for di, d in enumerate(dets):
            for tj, tid in enumerate(track_ids):
                px, py = preds[tid]
                dx, dy = d['cx'], d['cy']
                dist = math.hypot(px-dx, py-dy)
                vflow = d.get('median_vflow', 0.0)
                flow_term = 0.0
                if vflow < -MIN_UPWARD_FLOW_TO_COUNT:
                    flow_term += -40.0
                elif vflow > 0.5:
                    flow_term += 50.0
                last_int = tracks[tid].get('last_mean_int', None)
                intensity_term = 0.0
                if last_int is not None:
                    intensity_term = abs(last_int - d['mean_int']) * 0.05
                cost = dist + flow_term + intensity_term
                cost_matrix[di, tj] = cost
        flat_order = np.argsort(cost_matrix.ravel())
        used_d, used_t = set(), set()
        for flat in flat_order:
            di = flat // cost_matrix.shape[1]
            tj = flat % cost_matrix.shape[1]
            if di in used_d or tj in used_t: continue
            if cost_matrix[di, tj] < MAX_MATCH_DIST:
                used_d.add(di); used_t.add(tj)
                matched_det_to_tid[di] = track_ids[tj]

    used_tids = set()
    used_dets = set()
    for di, tid in matched_det_to_tid.items():
        d = dets[di]
        dx, dy = d['cx'], d['cy']
        tracks[tid]['kf'].correct(np.array([[np.float32(dx)], [np.float32(dy)]]))
        tracks[tid]['pos'] = (dx, dy)
        tracks[tid]['path'].append((dx, dy))
        tracks[tid]['missed'] = 0
        tracks[tid]['last_mean_int'] = d['mean_int']
        tracks[tid]['last_vflow'] = d['median_vflow']
        tracks[tid]['last_frame'] = frame_idx
        used_tids.add(tid)
        used_dets.add(di)

    to_delete = []
    for tid, tr in list(tracks.items()):
        if tid not in used_tids:
            tracks[tid]['missed'] += 1
            if tracks[tid]['missed'] > MAX_MISSED_FRAMES:
                last_x, last_y = tr['pos']
                last_vflow = tr.get('last_vflow', 0.0)
                last_mean = tr.get('last_mean_int', None)
                faded = False
                if last_mean is not None and tr.get('init_mean_int', None) is not None:
                    if last_mean < (tr['init_mean_int'] * FADE_INTENSITY_DROP):
                        faded = True
                left_flag = False
                if (last_y <= LEAVE_Y_THRESHOLD and last_vflow < -MIN_UPWARD_FLOW_TO_COUNT) or (faded and last_vflow < -0.5):
                    left_events.append((tid, frame_idx))
                    per_frame_left[frame_idx if frame_idx < len(per_frame_left) else -1] += 1
                    left_flag = True
                tr['left_flag'] = left_flag
                completed_tracks[tid] = tr
                to_delete.append(tid)
    for tid in to_delete:
        del tracks[tid]

    new_count = 0
    for di, d in enumerate(dets):
        if di in used_dets: continue
        dx, dy = d['cx'], d['cy']
        kf = create_kalman(dx, dy, vx=0.0, vy=-6.0)
        tid = next_id; next_id += 1
        tracks[tid] = {'id':tid, 'kf':kf, 'pos':(dx,dy), 'path':[(dx,dy)], 'missed':0,
                       'first_frame':frame_idx, 'last_frame':frame_idx, 'last_mean_int':d['mean_int'],
                       'init_mean_int':d['mean_int'], 'mean_int_history':[d['mean_int']], 'last_vflow':d['median_vflow']}
        new_events.append((tid, frame_idx))
        new_count += 1
    per_frame_new[frame_idx] = new_count

    ann = frame.copy()
    for tid, tr in tracks.items():
        col = id2color(tid)
        path = tr['path']
        for k in range(1, len(path)):
            x1,y1 = map(int, path[k-1]); x2,y2 = map(int, path[k])
            cv2.line(ann, (x1,y1), (x2,y2), col, 1)
        x,y = map(int, tr['pos'])
        cv2.circle(ann, (x,y), 3, col, -1)
        cv2.putText(ann, f"id{tid}", (x+4,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    for d in dets:
        x,y,w,h = d['bbox']
        cv2.rectangle(ann, (x,y), (x+w, y+h), (0,255,0), 1)
    cv2.putText(ann, f"Frame {frame_idx} | new:{new_count} | active:{len(tracks)}",
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imwrite(os.path.join(FRAMES_DIR, f"annot_{frame_idx:05d}.png"), ann)
    prev_gray = gray.copy()

cap.release()

for tid, tr in list(tracks.items()):
    tr['left_flag'] = False
    completed_tracks[tid] = tr
tracks.clear()

traj_rows = []
for tid, tr in completed_tracks.items():
    path = tr['path']
    dfp = pd.DataFrame(path, columns=['x','y'])
    dfp['frame'] = np.linspace(tr.get('first_frame',0), tr.get('last_frame', tr.get('first_frame',0)), len(dfp)).astype(int)
    os.makedirs(TRAJ_DIR, exist_ok=True)
    dfp.to_csv(os.path.join(TRAJ_DIR, f"traj_{tid:04d}.csv"), index=False)
    traj_rows.append({'id':tid, 'first_frame':tr.get('first_frame',None),
                      'last_frame':tr.get('last_frame',None),
                      'left_flag':tr.get('left_flag', False),
                      'path_len': len(tr.get('path',[]))})

pd.DataFrame(traj_rows).to_csv(os.path.join(OUTPUT_DIR, "completed_tracks_summary.csv"), index=False)

frames_done = total_frames
df_frame = pd.DataFrame({
    'frame': np.arange(frames_done),
    'new_bubbles': per_frame_new[:frames_done],
    'left_bubbles': per_frame_left[:frames_done]
})
df_frame['second'] = (df_frame['frame'] / fps).astype(int)
df_persec = df_frame.groupby('second')[['new_bubbles','left_bubbles']].sum().reset_index()
df_frame.to_csv(os.path.join(OUTPUT_DIR, "bubble_new_per_frame.csv"), index=False)
df_persec.to_csv(os.path.join(OUTPUT_DIR, "bubble_new_left_per_second.csv"), index=False)

sample_path = os.path.join(FRAMES_DIR, f"annot_{0:05d}.png")
sample_img = cv2.imread(sample_path) if os.path.exists(sample_path) else None
if sample_img is not None:
    H,W = sample_img.shape[:2]
    canvas = np.zeros((H,W,3), dtype=np.uint8)
    for tid, tr in completed_tracks.items():
        path = tr['path']
        col = id2color(tid)
        for k in range(1, len(path)):
            x1,y1 = map(int,path[k-1]); x2,y2 = map(int,path[k])
            cv2.line(canvas, (x1,y1), (x2,y2), col, 1)
    cv2.imwrite(os.path.join(OUTPUT_DIR,"combined_trajectories.png"), canvas)

plt.figure(figsize=(10,5))
plt.plot(df_persec['second'], df_persec['new_bubbles'], marker='o', label='New bubbles/sec')
plt.plot(df_persec['second'], df_persec['left_bubbles'], marker='s', label='Left bubbles/sec')
plt.xlabel('Time (s)'); plt.ylabel('Count')
plt.title('Bubble new vs left per second (hybrid tracker)')
plt.legend(); plt.grid(True)
plt.tight_layout()
out_plot = os.path.join(OUTPUT_DIR, "bubbles_new_vs_left_combined.png")
plt.savefig(out_plot, dpi=300)
plt.show()

print("\nDone.")
print("Outputs saved to:", OUTPUT_DIR)
print(" - Annotated frames:", FRAMES_DIR)
print(" - Per-track CSVs:", TRAJ_DIR)
print(" - Per-second CSV:", os.path.join(OUTPUT_DIR, 'bubble_new_left_per_second.csv'))
