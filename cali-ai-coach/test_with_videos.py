import sys, os, json, cv2, torch, numpy as np
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from core.coco_to_h36m import coco_to_h36m
from core.keyframes import extract_key_frames

DATA = Path(__file__).parent / 'data'
STD_VIDEO = str(DATA / 'push_up_true.mp4')
STU_VIDEO = str(DATA / 'push_up_wrong.mp4')

MODEL_NAME = 'yolo26x-pose.pt'
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

print(f'Loading YOLO...')
model = YOLO(MODEL_NAME)

def process_video(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()
    frames = []
    for result in model.predict(source=path, stream=True, imgsz=640, conf=0.25, device=DEVICE, verbose=False):
        kps = None
        if result.keypoints is not None and result.keypoints.data is not None:
            data = result.keypoints.data.detach().cpu().numpy().astype(np.float32)
            if data.size > 0:
                if data.shape[-1] == 2:
                    conf = np.ones(data.shape[:2] + (1,), np.float32)
                    data = np.concatenate([data, conf], axis=-1)
                scores = np.nanmean(data[:, :, 2], axis=1)
                if np.isfinite(scores).any():
                    kps = data[int(np.nanargmax(scores))]
        frames.append(coco_to_h36m(kps) if kps is not None else np.zeros((17, 3), np.float32))
    return np.stack(frames).astype(np.float32) if frames else np.empty((0, 17, 3), np.float32)

def normalize_pose(pose):
    """Pelvis center + spine length scale (giống adapter.js normalizePose)"""
    p = pose.copy()
    p[:, :2] -= pose[0:1, :2]  # trừ pelvis
    spine = p[8, :2]            # Thorax sau khi trừ pelvis
    sl = np.linalg.norm(spine) or 1
    p[:, :2] /= sl
    return p

def cos_sim(a, b):
    dot = (a * b).sum()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return dot / (na * nb + 1e-12)

BONES = [
    (14, 15, 'Tay trên phải', 0.995), (15, 16, 'Tay dưới phải', 0.995),
    (11, 12, 'Tay trên trái', 0.995), (12, 13, 'Tay dưới trái', 0.995),
    (1, 2,  'Chân trên phải', 0.990), (2, 3,  'Chân dưới phải', 0.990),
    (4, 5,  'Chân trên trái', 0.990), (5, 6,  'Chân dưới trái', 0.990),
    (0, 7,  'Bụng', 0.985), (7, 8, 'Ngực', 0.985),
    (8, 9,  'Cổ dưới', 0.980), (9, 10, 'Cổ trên', 0.980),
]

def compare_pose(student, standard, bones=None):
    if bones is None: bones = BONES
    errors = []
    for parent, child, name, threshold in bones:
        sv = student[child, :2] - student[parent, :2]
        mv = standard[child, :2] - standard[parent, :2]
        sim = cos_sim(sv, mv)
        if sim < threshold:
            errors.append({'bone': name, 'sim': float(sim)})
    return errors

def find_best_match(student, cp_list, bones=None):
    if bones is None: bones = BONES
    best, best_err = 0, float('inf')
    for i, cp in enumerate(cp_list):
        err = compare_pose(student, cp[:, :2], bones)
        if len(err) < best_err:
            best_err, best = len(err), i
    return best

print('Processing standard video...')
std_arr = process_video(STD_VIDEO)
print(f'  shape={std_arr.shape}')

print('Extracting keyframes from standard...')
kf_indices = extract_key_frames(std_arr, n_clusters=30)
checkpoints = [std_arr[i] for i in kf_indices]

# Pre-normalize checkpoints
norm_checkpoints = [normalize_pose(cp)[:, :2] for cp in checkpoints]

print('Processing student video...')
stu_arr = process_video(STU_VIDEO)
print(f'  shape={stu_arr.shape}')

# Rolling average filter
window = []
def smooth(frame):
    window.append(frame.copy())
    if len(window) > 5:
        window.pop(0)
    if len(window) < 2:
        return frame.copy()
    return np.mean(window, axis=0)

# Live evaluator
cur_cp = -1
total_reps = 0
initialized = False

print(f'\n--- Live Evaluation ({len(stu_arr)} frames) ---')
for f_idx in range(len(stu_arr)):
    raw = stu_arr[f_idx]
    norm = normalize_pose(raw)[:, :2]
    smoothed = smooth(norm)

    if not initialized:
        best = find_best_match(smoothed, norm_checkpoints)
        cur_cp = best
        initialized = True
        print(f'[Frame {f_idx}] INIT -> checkpoint {best}/30')
        continue

    if cur_cp >= len(norm_checkpoints):
        total_reps += 1
        print(f'[Frame {f_idx}] REP COMPLETE (total={total_reps})')
        cur_cp = 0

    errs = compare_pose(smoothed, norm_checkpoints[cur_cp])

    if len(errs) == 0:
        cur_cp += 1
        print(f'[Frame {f_idx}] PASS -> checkpoint {cur_cp}/30')
    else:
        print(f'[Frame {f_idx}] WAIT (checkpoint {cur_cp}/30, {len(errs)} errors)')

print(f'\n--- DONE: {total_reps} reps completed ---')
