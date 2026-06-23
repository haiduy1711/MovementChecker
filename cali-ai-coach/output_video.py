import sys, os, cv2, torch, json, numpy as np
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from core.coco_to_h36m import coco_to_h36m
from core.keyframes import extract_key_frames

DATA = Path(__file__).parent / 'data'
STD_VIDEO = str(DATA / 'push_up_true.mp4')
STU_VIDEO = str(DATA / 'push_up_wrong.mp4')
OUT_VIDEO = str(DATA / 'evaluation_result.mp4')

model = YOLO('yolo26x-pose.pt')
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# H36M 17 skeleton connections for drawing (parent→child, matches BONE_DEFS)
H36M_SKELETON = [
    (0, 1), (0, 4),             # Root → hips
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine
    (14, 15), (15, 16),         # right arm
    (11, 12), (12, 13),         # left arm
    (1, 2), (2, 3),             # right leg
    (4, 5), (5, 6),             # left leg
    (11, 7), (14, 8),           # shoulders → spine
]

H36M_JOINT_NAMES = [
    'Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'LShou', 'LElb', 'LWri', 'RShou', 'RElb', 'RWri',
]

# H36M bone definitions for evaluation only (parent, child, name, threshold)
BONE_DEFS = [
    (14, 15, 'R arm up', 0.995), (15, 16, 'R arm low', 0.995),
    (11, 12, 'L arm up', 0.995), (12, 13, 'L arm low', 0.995),
    (1, 2,  'R leg up', 0.990), (2, 3,  'R leg low', 0.990),
    (4, 5,  'L leg up', 0.990), (5, 6,  'L leg low', 0.990),
    (0, 7,  'Bung', 0.985), (7, 8, 'Nguc', 0.985),
    (8, 9,  'Co duoi', 0.3), (9, 10, 'Co tren', 0.3),
]

BONE_NAMES_LIST = [bd[2] for bd in BONE_DEFS]

def normalize_pose(pose):
    p = pose.copy()
    p[:, :2] -= pose[0:1, :2]
    spine = p[8, :2]
    sl = np.linalg.norm(spine) or 1
    p[:, :2] /= sl
    return p

def cos_sim(a, b):
    dot = (a * b).sum()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return dot / (na * nb + 1e-12)

def detect_errors(student_xy, target_xy):
    errs = {}
    for p, c, name, thresh in BONE_DEFS:
        sv = student_xy[c] - student_xy[p]
        mv = target_xy[c] - target_xy[p]
        if cos_sim(sv, mv) < thresh:
            errs[name] = True
    return errs


def check_push_up_alignment(norm_xy):
    """Kiểm tra alignment Push-up (tọa độ normalize). Trả về list feedback dicts."""
    feedback = []
    # Tay Phải: shoulder(14) vs wrist(16)
    for sho, wri, elb, side, bone in [(14, 16, 15, 'Phải', 'R arm up'),
                                       (11, 13, 12, 'Trái', 'L arm up')]:
        s = norm_xy[sho]; w = norm_xy[wri]; e = norm_xy[elb]
        dx = abs(s[0] - w[0])
        if dx > 0.05:
            sev = 'error' if dx > 0.08 else 'warning'
            feedback.append({'bone': bone, 'message':
                f'Tay {side}: vai lệch ngang {dx*100:.0f}%. Đưa vai về phía trước!',
                'severity': sev, 'type': 'alignment'})
        if e is not None:
            flare = abs(e[0] - s[0]) / (abs(s[0] - w[0]) + 1e-6)
            if flare > 2.0:
                fb_bone = 'R arm low' if side == 'Phải' else 'L arm low'
                feedback.append({'bone': fb_bone, 'message':
                    f'Tay {side}: khuỷu tay bung rộng. Giữ khuỷu sát thân!',
                    'severity': 'warning', 'type': 'flare'})
    # Hông không trễ/cong
    hip_y = (norm_xy[4][1] + norm_xy[1][1]) / 2
    sho_y = (norm_xy[11][1] + norm_xy[14][1]) / 2
    ank_y = (norm_xy[6][1] + norm_xy[3][1]) / 2
    exp_y = (sho_y + ank_y) / 2
    sag = hip_y - exp_y
    if sag > 0.08:
        feedback.append({'bone': 'Bung', 'message':
            f'Hông trễ xuống {sag*100:.0f}%. Siết cơ bụng giữ thẳng lưng!',
            'severity': 'error', 'type': 'sag'})
    elif sag < -0.08:
        feedback.append({'bone': 'Nguc', 'message':
            f'Mông đẩy cao {abs(sag)*100:.0f}%. Hạ hông xuống!',
            'severity': 'error', 'type': 'pike'})
    return feedback


def calc_score(num_bone_errs, num_alignment_errs, num_alignment_warns):
    bone_score = max(0, 100 - num_bone_errs * 8.33)
    align_ded = num_alignment_errs * 10 + num_alignment_warns * 5
    align_score = max(0, 100 - align_ded)
    return round(bone_score * 0.7 + align_score * 0.3)

def draw_h36m_skeleton(frame, h36m_kps, err_names_set, thickness=2):
    """Vẽ H36M skeleton trực tiếp lên người, tô đỏ bone lỗi, xanh bone đúng."""
    h36m_kps = h36m_kps.astype(np.float32)
    # Build lookup: (parent, child) → bone name from BONE_DEFS
    bone_map = {(bd[0], bd[1]): bd[2] for bd in BONE_DEFS}
    for (p, c) in H36M_SKELETON:
        if h36m_kps[p, 2] > 0.1 and h36m_kps[c, 2] > 0.1:
            pt1 = (int(h36m_kps[p, 0]), int(h36m_kps[p, 1]))
            pt2 = (int(h36m_kps[c, 0]), int(h36m_kps[c, 1]))
            name = bone_map.get((p, c))
            color = (0, 0, 255) if (name and name in err_names_set) else (100, 200, 100)
            cv2.arrowedLine(frame, pt1, pt2, color, thickness, tipLength=0.12)
    for j in range(17):
        if h36m_kps[j, 2] > 0.1:
            cv2.circle(frame, (int(h36m_kps[j, 0]), int(h36m_kps[j, 1])), 4, (0, 255, 255), -1)


def draw_h36m_vector_panel(frame, student_norm, target_norm, err_names_set, h, w):
    """Vẽ panel so sánh vector 12 bone H36M (student vs standard)."""
    panel_w, panel_h = 240, 200
    px, py = w - panel_w - 10, 65
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (25, 25, 25), -1)
    cv2.putText(frame, "Bone vectors (grey=std, col=stu)",
                (px + 5, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)

    cell_w = (panel_w - 15) // 3
    cell_h = (panel_h - 20) // 4

    for i, (p_idx, c_idx, name, _) in enumerate(BONE_DEFS):
        col = i % 3
        row = i // 3
        cx = px + 8 + col * cell_w + cell_w // 2
        cy = py + 18 + row * cell_h + cell_h // 2

        sv = student_norm[c_idx] - student_norm[p_idx]
        tv = target_norm[c_idx] - target_norm[p_idx]
        is_err = name in err_names_set
        scale = 12

        # Standard vector (grey dashed)
        tex = int(cx + tv[0] * scale)
        tey = int(cy + tv[1] * scale)
        cv2.arrowedLine(frame, (cx, cy), (tex, tey), (120, 120, 120), 1, tipLength=0.2)

        # Student vector (green/red)
        sex = int(cx + sv[0] * scale)
        sey = int(cy + sv[1] * scale)
        vec_color = (0, 0, 255) if is_err else (0, 255, 0)
        cv2.arrowedLine(frame, (cx, cy), (sex, sey), vec_color, 2, tipLength=0.2)

        cv2.putText(frame, name[:7], (cx - 12, cy + cell_h // 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1)

def draw_error_overlay(frame, err_indices, h, w):
    """Vẽ overlay skeleton H36M lỗi ở góc dưới phải."""
    overlay_h, overlay_w = 150, 120
    ox, oy = w - overlay_w - 10, h - overlay_h - 10
    cv2.rectangle(frame, (ox, oy), (ox + overlay_w, oy + overlay_h), (0, 0, 0), -1)
    if not err_indices:
        cv2.putText(frame, 'No errors', (ox + 5, oy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return
    for ei, idx in enumerate(sorted(err_indices)[:6]):
        ty = oy + 15 + ei * 18
        cv2.putText(frame, f'ERR: {BONE_NAMES_LIST[idx]}', (ox + 5, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# --- Process standard video ---
print('Processing standard video...')
cap = cv2.VideoCapture(STD_VIDEO)
std_frames = []
for result in model.predict(source=STD_VIDEO, stream=True, imgsz=640, conf=0.25, device=DEVICE, verbose=False):
    kps = None
    if result.keypoints is not None and result.keypoints.data is not None:
        data = result.keypoints.data.detach().cpu().numpy().astype(np.float32)
        if data.size > 0:
            if data.shape[-1] == 2:
                c = np.ones(data.shape[:2] + (1,), np.float32)
                data = np.concatenate([data, c], axis=-1)
            scores = np.nanmean(data[:, :, 2], axis=1)
            if np.isfinite(scores).any():
                kps = data[int(np.nanargmax(scores))]
    std_frames.append(coco_to_h36m(kps) if kps is not None else np.zeros((17, 3), np.float32))
cap.release()
std_arr = np.stack(std_frames).astype(np.float32)
print(f'  Standard: {std_arr.shape}')

# Extract keyframes
kf_indices = extract_key_frames(std_arr, n_clusters=30)
checkpoints = [std_arr[i] for i in kf_indices]
print(f'  Keyframes: {len(checkpoints)}')

# --- Process student video ---
print('Processing student video...')
cap = cv2.VideoCapture(STU_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
if not out.isOpened():
    out = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

# Pre-normalize checkpoints
norm_checkpoints = [normalize_pose(cp)[:, :2] for cp in checkpoints]

window = []
cur_cp = -1
total_reps = 0
initialized = False
f_idx = 0

for result in model.predict(source=STU_VIDEO, stream=True, imgsz=640, conf=0.25, device=DEVICE, verbose=False):
    frame = result.orig_img.copy() if hasattr(result, 'orig_img') else np.zeros((h, w, 3), dtype=np.uint8)

    # Extract raw COCO keypoints (for drawing on original frame)
    kps = None
    if result.keypoints is not None and result.keypoints.data is not None:
        data = result.keypoints.data.detach().cpu().numpy().astype(np.float32)
        if data.size > 0:
            if data.shape[-1] == 2:
                c = np.ones(data.shape[:2] + (1,), np.float32)
                data = np.concatenate([data, c], axis=-1)
            scores = np.nanmean(data[:, :, 2], axis=1)
            if np.isfinite(scores).any():
                kps = data[int(np.nanargmax(scores))]

    # Convert to H36M for evaluation
    h36m = coco_to_h36m(kps) if kps is not None else np.zeros((17, 3), np.float32)
    norm = normalize_pose(h36m)[:, :2]

    # Smooth
    window.append(norm.copy())
    if len(window) > 5:
        window.pop(0)
    smoothed = norm if len(window) < 2 else np.mean(window, axis=0)

    # Evaluate
    err_names_set = set()
    if not initialized:
        best, best_err = 0, float('inf')
        for i, cp in enumerate(norm_checkpoints):
            errs = detect_errors(smoothed, cp)
            if len(errs) < best_err:
                best_err, best = len(errs), i
        cur_cp = best
        initialized = True
        status_text = f'INIT CP {best}'
        err_indices = set()
    elif cur_cp >= len(norm_checkpoints):
        total_reps += 1
        cur_cp = 0
        status_text = f'REP {total_reps}!'
        err_indices = set()
    else:
        errs = detect_errors(smoothed, norm_checkpoints[cur_cp])
        err_names_set = set(errs.keys())
        err_indices = {i for i, bd in enumerate(BONE_DEFS) if bd[2] in err_names_set}
        if len(errs) == 0:
            cur_cp += 1
            status_text = f'PASS CP {cur_cp}'
        else:
            status_text = f'WAIT CP {cur_cp} ({len(errs)} err)'

    # Draw H36M skeleton directly on person (pixel-space keypoints before normalize)
    if kps is not None:
        draw_h36m_skeleton(frame, h36m, err_names_set, thickness=2)

    # Alignment check + scoring (chỉ khi có checkpoint)
    alignment_fb = []
    score = 0
    if cur_cp >= 0 and cur_cp < len(norm_checkpoints):
        alignment_fb = check_push_up_alignment(smoothed)
        num_align_err = sum(1 for f in alignment_fb if f['severity'] == 'error')
        num_align_warn = sum(1 for f in alignment_fb if f['severity'] == 'warning')
        score = calc_score(len(err_indices), num_align_err, num_align_warn)

    # Score bar (top-center)
    bar_w = 160
    bar_x = (w - bar_w) // 2
    cv2.rectangle(frame, (bar_x, 5), (bar_x + bar_w, 22), (40, 40, 40), -1)
    fill = max(0, min(bar_w, int(bar_w * score / 100)))
    color = (0, 255, 0) if score >= 80 else (0, 255, 255) if score >= 50 else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, 5), (bar_x + fill, 22), color, -1)
    cv2.putText(frame, f'Score: {score}/100', (bar_x + 5, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # HUD
    cv2.rectangle(frame, (0, 25), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, f'Frame {f_idx} | {status_text} | CP {cur_cp}/{len(norm_checkpoints)} | Reps {total_reps}',
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Error overlay (bottom-right)
    draw_error_overlay(frame, err_indices, h, w)

    # Feedback messages (bottom-left)
    if alignment_fb:
        fb_x, fb_y = 10, h - 20 - len(alignment_fb) * 18
        cv2.rectangle(frame, (0, fb_y - 5), (360, h - 5), (0, 0, 0), -1)
        for fi, fb in enumerate(alignment_fb[:4]):
            fb_color = (0, 0, 255) if fb['severity'] == 'error' else (0, 255, 255)
            cv2.putText(frame, fb['message'][:45], (10, fb_y + fi * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, fb_color, 1)

    # Bone vector comparison panel (top-right)
    if len(smoothed) > 0 and cur_cp >= 0 and cur_cp < len(norm_checkpoints):
        draw_h36m_vector_panel(frame, smoothed, norm_checkpoints[cur_cp], err_names_set, h, w)

    out.write(frame)
    f_idx += 1

cap.release()
out.release()
print(f'\nSaved {OUT_VIDEO} ({f_idx} frames, {total_reps} reps)')
