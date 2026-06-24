import sys, os, cv2, json, numpy as np
from pathlib import Path
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

sys.path.insert(0, os.path.dirname(__file__))
from core.mediapipe_to_h36m import mediapipe_to_h36m

DATA = Path(__file__).parent / 'data'
CHECKPOINTS_PATH = str(DATA / 'push_up_keyframes.json')
POSE_MODEL_PATH = str(DATA / 'pose_landmarker_full.task')

H36M_SKELETON = [
    (0, 1), (0, 4),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (14, 15), (15, 16),
    (11, 12), (12, 13),
    (1, 2), (2, 3),
    (4, 5), (5, 6),
    (11, 7), (14, 8),
]

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
    worst = ('', 1.0)
    for p, c, name, thresh in BONE_DEFS:
        sv = student_xy[c] - student_xy[p]
        mv = target_xy[c] - target_xy[p]
        cs = cos_sim(sv, mv)
        if cs < thresh:
            errs[name] = True
        if cs < worst[1]:
            worst = (name, cs)
    return errs, worst[0], worst[1]


ALIGN_THRESH = 0.3
VEC_EPS = 1e-4  # vector quá ngắn → bỏ qua, tính là đúng

def align_ok(student, target):
    ok = 0; total = 0
    for p, c, name, _ in BONE_DEFS:
        sv = student[c] - student[p]
        mv = target[c] - target[p]
        sn = np.linalg.norm(sv); mn = np.linalg.norm(mv)
        if sn < VEC_EPS or mn < VEC_EPS:
            ok += 1  # vector quá ngắn → bỏ qua, auto pass
        elif cos_sim(sv, mv) >= ALIGN_THRESH:
            ok += 1
        total += 1
    return ok / total >= 0.8


def check_push_up_alignment(norm_xy):
    feedback = []
    for sho, wri, elb, side, bone in [(14, 16, 15, 'Phai', 'R arm up'),
                                       (11, 13, 12, 'Trai', 'L arm up')]:
        s = norm_xy[sho]; w = norm_xy[wri]; e = norm_xy[elb]
        dx = abs(s[0] - w[0])
        if dx > 0.05:
            sev = 'error' if dx > 0.08 else 'warning'
            feedback.append({'bone': bone, 'message':
                f'Tay {side}: vai lech ngang {dx*100:.0f}%. Dua vai ve phia truoc!',
                'severity': sev, 'type': 'alignment'})
        if e is not None:
            flare = abs(e[0] - s[0]) / (abs(s[0] - w[0]) + 1e-6)
            if flare > 2.0:
                fb_bone = 'R arm low' if side == 'Phai' else 'L arm low'
                feedback.append({'bone': fb_bone, 'message':
                    f'Tay {side}: khuyu tay bung rong. Giu khuyu sat than!',
                    'severity': 'warning', 'type': 'flare'})
    hip_y = (norm_xy[4][1] + norm_xy[1][1]) / 2
    sho_y = (norm_xy[11][1] + norm_xy[14][1]) / 2
    ank_y = (norm_xy[6][1] + norm_xy[3][1]) / 2
    exp_y = (sho_y + ank_y) / 2
    sag = hip_y - exp_y
    if sag > 0.08:
        feedback.append({'bone': 'Bung', 'message':
            f'Hong tre xg {sag*100:.0f}%. Siet co bung giu thang lung!',
            'severity': 'error', 'type': 'sag'})
    elif sag < -0.08:
        feedback.append({'bone': 'Nguc', 'message':
            f'Cac day cao {abs(sag)*100:.0f}%. Ha hong xuong!',
            'severity': 'error', 'type': 'pike'})
    return feedback


def calc_score(num_bone_errs, num_alignment_errs, num_alignment_warns):
    bone_score = max(0, 100 - num_bone_errs * 8.33)
    align_ded = num_alignment_errs * 10 + num_alignment_warns * 5
    align_score = max(0, 100 - align_ded)
    return round(bone_score * 0.7 + align_score * 0.3)


def draw_h36m_skeleton(frame, h36m_kps, err_names_set, thickness=2):
    h36m_kps = h36m_kps.astype(np.float32)
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
        tex = int(cx + tv[0] * scale)
        tey = int(cy + tv[1] * scale)
        cv2.arrowedLine(frame, (cx, cy), (tex, tey), (120, 120, 120), 1, tipLength=0.2)
        sex = int(cx + sv[0] * scale)
        sey = int(cy + sv[1] * scale)
        vec_color = (0, 0, 255) if is_err else (0, 255, 0)
        cv2.arrowedLine(frame, (cx, cy), (sex, sey), vec_color, 2, tipLength=0.2)
        cv2.putText(frame, name[:7], (cx - 12, cy + cell_h // 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1)


def draw_error_overlay(frame, err_indices, h, w):
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


def draw_guide_skeleton(frame, norm_checkpoint, h, w):
    """Vẽ khung xương ảo hướng dẫn để người dùng căn chỉnh camera."""
    if norm_checkpoint is None:
        return
    pts = norm_checkpoint[:, :2]
    xs, ys = pts[:, 0], pts[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cw_n, ch_n = x_max - x_min, y_max - y_min
    if cw_n < 0.01 or ch_n < 0.01:
        return
    scale_h = (h * 0.55) / ch_n
    scale_w = (w * 0.55) / cw_n
    scale = min(scale_h, scale_w)
    cx, cy = w // 2, h // 2
    mx_n, my_n = (x_min + x_max) / 2, (y_min + y_max) / 2

    overlay = frame.copy()
    alpha = 0.4

    bone_map = {(bd[0], bd[1]): bd[2] for bd in BONE_DEFS}
    for (p, c) in H36M_SKELETON:
        x1 = int(cx + (norm_checkpoint[p, 0] - mx_n) * scale)
        y1 = int(cy + (norm_checkpoint[p, 1] - my_n) * scale)
        x2 = int(cx + (norm_checkpoint[c, 0] - mx_n) * scale)
        y2 = int(cy + (norm_checkpoint[c, 1] - my_n) * scale)
        name = bone_map.get((p, c))
        color = (0, 255, 255) if name else (200, 200, 200)
        cv2.arrowedLine(overlay, (x1, y1), (x2, y2), color, 2, tipLength=0.12)
    for j in range(17):
        x = int(cx + (norm_checkpoint[j, 0] - mx_n) * scale)
        y = int(cy + (norm_checkpoint[j, 1] - my_n) * scale)
        cv2.circle(overlay, (x, y), 5, (0, 255, 255), -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Hướng dẫn
    tx, ty = cx - 150, int(cy + ch_n * scale / 2 + 30)
    cv2.rectangle(frame, (tx - 10, ty - 25), (tx + 310, ty + 10), (0, 0, 0), -1)
    cv2.putText(frame, 'Dung vao khung xuong gia de can chinh',
                (tx, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(frame, 'Sau do thuc hien dong tac hut dat',
                (tx, ty + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def main():
    print('Loading checkpoints...')
    # Ưu tiên file NPZ (calibrate.py) - load nhanh hơn
    npz_path = DATA / 'calib_checkpoints.npz'
    json_path = Path(CHECKPOINTS_PATH)
    if npz_path.exists():
        data = np.load(npz_path)
        raw_checkpoints = data['raw_checkpoints']
        norm_checkpoints = data['norm_checkpoints']
        print(f'  Loaded NPZ: {raw_checkpoints.shape[0]} checkpoints, norm {norm_checkpoints.shape}')
    elif json_path.exists():
        with open(str(json_path)) as f:
            raw = json.load(f)
        checkpoints_list = [np.array([[j['x'], j['y'], j.get('z', 0)] for j in frame], np.float32) for frame in raw]
        raw_checkpoints = np.stack(checkpoints_list)
        norm_checkpoints = np.stack([normalize_pose(cp)[:, :2] for cp in checkpoints_list])
        print(f'  Loaded JSON: {len(checkpoints_list)} checkpoints')
    else:
        print('No checkpoint file found! Run calibrate.py first.')
        return
    print(f'  Ready for real-time evaluation')

    print('Starting MediaPipe Pose (task API)...')
    base = mp_tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose = mp_vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0 or h == 0:
        w, h = 640, 480

    window = []      # rolling window cho normalize coord (evaluation)
    raw_window = []   # rolling window cho raw pixel-space H36M (drawing)
    cur_cp = -1
    total_reps = 0
    initialized = False
    aligned_frames = 0
    f_idx = 0

    print('Live camera running. Press Q to quit.')
    flip = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if flip:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(f_idx * 33.33)
        result = pose.detect_for_video(img, ts)

        if result.pose_landmarks:
            h36m = mediapipe_to_h36m(result.pose_landmarks[0], w, h)
        else:
            h36m = np.zeros((17, 3), np.float32)

        # Smooth raw pixel-space keypoints for stable drawing
        raw_window.append(h36m.copy())
        if len(raw_window) > 7:
            raw_window.pop(0)
        h36m_smooth = h36m if len(raw_window) < 3 else np.mean(raw_window, axis=0).astype(np.float32)

        norm = normalize_pose(h36m_smooth)[:, :2]

        window.append(norm.copy())
        if len(window) > 7:
            window.pop(0)
        smoothed = norm if len(window) < 3 else np.mean(window, axis=0)

        worst_name = ''; worst_cs = 1.0
        err_names_set = set()
        if h36m[:, 2].max() < 0.1:
            status_text = 'NO POSE'
            err_indices = set()
            draw_guide_skeleton(frame, norm_checkpoints[0], h, w)
        elif not initialized:
            aligned = align_ok(smoothed, norm_checkpoints[0])
            if aligned:
                aligned_frames += 1
                if aligned_frames >= 4:
                    # Guide biến mất, bắt đầu đánh giá
                    best, best_err = 0, float('inf')
                    for i, cp in enumerate(norm_checkpoints):
                        e, _, _ = detect_errors(smoothed, cp)
                        if len(e) < best_err:
                            best_err, best = len(e), i
                    cur_cp = best
                    initialized = True
                    status_text = f'INIT CP {best}'
                    err_indices = set()
                else:
                    status_text = f'CAN CHINH... ({aligned_frames}/4)'
                    err_indices = set()
                    draw_guide_skeleton(frame, norm_checkpoints[0], h, w)
            else:
                aligned_frames = 0
                status_text = 'CAN CHINH...'
                err_indices = set()
                draw_guide_skeleton(frame, norm_checkpoints[0], h, w)
        elif cur_cp >= len(norm_checkpoints):
            total_reps += 1
            cur_cp = 0
            status_text = f'REP {total_reps}!'
            err_indices = set()
        else:
            errs, worst_name, worst_cs = detect_errors(smoothed, norm_checkpoints[cur_cp])
            err_names_set = set(errs.keys())
            err_indices = {i for i, bd in enumerate(BONE_DEFS) if bd[2] in err_names_set}
            if len(errs) == 0:
                cur_cp += 1
                status_text = f'PASS CP {cur_cp}'
            else:
                status_text = f'WAIT CP {cur_cp} ({len(errs)} err)'

        if h36m[:, 2].max() > 0.1:
            draw_h36m_skeleton(frame, h36m_smooth, err_names_set, thickness=2)

        alignment_fb = []
        score = 0
        if cur_cp >= 0 and cur_cp < len(norm_checkpoints):
            alignment_fb = check_push_up_alignment(smoothed)
            num_align_err = sum(1 for f in alignment_fb if f['severity'] == 'error')
            num_align_warn = sum(1 for f in alignment_fb if f['severity'] == 'warning')
            score = calc_score(len(err_indices), num_align_err, num_align_warn)

        bar_w = 160
        bar_x = (w - bar_w) // 2
        cv2.rectangle(frame, (bar_x, 5), (bar_x + bar_w, 22), (40, 40, 40), -1)
        fill = max(0, min(bar_w, int(bar_w * score / 100)))
        color = (0, 255, 0) if score >= 80 else (0, 255, 255) if score >= 50 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, 5), (bar_x + fill, 22), color, -1)
        cv2.putText(frame, f'Score: {score}/100', (bar_x + 5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.rectangle(frame, (0, 25), (w, 80), (0, 0, 0), -1)
        cv2.putText(frame, f'Frame {f_idx} | {status_text} | CP {cur_cp}/{len(norm_checkpoints)} | Reps {total_reps}',
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if worst_name:
            cv2.putText(frame, f'Worst: {worst_name} ({worst_cs:.2f})', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        draw_error_overlay(frame, err_indices, h, w)

        if alignment_fb:
            fb_y = h - 20 - len(alignment_fb) * 18
            cv2.rectangle(frame, (0, fb_y - 5), (360, h - 5), (0, 0, 0), -1)
            for fi, fb in enumerate(alignment_fb[:4]):
                fb_color = (0, 0, 255) if fb['severity'] == 'error' else (0, 255, 255)
                cv2.putText(frame, fb['message'][:45], (10, fb_y + fi * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, fb_color, 1)

        cv2.imshow('Cali AI Coach - Live (MediaPipe)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        f_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f'Done. {f_idx} frames processed, {total_reps} reps.')


if __name__ == '__main__':
    main()
