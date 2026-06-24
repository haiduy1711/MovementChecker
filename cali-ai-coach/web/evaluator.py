import sys, os, cv2, numpy as np, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.mediapipe_to_h36m import mediapipe_to_h36m

H36M_SKELETON = [
    (0, 1), (0, 4), (0, 7), (7, 8), (8, 9), (9, 10),
    (14, 15), (15, 16), (11, 12), (12, 13),
    (1, 2), (2, 3), (4, 5), (5, 6), (11, 7), (14, 8),
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


def normalize_pose(p):
    p = p.copy(); p[:, :2] -= p[0:1, :2]
    sl = np.linalg.norm(p[8, :2]) or 1; p[:, :2] /= sl; return p


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def detect_errors(s, t):
    e = {}; worst = ('', 1.0)
    for p, c, n, th in BONE_DEFS:
        sv = s[c] - s[p]; mv = t[c] - t[p]
        cs = cos_sim(sv, mv)
        if cs < th: e[n] = True
        if cs < worst[1]: worst = (n, cs)
    return e, worst[0], worst[1]


def check_push_up_alignment(norm_xy):
    fb = []
    for sho, wri, elb, side, bone in [(14, 16, 15, 'Phai', 'R arm up'),
                                       (11, 13, 12, 'Trai', 'L arm up')]:
        s = norm_xy[sho]; w = norm_xy[wri]; e = norm_xy[elb]
        dx = abs(s[0] - w[0])
        if dx > 0.05:
            fb.append({'bone': bone, 'message': f'Tay {side}: vai lech ngang {dx*100:.0f}%',
                       'severity': 'error' if dx > 0.08 else 'warning', 'type': 'alignment'})
        if e is not None:
            flare = abs(e[0] - s[0]) / (abs(s[0] - w[0]) + 1e-6)
            if flare > 2.0:
                bn = 'R arm low' if side == 'Phai' else 'L arm low'
                fb.append({'bone': bn, 'message': f'Tay {side}: khuyu tay bung rong',
                           'severity': 'warning', 'type': 'flare'})
    hip_y = (norm_xy[4][1] + norm_xy[1][1]) / 2
    sho_y = (norm_xy[11][1] + norm_xy[14][1]) / 2
    dy = sho_y - hip_y
    if dy > 0.15:
        fb.append({'bone': 'Nguc', 'message': f'Hong sag {dy*100:.0f}%. Giu mong!',
                   'severity': 'error', 'type': 'sag'})
    elif dy > 0.08:
        fb.append({'bone': 'Nguc', 'message': f'Hong hoi sag {dy*100:.0f}%',
                   'severity': 'warning', 'type': 'sag'})
    return fb


def calc_score(n_bone_errs, n_align_errs, n_align_warns):
    bs = max(0, 100 - n_bone_errs * 8.33)
    ad = n_align_errs * 10 + n_align_warns * 5
    as_ = max(0, 100 - ad)
    return round(bs * 0.7 + as_ * 0.3)


def draw_h36m_skeleton(frame, kps, err_names_set, thickness=2):
    kps = kps.astype(np.float32)
    bone_map = {(bd[0], bd[1]): bd[2] for bd in BONE_DEFS}
    for (p, c) in H36M_SKELETON:
        if kps[p, 2] > 0.1 and kps[c, 2] > 0.1:
            pt1 = (int(kps[p, 0]), int(kps[p, 1]))
            pt2 = (int(kps[c, 0]), int(kps[c, 1]))
            name = bone_map.get((p, c))
            color = (0, 0, 255) if (name and name in err_names_set) else (0, 255, 0)
            cv2.arrowedLine(frame, pt1, pt2, color, thickness, tipLength=0.2)
    for j in range(17):
        if kps[j, 2] > 0.1:
            cv2.circle(frame, (int(kps[j, 0]), int(kps[j, 1])), 4, (0, 255, 255), -1)


def draw_error_overlay(frame, err_indices, h, w):
    ow, oh = 120, 150; ox, oy = w - ow - 10, h - oh - 10
    cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (0, 0, 0), -1)
    if not err_indices:
        cv2.putText(frame, 'No errors', (ox + 5, oy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return
    for ei, idx in enumerate(sorted(err_indices)[:6]):
        cv2.putText(frame, f'ERR: {BONE_NAMES_LIST[idx]}', (ox + 5, oy + 15 + ei * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


def draw_guide_skeleton(frame, norm_cp, h, w):
    """Vẽ khung xương ảo mờ từ normalized checkpoint."""
    if norm_cp is None:
        return
    pts = norm_cp[:, :2]
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
    alpha = 0.35
    bone_map = {(bd[0], bd[1]): bd[2] for bd in BONE_DEFS}
    for (p, c) in H36M_SKELETON:
        x1 = int(cx + (norm_cp[p, 0] - mx_n) * scale)
        y1 = int(cy + (norm_cp[p, 1] - my_n) * scale)
        x2 = int(cx + (norm_cp[c, 0] - mx_n) * scale)
        y2 = int(cy + (norm_cp[c, 1] - my_n) * scale)
        name = bone_map.get((p, c))
        color = (0, 255, 255) if name else (200, 200, 200)
        cv2.arrowedLine(overlay, (x1, y1), (x2, y2), color, 2, tipLength=0.12)
    for j in range(17):
        x = int(cx + (norm_cp[j, 0] - mx_n) * scale)
        y = int(cy + (norm_cp[j, 1] - my_n) * scale)
        cv2.circle(overlay, (x, y), 5, (0, 255, 255), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def evaluate_video(input_path, output_path, norm_cp, model_path, progress_cb=None):
    """Process video through MediaPipe, write annotated output, return summary."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    base = mp_tasks.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base, running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_vision.PoseLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

    window = []; raw_window = []; cur_cp = -1; total_reps = 0; initialized = False
    all_scores = []; f_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose.detect_for_video(img, int(f_idx * (1000 / fps)))

        if result.pose_landmarks:
            h36m = mediapipe_to_h36m(result.pose_landmarks[0], w, h)
        else:
            h36m = np.zeros((17, 3), np.float32)

        raw_window.append(h36m.copy())
        if len(raw_window) > 7:
            raw_window.pop(0)
        h36m_smooth = h36m if len(raw_window) < 3 else np.mean(raw_window, axis=0).astype(np.float32)

        norm = normalize_pose(h36m_smooth)[:, :2]
        window.append(norm.copy())
        if len(window) > 7:
            window.pop(0)
        smoothed = norm if len(window) < 3 else np.mean(window, axis=0)

        err_names_set = set()
        err_indices = set()
        if not initialized:
            best, be = 0, float('inf')
            for i, cp in enumerate(norm_cp):
                errs, _, _ = detect_errors(smoothed, cp)
                if len(errs) < be:
                    be, best = len(errs), i
            cur_cp = best
            initialized = True
        elif cur_cp >= len(norm_cp):
            total_reps += 1
            cur_cp = 0
        else:
            errs, worst_name, worst_cs = detect_errors(smoothed, norm_cp[cur_cp])
            err_names_set = set(errs.keys())
            err_indices = {i for i, bd in enumerate(BONE_DEFS) if bd[2] in err_names_set}
            if len(errs) == 0:
                cur_cp += 1

        if result.pose_landmarks:
            draw_h36m_skeleton(frame, h36m_smooth, err_names_set, thickness=2)

        score = 0
        alignment_fb = []
        if cur_cp >= 0 and cur_cp < len(norm_cp):
            alignment_fb = check_push_up_alignment(smoothed)
            nae = sum(1 for f in alignment_fb if f['severity'] == 'error')
            naw = sum(1 for f in alignment_fb if f['severity'] == 'warning')
            score = calc_score(len(err_indices), nae, naw)
            all_scores.append(score)

        bw, bx = 160, (w - 160) // 2
        cv2.rectangle(frame, (bx, 5), (bx + bw, 22), (40, 40, 40), -1)
        fill = max(0, min(bw, int(bw * score / 100)))
        sc = (0, 255, 0) if score >= 80 else (0, 255, 255) if score >= 50 else (0, 0, 255)
        cv2.rectangle(frame, (bx, 5), (bx + fill, 22), sc, -1)
        cv2.putText(frame, f'Score: {score}/100', (bx + 5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(frame, f'Frame {f_idx} | CP {cur_cp}/{len(norm_cp)} | Reps {total_reps}',
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        draw_error_overlay(frame, err_indices, h, w)

        if alignment_fb:
            fb_y = h - 20 - len(alignment_fb) * 18
            cv2.rectangle(frame, (0, fb_y - 5), (360, h - 5), (0, 0, 0), -1)
            for i, fb in enumerate(alignment_fb):
                fc = (0, 255, 255) if fb['severity'] == 'warning' else (0, 0, 255)
                cv2.putText(frame, fb['message'], (10, fb_y + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, fc, 1)

        out.write(frame)
        f_idx += 1
        if progress_cb and f_idx % 10 == 0:
            progress_cb(f_idx, total)

    cap.release(); out.release(); pose.close()
    avg_score = round(np.mean(all_scores)) if all_scores else 0
    return {'total_frames': f_idx, 'reps': total_reps, 'avg_score': avg_score,
            'peak_score': max(all_scores) if all_scores else 0}
