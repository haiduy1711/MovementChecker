import sys, os, cv2, json, numpy as np
from pathlib import Path
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

sys.path.insert(0, os.path.dirname(__file__))
from core.mediapipe_to_h36m import mediapipe_to_h36m

DATA = Path(__file__).parent / 'data'
STU_VIDEO = str(DATA / 'push_up_wrong.mp4')
OUT_VIDEO = str(DATA / 'eval_mediapipe_result.mp4')
MODEL_PATH = str(DATA / 'pose_landmarker_full.task')
CHECKPOINTS_NPZ = str(DATA / 'calib_checkpoints.npz')

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


ALIGN_THRESH = 0.3  # relaxed threshold for guide alignment check

def detect_errors(s, t, th_factor=1.0):
    e = {}
    for p, c, n, th in BONE_DEFS:
        sv = s[c] - s[p]; mv = t[c] - t[p]
        if cos_sim(sv, mv) < th * th_factor: e[n] = True
    return e


def align_ok(student, target):
    VEC_EPS = 1e-4
    ok = 0; total = 0
    for p, c, n, _ in BONE_DEFS:
        sv = student[c] - student[p]; mv = target[c] - target[p]
        sn = np.linalg.norm(sv); mn = np.linalg.norm(mv)
        if sn < VEC_EPS or mn < VEC_EPS:
            ok += 1
        elif cos_sim(sv, mv) >= ALIGN_THRESH:
            ok += 1
        total += 1
    return ok == total


def check_push_up_alignment(norm_xy):
    fb = []
    for sho, wri, elb, side, bone in [(14, 16, 15, 'Phai', 'R arm up'),
                                       (11, 13, 12, 'Trai', 'L arm up')]:
        s = norm_xy[sho]; w = norm_xy[wri]; e = norm_xy[elb]
        dx = abs(s[0] - w[0])
        if dx > 0.05:
            sev = 'error' if dx > 0.08 else 'warning'
            fb.append({'bone': bone, 'message': f'Tay {side}: vai lech ngang {dx*100:.0f}%. Dua vai ve phia truoc!',
                       'severity': sev, 'type': 'alignment'})
        if e is not None:
            flare = abs(e[0] - s[0]) / (abs(s[0] - w[0]) + 1e-6)
            if flare > 2.0:
                bn = 'R arm low' if side == 'Phai' else 'L arm low'
                fb.append({'bone': bn, 'message': f'Tay {side}: khuyu tay bung rong. Giu khuyu sat than!',
                           'severity': 'warning', 'type': 'flare'})
    hip_y = (norm_xy[4][1] + norm_xy[1][1]) / 2
    sho_y = (norm_xy[11][1] + norm_xy[14][1]) / 2
    ank_y = (norm_xy[6][1] + norm_xy[3][1]) / 2
    sag = hip_y - (sho_y + ank_y) / 2
    if sag > 0.08:
        fb.append({'bone': 'Bung', 'message': f'Hong tre xg {sag*100:.0f}%. Siet co bung!',
                   'severity': 'error', 'type': 'sag'})
    elif sag < -0.08:
        fb.append({'bone': 'Nguc', 'message': f'Cong lung {abs(sag)*100:.0f}%. Ha hong xuong!',
                   'severity': 'error', 'type': 'pike'})
    return fb


def calc_score(n_bone_err, n_align_err, n_align_warn):
    bs = max(0, 100 - n_bone_err * 8.33)
    ad = n_align_err * 10 + n_align_warn * 5
    return round(bs * 0.7 + max(0, 100 - ad) * 0.3)


def draw_h36m_skeleton(frame, h36m_kps, err_names_set, thickness=2):
    h36m_kps = h36m_kps.astype(np.float32)
    bone_map = {(bd[0], bd[1]): bd[2] for bd in BONE_DEFS}
    for (p, c) in H36M_SKELETON:
        if h36m_kps[p, 2] > 0.1 and h36m_kps[c, 2] > 0.1:
            pt1 = (int(h36m_kps[p, 0]), int(h36m_kps[p, 1]))
            pt2 = (int(h36m_kps[c, 0]), int(h36m_kps[c, 1]))
            name = bone_map.get((p, c))
            col = (0, 0, 255) if (name and name in err_names_set) else (100, 200, 100)
            cv2.arrowedLine(frame, pt1, pt2, col, thickness, tipLength=0.12)
    for j in range(17):
        if h36m_kps[j, 2] > 0.1:
            cv2.circle(frame, (int(h36m_kps[j, 0]), int(h36m_kps[j, 1])), 4, (0, 255, 255), -1)


def draw_h36m_vector_panel(frame, s_norm, t_norm, err_names_set, h, w):
    pw, ph = 240, 200; px, py = w - pw - 10, 65
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (25, 25, 25), -1)
    cv2.putText(frame, "Bone vectors (grey=std, col=stu)", (px + 5, py + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
    cw = (pw - 15) // 3; ch = (ph - 20) // 4
    for i, (pi, ci, name, _) in enumerate(BONE_DEFS):
        col = i % 3; row = i // 3
        cx = px + 8 + col * cw + cw // 2; cy = py + 18 + row * ch + ch // 2
        sv = s_norm[ci] - s_norm[pi]; tv = t_norm[ci] - t_norm[pi]
        is_err = name in err_names_set; sc = 12
        tex = int(cx + tv[0] * sc); tey = int(cy + tv[1] * sc)
        cv2.arrowedLine(frame, (cx, cy), (tex, tey), (120, 120, 120), 1, tipLength=0.2)
        sex = int(cx + sv[0] * sc); sey = int(cy + sv[1] * sc)
        vc = (0, 0, 255) if is_err else (0, 255, 0)
        cv2.arrowedLine(frame, (cx, cy), (sex, sey), vc, 2, tipLength=0.2)
        cv2.putText(frame, name[:7], (cx - 12, cy + ch // 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1)


def draw_error_overlay(frame, err_indices, h, w):
    ow, oh = 120, 150; ox, oy = w - ow - 10, h - oh - 10
    cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (0, 0, 0), -1)
    if not err_indices:
        cv2.putText(frame, 'No errors', (ox + 5, oy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1); return
    for ei, idx in enumerate(sorted(err_indices)[:6]):
        cv2.putText(frame, f'ERR: {BONE_NAMES_LIST[idx]}', (ox + 5, oy + 15 + ei * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


def main():
    print('Loading checkpoints from NPZ...')
    d = np.load(CHECKPOINTS_NPZ)
    norm_cp = d['norm_checkpoints']
    raw_cp = d['raw_checkpoints']
    print(f'  {len(norm_cp)} checkpoints loaded')

    print('Loading MediaPipe Pose...')
    base = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base, running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_vision.PoseLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(STU_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))
    if not out.isOpened():
        out = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

    print(f'Video: {total} frames, {w}x{h}, {fps}fps')

    window = []; raw_window = []; cur_cp = -1; total_reps = 0; initialized = False; f_idx = 0

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
        if not initialized:
            best, be = 0, float('inf')
            for i, cp in enumerate(norm_cp):
                errs = detect_errors(smoothed, cp)
                if len(errs) < be:
                    be, best = len(errs), i
            cur_cp = best
            initialized = True
            status_text = f'INIT CP {best}'
            err_indices = set()
        elif cur_cp >= len(norm_cp):
            total_reps += 1
            cur_cp = 0
            status_text = f'REP {total_reps}!'
            err_indices = set()
        else:
            errs = detect_errors(smoothed, norm_cp[cur_cp])
            err_names_set = set(errs.keys())
            err_indices = {i for i, bd in enumerate(BONE_DEFS) if bd[2] in err_names_set}
            if len(errs) == 0:
                cur_cp += 1
                status_text = f'PASS CP {cur_cp}'
            else:
                status_text = f'WAIT CP {cur_cp} ({len(errs)} err)'

        if result.pose_landmarks:
            draw_h36m_skeleton(frame, h36m_smooth, err_names_set, thickness=2)

        alignment_fb = []; score = 0
        if cur_cp >= 0 and cur_cp < len(norm_cp):
            alignment_fb = check_push_up_alignment(smoothed)
            nae = sum(1 for f in alignment_fb if f['severity'] == 'error')
            naw = sum(1 for f in alignment_fb if f['severity'] == 'warning')
            score = calc_score(len(err_indices), nae, naw)

        bw, bx = 160, (w - 160) // 2
        cv2.rectangle(frame, (bx, 5), (bx + bw, 22), (40, 40, 40), -1)
        fill = max(0, min(bw, int(bw * score / 100)))
        sc = (0, 255, 0) if score >= 80 else (0, 255, 255) if score >= 50 else (0, 0, 255)
        cv2.rectangle(frame, (bx, 5), (bx + fill, 22), sc, -1)
        cv2.putText(frame, f'Score: {score}/100', (bx + 5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.rectangle(frame, (0, 25), (w, 62), (0, 0, 0), -1)
        cv2.putText(frame, f'Frame {f_idx} | {status_text} | CP {cur_cp}/{len(norm_cp)} | Reps {total_reps}',
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        draw_error_overlay(frame, err_indices, h, w)

        if alignment_fb:
            fb_y = h - 20 - len(alignment_fb) * 18
            cv2.rectangle(frame, (0, fb_y - 5), (360, h - 5), (0, 0, 0), -1)
            for fi, fb in enumerate(alignment_fb[:4]):
                c = (0, 0, 255) if fb['severity'] == 'error' else (0, 255, 255)
                cv2.putText(frame, fb['message'][:45], (10, fb_y + fi * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        if len(smoothed) > 0 and cur_cp >= 0 and cur_cp < len(norm_cp):
            draw_h36m_vector_panel(frame, smoothed, norm_cp[cur_cp], err_names_set, h, w)

        out.write(frame)
        f_idx += 1
        if f_idx % 30 == 0:
            print(f'  {f_idx}/{total} frames')

    cap.release()
    out.release()
    pose.close()
    print(f'\nSaved {OUT_VIDEO} ({f_idx} frames, {total_reps} reps)')


if __name__ == '__main__':
    main()
