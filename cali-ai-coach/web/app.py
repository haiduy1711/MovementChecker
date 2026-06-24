import sys, os, cv2, numpy as np, json, time, uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent.parent))
from web.evaluator import evaluate_video, normalize_pose, cos_sim, detect_errors, check_push_up_alignment, calc_score, draw_h36m_skeleton, draw_error_overlay, BONE_DEFS, BONE_NAMES_LIST, H36M_SKELETON
from core.mediapipe_to_h36m import mediapipe_to_h36m

app = Flask(__name__)
BASE = Path(__file__).parent
DATA = BASE.parent / 'data'
UPLOAD = BASE / 'uploads'
RESULTS = BASE / 'results'
MODEL_PATH = str(DATA / 'pose_landmarker_full.task')
CHECKPOINTS_PATH = str(DATA / 'calib_checkpoints.npz')

UPLOAD.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)

# Load checkpoints once
d = np.load(CHECKPOINTS_PATH)
norm_cp = d['norm_checkpoints']


def gen_camera_feed():
    """MJPEG stream: webcam → MediaPipe → annotated frames."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    base = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base, running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_vision.PoseLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera not available\r\n'
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w, h = 640, 480
    flip = True

    window = []; raw_window = []; cur_cp = -1; total_reps = 0; initialized = False; aligned_frames = 0; f_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if flip:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose.detect_for_video(img, int(f_idx * 33.33))

        if result.pose_landmarks:
            h36m = mediapipe_to_h36m(result.pose_landmarks[0], w, h)
        else:
            h36m = np.zeros((17, 3), np.float32)

        raw_window.append(h36m.copy())
        if len(raw_window) > 7: raw_window.pop(0)
        h36m_smooth = h36m if len(raw_window) < 3 else np.mean(raw_window, axis=0).astype(np.float32)

        norm = normalize_pose(h36m_smooth)[:, :2]
        window.append(norm.copy())
        if len(window) > 7: window.pop(0)
        smoothed = norm if len(window) < 3 else np.mean(window, axis=0)

        err_names_set = set()
        status_text = ''
        if not initialized:
            aligned = True
            for p, c, n, th in BONE_DEFS:
                sv = smoothed[c] - smoothed[p]
                mv = norm_cp[0][c] - norm_cp[0][p]
                if np.linalg.norm(sv) > 1e-4 and np.linalg.norm(mv) > 1e-4:
                    if cos_sim(sv, mv) < 0.3:
                        aligned = False; break
            if aligned:
                aligned_frames += 1
                if aligned_frames >= 5:
                    best, be = 0, float('inf')
                    for i, cp in enumerate(norm_cp):
                        errs = detect_errors(smoothed, cp)
                        if len(errs) < be: be, best = len(errs), i
                    cur_cp = best
                    initialized = True
                    status_text = f'INIT CP {best}'
                else:
                    status_text = f'CAN CHINH... {aligned_frames}/5'
            else:
                aligned_frames = 0
                status_text = 'CAN CHINH...'
        elif cur_cp >= len(norm_cp):
            total_reps += 1
            cur_cp = 0
            status_text = f'REP {total_reps}!'
        else:
            errs = detect_errors(smoothed, norm_cp[cur_cp])
            err_names_set = set(errs.keys())
            if len(errs) == 0:
                cur_cp += 1
                status_text = f'PASS CP {cur_cp}'
            else:
                status_text = f'WAIT CP {cur_cp} ({len(errs)} err)'

        err_indices = {i for i, bd in enumerate(BONE_DEFS) if bd[2] in err_names_set}
        if result.pose_landmarks:
            draw_h36m_skeleton(frame, h36m_smooth, err_names_set, thickness=2)

        score = 0; fb = []
        if cur_cp >= 0 and cur_cp < len(norm_cp):
            fb = check_push_up_alignment(smoothed)
            nae = sum(1 for f in fb if f['severity'] == 'error')
            naw = sum(1 for f in fb if f['severity'] == 'warning')
            score = calc_score(len(err_indices), nae, naw)

        bw, bx = 160, (w - 160) // 2
        cv2.rectangle(frame, (bx, 5), (bx + bw, 22), (40, 40, 40), -1)
        fill = max(0, min(bw, int(bw * score / 100)))
        sc = (0, 255, 0) if score >= 80 else (0, 255, 255) if score >= 50 else (0, 0, 255)
        cv2.rectangle(frame, (bx, 5), (bx + fill, 22), sc, -1)
        cv2.putText(frame, f'Score: {score}/100', (bx + 5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, status_text, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        draw_error_overlay(frame, err_indices, h, w)

        if fb:
            fb_y = h - 20 - len(fb) * 18
            cv2.rectangle(frame, (0, fb_y - 5), (360, h - 5), (0, 0, 0), -1)
            for i, f in enumerate(fb):
                fc = (0, 255, 255) if f['severity'] == 'warning' else (0, 0, 255)
                cv2.putText(frame, f['message'], (10, fb_y + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, fc, 1)

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret: continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        f_idx += 1

    cap.release(); pose.close()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_camera_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No file', 400
    file = request.files['video']
    if not file.filename:
        return 'No file', 400

    ext = os.path.splitext(file.filename)[1] or '.mp4'
    uid = uuid.uuid4().hex[:12]
    in_name = uid + ext
    out_name = uid + '_result.mp4'
    in_path = str(UPLOAD / in_name)
    out_path = str(RESULTS / out_name)

    file.save(in_path)

    try:
        result = evaluate_video(in_path, out_path, norm_cp, MODEL_PATH)
        result['video'] = out_name
        result['id'] = uid
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)

    return render_template('result.html', **result)


@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(str(RESULTS), filename)


@app.route('/data/<filename>')
def data_file(filename):
    return send_from_directory(str(DATA), filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
