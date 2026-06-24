import sys, os, cv2, numpy as np, json, time, uuid, pickle
from pathlib import Path
from collections import deque
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent.parent))
from web.evaluator import (
    evaluate_video, normalize_pose, cos_sim, detect_errors,
    check_push_up_alignment, calc_score, draw_h36m_skeleton,
    draw_error_overlay, draw_guide_skeleton, BONE_DEFS, BONE_NAMES_LIST, H36M_SKELETON,
)
from core.mediapipe_to_h36m import mediapipe_to_h36m
from core.keyframes import extract_key_frames

app = Flask(__name__)
BASE = Path(__file__).parent
DATA = BASE.parent / 'data'
UPLOAD = BASE / 'uploads'
RESULTS = BASE / 'results'
MODEL_PATH = str(DATA / 'pose_landmarker_full.task')
CHECKPOINTS_PATH = str(DATA / 'calib_checkpoints.npz')
EXERCISES_PKL_DIR = BASE / 'exercises'

UPLOAD.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True); EXERCISES_PKL_DIR.mkdir(exist_ok=True)

# Default push-up checkpoints
d = np.load(CHECKPOINTS_PATH)
DEFAULT_NORM_CP = d['norm_checkpoints']

# Global camera status (polled by frontend HUD)
cam_status = {'score': 0, 'reps': 0, 'cp': 0, 'total_cp': 30, 'status': 'WAITING', 'errors': [], 'feedbacks': []}

# Shared evaluation state for phone camera (/process_frame)
eval_state = {
    'window': deque(maxlen=7),
    'raw_window': deque(maxlen=7),
    'cur_cp': -1, 'total_reps': 0,
    'initialized': False, 'aligned_frames': 0, 'f_idx': 0, 'show_guide': True,
}
norm_cp_cache = {}

# Global MediaPipe detector (shared across requests)
_pose_detector = None


def get_pose_detector():
    global _pose_detector
    if _pose_detector is None:
        _pose_detector = make_pose_detector()
    return _pose_detector


def load_checkpoints(exercise):
    """Load norm_checkpoints for given exercise name. Falls back to default."""
    if not exercise or exercise == 'hit-dat':
        return DEFAULT_NORM_CP
    # Try PKL files from admin
    pkl_path = EXERCISES_PKL_DIR / f'{secure_filename(exercise)}.pkl'
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data['norm_checkpoints']
    return DEFAULT_NORM_CP


def get_exercise_list():
    """Return list of available exercises with metadata."""
    exercises = [
        {'id': 'hit-dat', 'name': 'Hít đất - Push-up', 'free': True, 'checkpoints': True},
        {'id': 'squat', 'name': 'Squat', 'free': True, 'checkpoints': False},
        {'id': 'plank', 'name': 'Plank', 'free': True, 'checkpoints': False},
        {'id': 'handstand', 'name': 'Handstand - Trồng chuối', 'free': False},
        {'id': 'muscle-up', 'name': 'Muscle-up - Lên xà nâng cao', 'free': False},
        {'id': 'planche', 'name': 'Planche', 'free': False},
    ]
    # Scan admin PKL files for custom exercises
    for f in EXERCISES_PKL_DIR.glob('*.pkl'):
        with open(f, 'rb') as fh:
            data = pickle.load(fh)
        eid = secure_filename(data.get('name', f.stem))
        exists = any(e['id'] == eid for e in exercises)
        if not exists:
            exercises.append({
                'id': eid, 'name': data.get('name', f.stem),
                'free': True, 'checkpoints': True, 'custom': True,
            })
    return exercises


def make_pose_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    base = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base, running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp_vision.PoseLandmarker.create_from_options(opts)


def process_pose_frame(frame, norm_cp, state, pose=None, w=640, h=480, flip=True):
    """Run MediaPipe + evaluation on one frame, return annotated frame + state updates."""
    import mediapipe as mp
    if pose is None:
        pose = get_pose_detector()

    if flip:
        frame = cv2.flip(frame, 1)

    state['f_idx'] += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose.detect_for_video(img, int(state['f_idx'] * 33.33))

    if result.pose_landmarks:
        h36m = mediapipe_to_h36m(result.pose_landmarks[0], w, h)
    else:
        h36m = np.zeros((17, 3), np.float32)

    state['raw_window'].append(h36m.copy())
    h36m_smooth = h36m if len(state['raw_window']) < 3 else np.mean(state['raw_window'], axis=0).astype(np.float32)

    norm = normalize_pose(h36m_smooth)[:, :2]
    state['window'].append(norm.copy())
    smoothed = norm if len(state['window']) < 3 else np.mean(state['window'], axis=0)

    worst_name = ''; worst_cs = 1.0
    err_names_set = set()
    status_text = ''

    if not state['initialized']:
        state['aligned_frames'] = 0
        aligned = True
        for p, c, n, th in BONE_DEFS:
            sv = smoothed[c] - smoothed[p]
            mv = norm_cp[0][c] - norm_cp[0][p]
            if np.linalg.norm(sv) > 1e-4 and np.linalg.norm(mv) > 1e-4:
                if cos_sim(sv, mv) < 0.3:
                    aligned = False; break
        if aligned:
            state['aligned_frames'] += 1
            if state['aligned_frames'] >= 4:
                best, be = 0, float('inf')
                for i, cp in enumerate(norm_cp):
                    errs, _, _ = detect_errors(smoothed, cp)
                    if len(errs) < be: be, best = len(errs), i
                state['cur_cp'] = best
                state['initialized'] = True
                state['show_guide'] = False
                status_text = f'INIT CP {best}'
            else:
                status_text = f'CAN CHINH... {state["aligned_frames"]}/4'
        else:
            state['aligned_frames'] = 0
            status_text = 'CAN CHINH...'
    elif state['cur_cp'] >= len(norm_cp):
        state['total_reps'] += 1
        state['cur_cp'] = 0
        status_text = f'REP {state["total_reps"]}!'
    else:
        errs, worst_name, worst_cs = detect_errors(smoothed, norm_cp[state['cur_cp']])
        err_names_set = set(errs.keys())
        if len(errs) == 0:
            state['cur_cp'] += 1
            status_text = f'PASS CP {state["cur_cp"]}'
        else:
            status_text = f'WAIT CP {state["cur_cp"]} ({len(errs)} err)'

    err_indices = {i for i, bd in enumerate(BONE_DEFS) if bd[2] in err_names_set}

    if state['show_guide'] and not state['initialized']:
        draw_guide_skeleton(frame, norm_cp[0], h, w)
    if result.pose_landmarks:
        draw_h36m_skeleton(frame, h36m_smooth, err_names_set, thickness=2)

    score = 0; fb = []
    if state['cur_cp'] >= 0 and state['cur_cp'] < len(norm_cp):
        fb = check_push_up_alignment(smoothed)
        nae = sum(1 for f in fb if f['severity'] == 'error')
        naw = sum(1 for f in fb if f['severity'] == 'warning')
        score = calc_score(len(err_indices), nae, naw)

    cam_status.update({
        'score': score, 'reps': state['total_reps'],
        'cp': max(0, state['cur_cp']), 'total_cp': len(norm_cp),
        'status': status_text,
        'errors': [BONE_NAMES_LIST[i] for i in err_indices],
        'feedbacks': [f['message'] for f in fb],
    })

    # Draw HUD
    bw, bx = 160, (w - 160) // 2
    cv2.rectangle(frame, (bx, 5), (bx + bw, 22), (40, 40, 40), -1)
    fill = max(0, min(bw, int(bw * score / 100)))
    sc = (0, 255, 0) if score >= 80 else (0, 255, 255) if score >= 50 else (0, 0, 255)
    cv2.rectangle(frame, (bx, 5), (bx + fill, 22), sc, -1)
    cv2.putText(frame, f'Score: {score}/100', (bx + 5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if state['show_guide'] and not state['initialized']:
        status_text = 'CAN CHINH...' if not status_text else status_text

    cv2.putText(frame, status_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if worst_name:
        cv2.putText(frame, f'Worst: {worst_name} ({worst_cs:.2f})', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    draw_error_overlay(frame, err_indices, h, w)

    if fb:
        fb_y = h - 20 - len(fb) * 18
        cv2.rectangle(frame, (0, fb_y - 5), (360, h - 5), (0, 0, 0), -1)
        for i, f in enumerate(fb):
            fc = (0, 255, 255) if f['severity'] == 'warning' else (0, 0, 255)
            cv2.putText(frame, f['message'], (10, fb_y + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fc, 1)

    return frame


def gen_camera_feed(exercise='hit-dat'):
    global cam_status, eval_state
    norm_cp = load_checkpoints(exercise)
    pose = make_pose_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera not available\r\n'
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w, h = 640, 480

    # Reset state
    eval_state = {
        'window': deque(maxlen=7), 'raw_window': deque(maxlen=7),
        'cur_cp': -1, 'total_reps': 0, 'initialized': False,
        'aligned_frames': 0, 'f_idx': 0, 'show_guide': True,
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = process_pose_frame(frame, norm_cp, eval_state, pose=pose, w=w, h=h, flip=True)
        ret2, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret2: continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release(); pose.close()


@app.route('/')
def index():
    exercises = get_exercise_list()
    return render_template('index.html', exercises=exercises)


@app.route('/exercise/<name>')
def exercise(name):
    norm_cp = load_checkpoints(name)
    has_cp = len(norm_cp) > 0
    return render_template('exercise.html', exercise=name, has_checkpoints=has_cp)


@app.route('/admin')
def admin():
    return render_template('admin.html')


@app.route('/api/status')
def api_status():
    return jsonify(cam_status)


@app.route('/api/exercises')
def api_exercises():
    return jsonify(get_exercise_list())


@app.route('/video_feed')
def video_feed():
    exercise = request.args.get('exercise', 'hit-dat')
    return Response(gen_camera_feed(exercise),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    global eval_state, norm_cp_cache
    exercise = request.form.get('exercise', 'hit-dat')
    if exercise not in norm_cp_cache:
        norm_cp_cache[exercise] = load_checkpoints(exercise)
    norm_cp = norm_cp_cache[exercise]

    if 'frame' not in request.files:
        return jsonify({'error': 'No frame'}), 400

    file = request.files['frame']
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Bad image'}), 400

    h, w = frame.shape[:2]
    annotated = process_pose_frame(frame, norm_cp, eval_state, w=w, h=h, flip=False)
    ret, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ret:
        return jsonify({'error': 'Encode failed'}), 500

    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No file', 400
    file = request.files['video']
    if not file.filename:
        return 'No file', 400
    ext = os.path.splitext(file.filename)[1] or '.mp4'
    uid = uuid.uuid4().hex[:12]
    in_path = str(UPLOAD / f'{uid}{ext}')
    out_path = str(RESULTS / f'{uid}_result.mp4')
    file.save(in_path)
    try:
        result = evaluate_video(in_path, out_path, DEFAULT_NORM_CP, MODEL_PATH)
        result['video'] = f'{uid}_result.mp4'
        result['id'] = uid
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(in_path): os.remove(in_path)
    return render_template('result.html', **result)


@app.route('/process_reference', methods=['POST'])
def process_reference():
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Missing exercise name'}), 400
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    file = request.files['video']
    if not file.filename:
        return jsonify({'error': 'No file'}), 400
    ext = os.path.splitext(file.filename)[1] or '.mp4'
    uid = uuid.uuid4().hex[:12]
    video_path = str(UPLOAD / f'{uid}{ext}')
    file.save(video_path)

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        base = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base, running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1, min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose = mp_vision.PoseLandmarker.create_from_options(opts)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_frames = []; f_idx = 0
        print(f'Processing {total} frames for "{name}"...')
        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = pose.detect_for_video(img, int(f_idx * (1000 / fps)))
            if result.pose_landmarks:
                h36m = mediapipe_to_h36m(result.pose_landmarks[0], w, h)
            else:
                h36m = np.zeros((17, 3), np.float32)
            all_frames.append(h36m); f_idx += 1
        cap.release(); pose.close()
        std_arr = np.stack(all_frames)
        kf_idx = extract_key_frames(std_arr, n_clusters=30)
        raw_cp = std_arr[kf_idx]
        norm_list = [normalize_pose(kp)[:, :2] for kp in raw_cp]
        norm_cp_new = np.stack(norm_list)
        safe_name = secure_filename(name) or f'exercise_{uid}'
        pkl_path = str(EXERCISES_PKL_DIR / f'{safe_name}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'name': name, 'uid': uid,
                'raw_checkpoints': raw_cp,
                'norm_checkpoints': norm_cp_new,
                'kf_indices': kf_idx,
                'source_video': video_path,
                'total_frames': total,
            }, f)
        os.remove(video_path)
        return jsonify({'success': True, 'name': name,
                        'keyframes': len(kf_idx), 'total_frames': total,
                        'pkl_file': f'{safe_name}.pkl'})
    except Exception as e:
        if os.path.exists(video_path): os.remove(video_path)
        return jsonify({'error': str(e)}), 500


@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(str(RESULTS), filename)


@app.route('/data/<filename>')
def data_file(filename):
    return send_from_directory(str(DATA), filename)


if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    for ip in socket.gethostbyname_ex(hostname)[2]:
        if ip.startswith('192.') or ip.startswith('10.') or ip.startswith('172.'):
            print(f'  http://{ip}:5000')
    print()
    print('Neu bi chan camera (getUserMedia):')
    print('  Chrome => chrome://flags/#unsafely-treat-insecure-origin-as-secure')
    print('  Them http://<IP_May_Chu>:5000 vao danh sach, Relaunch')
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
