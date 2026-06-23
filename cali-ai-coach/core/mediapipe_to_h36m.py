import numpy as np

MP = {
    'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32,
}

H36M_17_NAMES = [
    'Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'NeckBase', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist',
]


def weighted_mp(mp_landmarks, indices, w, h):
    pts = []
    confs = []
    for i in indices:
        lm = mp_landmarks[i]
        pts.append([lm.x * w, lm.y * h])
        confs.append(lm.visibility)
    pts = np.array(pts, np.float32)
    confs = np.array(confs, np.float32)
    valid = confs > 0.5
    if not np.any(valid):
        return np.zeros(3, np.float32)
    pts, confs = pts[valid], confs[valid]
    weights = confs / max(confs.sum(), 1e-6)
    xy = (pts * weights[:, None]).sum(axis=0)
    return np.array([xy[0], xy[1], confs.mean()], np.float32)


def existing_mean(points):
    pts = np.asarray(points, dtype=np.float32)
    conf = pts[:, 2]
    valid = conf > 0.5
    if not np.any(valid):
        return np.zeros(3, np.float32)
    pts, conf = pts[valid], conf[valid]
    w = conf / max(conf.sum(), 1e-6)
    xy = (pts[:, :2] * w[:, None]).sum(axis=0)
    return np.array([xy[0], xy[1], conf.mean()], np.float32)


def mediapipe_to_h36m(mp_landmarks, image_w, image_h):
    """MediaPipe 33 landmarks -> H36M 17 keypoints (x, y, conf in pixel space)."""
    if mp_landmarks is None:
        return np.zeros((17, 3), np.float32)

    def get_mp(idx):
        lm = mp_landmarks[idx]
        return np.array([lm.x * image_w, lm.y * image_h, lm.visibility], np.float32)

    pelvis = weighted_mp(mp_landmarks, [23, 24], image_w, image_h)
    shoulder_c = weighted_mp(mp_landmarks, [11, 12], image_w, image_h)
    head = weighted_mp(mp_landmarks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], image_w, image_h)

    h36m = np.zeros((17, 3), np.float32)
    h36m[0] = pelvis
    h36m[1] = get_mp(24)
    h36m[2] = get_mp(26)
    h36m[3] = get_mp(28)
    h36m[4] = get_mp(23)
    h36m[5] = get_mp(25)
    h36m[6] = get_mp(27)
    h36m[7] = existing_mean([pelvis, shoulder_c])
    h36m[8] = shoulder_c
    h36m[9] = existing_mean([shoulder_c, head])
    h36m[10] = head
    h36m[11] = get_mp(11)
    h36m[12] = get_mp(13)
    h36m[13] = get_mp(15)
    h36m[14] = get_mp(12)
    h36m[15] = get_mp(14)
    h36m[16] = get_mp(16)
    return h36m.astype(np.float32)
