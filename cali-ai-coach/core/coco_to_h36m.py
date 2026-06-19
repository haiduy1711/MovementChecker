import numpy as np

COCO = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16,
}

H36M_17_NAMES = [
    'Root', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine',
    'Thorax', 'NeckBase', 'Head', 'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist',
]

def weighted_point(kps: np.ndarray, names) -> np.ndarray:
    indices = [COCO[x] if isinstance(x, str) else int(x) for x in names]
    pts = kps[indices].astype(np.float32)
    conf = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (conf > 0)
    if not np.any(valid):
        return np.zeros(3, dtype=np.float32)
    pts, conf = pts[valid], conf[valid]
    w = conf / max(float(conf.sum()), 1e-6)
    xy = (pts[:, :2] * w[:, None]).sum(axis=0)
    return np.array([xy[0], xy[1], float(conf.mean())], dtype=np.float32)


def weighted_existing(points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    conf = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (conf > 0)
    if not np.any(valid):
        return np.zeros(3, dtype=np.float32)
    pts, conf = pts[valid], conf[valid]
    w = conf / max(float(conf.sum()), 1e-6)
    xy = (pts[:, :2] * w[:, None]).sum(axis=0)
    return np.array([xy[0], xy[1], float(conf.mean())], dtype=np.float32)


def coco_to_h36m(coco_kps: np.ndarray) -> np.ndarray:
    """Convert COCO 17 keypoints -> H36M 17 joints (x, y, conf)."""
    if coco_kps.shape[-1] == 2:
        coco_kps = np.concatenate(
            [coco_kps.astype(np.float32), np.ones((coco_kps.shape[0], 1), np.float32)], axis=1)
    coco_kps = coco_kps.astype(np.float32)
    h36m = np.zeros((17, 3), np.float32)
    pelvis = weighted_point(coco_kps, ['left_hip', 'right_hip'])
    shoulder_c = weighted_point(coco_kps, ['left_shoulder', 'right_shoulder'])
    head = weighted_point(coco_kps, ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'])
    h36m[0]  = pelvis
    h36m[1]  = coco_kps[COCO['right_hip']]
    h36m[2]  = coco_kps[COCO['right_knee']]
    h36m[3]  = coco_kps[COCO['right_ankle']]
    h36m[4]  = coco_kps[COCO['left_hip']]
    h36m[5]  = coco_kps[COCO['left_knee']]
    h36m[6]  = coco_kps[COCO['left_ankle']]
    h36m[7]  = weighted_existing([pelvis, shoulder_c])
    h36m[8]  = shoulder_c
    h36m[9]  = weighted_existing([shoulder_c, head])
    h36m[10] = head
    h36m[11] = coco_kps[COCO['left_shoulder']]
    h36m[12] = coco_kps[COCO['left_elbow']]
    h36m[13] = coco_kps[COCO['left_wrist']]
    h36m[14] = coco_kps[COCO['right_shoulder']]
    h36m[15] = coco_kps[COCO['right_elbow']]
    h36m[16] = coco_kps[COCO['right_wrist']]
    return h36m.astype(np.float32)
