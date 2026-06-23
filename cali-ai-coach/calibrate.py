"""
Calibrate: xử lý video mẫu chuẩn bằng YOLO 1 lần, lưu checkpoint đã normalize.
Sau đó live_mediapipe.py load file này để chạy real-time không cần YOLO nữa.
"""
import sys, os, cv2, torch, json, numpy as np
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from core.coco_to_h36m import coco_to_h36m
from core.keyframes import extract_key_frames

DATA = Path(__file__).parent / 'data'
STD_VIDEO = str(DATA / 'push_up_true.mp4')
OUTPUT = str(DATA / 'calib_checkpoints.npz')

BONE_DEFS = [
    (14, 15, 'R arm up', 0.995), (15, 16, 'R arm low', 0.995),
    (11, 12, 'L arm up', 0.995), (12, 13, 'L arm low', 0.995),
    (1, 2,  'R leg up', 0.990), (2, 3,  'R leg low', 0.990),
    (4, 5,  'L leg up', 0.990), (5, 6,  'L leg low', 0.990),
    (0, 7,  'Bung', 0.985), (7, 8, 'Nguc', 0.985),
    (8, 9,  'Co duoi', 0.3), (9, 10, 'Co tren', 0.3),
]

H36M_SKELETON = [
    (0, 1), (0, 4), (0, 7), (7, 8), (8, 9), (9, 10),
    (14, 15), (15, 16), (11, 12), (12, 13),
    (1, 2), (2, 3), (4, 5), (5, 6), (11, 7), (14, 8),
]


def normalize_pose(pose):
    p = pose.copy()
    p[:, :2] -= p[0:1, :2]
    spine = p[8, :2]
    sl = np.linalg.norm(spine) or 1
    p[:, :2] /= sl
    return p


def main():
    print('=== Calibration ===')
    model = YOLO('yolo26x-pose.pt')
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    print(f'Processing standard video: {STD_VIDEO}')
    cap = cv2.VideoCapture(STD_VIDEO)
    if not cap.isOpened():
        print('Cannot open standard video')
        return
    cap.release()

    std_frames = []
    for result in model.predict(source=STD_VIDEO, stream=True, imgsz=640,
                                conf=0.25, device=device, verbose=False):
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

    std_arr = np.stack(std_frames).astype(np.float32)
    print(f'  Total frames: {std_arr.shape[0]}')

    # Extract keyframes via K-Means
    kf_indices = extract_key_frames(std_arr, n_clusters=30)
    raw_checkpoints = np.stack([std_arr[i] for i in kf_indices]).astype(np.float32)
    print(f'  Keyframes: {len(kf_indices)}')

    # Pre-normalize (chỉ lấy xy)
    norm_list = [normalize_pose(cp)[:, :2] for cp in raw_checkpoints]
    norm_checkpoints = np.stack(norm_list).astype(np.float32)

    # Save
    np.savez_compressed(OUTPUT,
                        raw_checkpoints=raw_checkpoints,
                        norm_checkpoints=norm_checkpoints,
                        kf_indices=np.array(kf_indices, dtype=np.int32),
                        exercise='push_up',
                        source_video=STD_VIDEO)
    print(f'Saved: {OUTPUT}')
    print(f'  Raw: {raw_checkpoints.shape}, Norm: {norm_checkpoints.shape}')
    print('\nDone. Now live_mediapipe.py can load this file instantly.')


if __name__ == '__main__':
    main()
