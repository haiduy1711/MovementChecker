const YOLO_TO_CORE_MAPPING = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

const H36M_NAMES = [
  'Root', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
  'Spine', 'Thorax', 'NeckBase', 'Head',
  'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist',
];

/**
 * Chuẩn hóa tọa độ 17 điểm H36M:
 * 1. Dịch tâm về pelvis (index 0)
 * 2. Scale bằng spine length (khoảng cách pelvis → Thorax index 8)
 */
function normalizePose(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length < 17) {
    throw new Error(`normalizePose: Yêu cầu mảng 17 điểm H36M, nhận được: ${landmarks?.length}`);
  }

  const px = landmarks[0].x;
  const py = landmarks[0].y;

  // Thorax (index 8) sau khi trừ pelvis = spine vector
  const sx = landmarks[8].x - px;
  const sy = landmarks[8].y - py;
  const spineLen = Math.sqrt(sx * sx + sy * sy) || 1;

  return landmarks.map((src) => ({
    x: (src.x - px) / spineLen,
    y: (src.y - py) / spineLen,
    z: 0,
  }));
}

/**
 * Chuyển MediaPipe 33 landmarks → H36M 17 joints (pixel space)
 * landmarks: mảng 33 object {x, y, z, visibility?} từ MediaPipe
 * imageW, imageH: kích thước ảnh gốc
 */
function mediapipe33toH36M(landmarks, imageW, imageH) {
  const h36m = new Array(17).fill(null).map(() => ({ x: 0, y: 0, z: 0 }));

  const weightedAvg = (indices) => {
    let sx = 0, sy = 0, sw = 0;
    for (const i of indices) {
      if (!landmarks[i]) continue;
      const w = landmarks[i].visibility || 0.5;
      sx += landmarks[i].x * imageW * w;
      sy += landmarks[i].y * imageH * w;
      sw += w;
    }
    if (sw < 1e-6) return { x: 0, y: 0, z: 0 };
    return { x: sx / sw, y: sy / sw, z: 0 };
  };

  const pelvis = weightedAvg([23, 24]);
  const shoulderC = weightedAvg([11, 12]);
  const head = weightedAvg([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

  const get = i => landmarks[i] ? {
    x: landmarks[i].x * imageW,
    y: landmarks[i].y * imageH,
    z: 0,
  } : { x: 0, y: 0, z: 0 };

  h36m[0] = pelvis;
  h36m[1] = get(24);  h36m[2] = get(26);  h36m[3] = get(28);
  h36m[4] = get(23);  h36m[5] = get(25);  h36m[6] = get(27);
  h36m[7] = { x: (pelvis.x + shoulderC.x) / 2, y: (pelvis.y + shoulderC.y) / 2, z: 0 };
  h36m[8] = shoulderC;
  h36m[9] = { x: (shoulderC.x + head.x) / 2, y: (shoulderC.y + head.y) / 2, z: 0 };
  h36m[10] = head;
  h36m[11] = get(11); h36m[12] = get(13); h36m[13] = get(15);
  h36m[14] = get(12); h36m[15] = get(14); h36m[16] = get(16);

  return h36m;
}

/**
 * Giữ alias cũ cho tương thích ngược
 */
function mediapipeToH36M(landmarks) {
  return normalizePose(landmarks);
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { normalizePose, mediapipe33toH36M, mediapipeToH36M, H36M_NAMES, YOLO_TO_CORE_MAPPING };
}
