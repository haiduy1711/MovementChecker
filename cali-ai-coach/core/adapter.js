const YOLO_TO_CORE_MAPPING = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

const H36M_NAMES = [
  'Root', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
  'Spine', 'Thorax', 'NeckBase', 'Head',
  'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist',
];

function mediapipeToH36M(yoloLandmarks) {
  if (!Array.isArray(yoloLandmarks) || yoloLandmarks.length < 17) {
    throw new Error(`adapter: Yêu cầu mảng 17 điểm H36M, nhận được: ${yoloLandmarks?.length}`);
  }

  const pelvisX = yoloLandmarks[0].x;
  const pelvisY = yoloLandmarks[0].y;

  return yoloLandmarks.map((src) => ({
    x: src.x - pelvisX,
    y: src.y - pelvisY,
    z: 0,
  }));
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { mediapipeToH36M, H36M_NAMES, YOLO_TO_CORE_MAPPING };
}
