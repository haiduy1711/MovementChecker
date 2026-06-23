/**
 * Công cụ chẩn đoán Normalization
 * Log tọa độ XY của vai (H36M[11]=LShoulder, H36M[14]=RShoulder)
 * và hông (H36M[4]=LHip, H36M[1]=RHip) sau normalize.
 * Cảnh báo nếu giá trị vượt [-1, 1].
 */
const fs = require('fs');
const path = require('path');
const { normalizePose, mediapipeToH36M, H36M_NAMES } = require('./core/adapter');

// Load test data
const files = [
  'data/standard/push_up_h36m.json',
  'data/test/student_h36m.json',
];
if (!fs.existsSync(files[0])) {
  console.error('Missing data files. Run test_with_videos.py or Colab export first.');
}

function analyze(label, landmarks) {
  console.log(`\n=== ${label} ===`);
  console.log('Raw 17-point (first 3):', JSON.stringify(landmarks.slice(0, 3)));

  const normalized = normalizePose(landmarks);

  // H36M indices: LShoulder=11, RShoulder=14, LHip=4, RHip=1
  const checks = [
    { idx: 11, name: 'LShoulder' },
    { idx: 14, name: 'RShoulder' },
    { idx: 4,  name: 'LHip'      },
    { idx: 1,  name: 'RHip'      },
  ];

  let outOfRange = false;
  for (const { idx, name } of checks) {
    const p = normalized[idx];
    const inRange = Math.abs(p.x) <= 1 && Math.abs(p.y) <= 1;
    if (!inRange) outOfRange = true;
    const warn = inRange ? '' : ' *** OUT OF RANGE ***';
    console.log(`  ${name}[${idx}]: (${p.x.toFixed(4)}, ${p.y.toFixed(4)})${warn}`);
  }

  // Spine length after center
  const spineX = landmarks[8].x - landmarks[0].x;
  const spineY = landmarks[8].y - landmarks[0].y;
  const spineLen = Math.sqrt(spineX * spineX + spineY * spineY);
  console.log(`  Spine length: ${spineLen.toFixed(4)}`);

  if (outOfRange) {
    console.log('  ❌ NORMALIZE THẤT BẠI: có giá trị vượt [-1, 1]');
  } else {
    console.log('  ✅ Normalize OK: mọi giá trị trong [-1, 1]');
  }
}

// Load and analyze
for (const f of files) {
  if (!fs.existsSync(f)) continue;
  const data = JSON.parse(fs.readFileSync(f, 'utf-8'));
  for (const frame of data.frames) {
    analyze(`${path.basename(f)} frame ${frame.frame} (${frame.phase || ''})`, frame.landmarks);
  }
}

// Also check keyframes
const kfPath = 'data/push_up_keyframes.json';
if (fs.existsSync(kfPath)) {
  const kf = JSON.parse(fs.readFileSync(kfPath, 'utf-8'));
  analyze(`${path.basename(kfPath)} checkpoint 0`, kf[0]);
  analyze(`${path.basename(kfPath)} checkpoint 15`, kf[15]);
  analyze(`${path.basename(kfPath)} checkpoint 29`, kf[29]);
}
