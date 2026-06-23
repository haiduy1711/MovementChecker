// ─── H36M 17-joint skeleton ───
// Indices: 0=Root, 1=RHip, 2=RKnee, 3=RAnkle, 4=LHip, 5=LKnee, 6=LAnkle,
//          7=Spine, 8=Thorax, 9=Neck, 10=Head,
//          11=LShou, 12=LElb, 13=LWri, 14=RShou, 15=RElb, 16=RWri

const H36M_JOINT_NAMES = [
  'Root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
  'Spine', 'Thorax', 'Neck', 'Head',
  'LShou', 'LElb', 'LWri', 'RShou', 'RElb', 'RWri',
];

// Connections for drawing a full H36M skeleton (parent→child)
const H36M_SKELETON = [
  [0, 1], [0, 4],             // Root → hips
  [0, 7], [7, 8], [8, 9], [9, 10], // spine
  [14, 15], [15, 16],         // right arm
  [11, 12], [12, 13],         // left arm
  [1, 2], [2, 3],             // right leg
  [4, 5], [5, 6],             // left leg
  [11, 7], [14, 8],           // shoulders → spine
];

// The 12 evaluated bones: { parent, child, name, threshold }
// Threshold = cosine similarity minimum; arms strictest, neck loosest
const BONE_STRUCTURES = [
  { parent: 14, child: 15, name: 'R arm up',   threshold: 0.995 },
  { parent: 15, child: 16, name: 'R arm low',  threshold: 0.995 },
  { parent: 11, child: 12, name: 'L arm up',   threshold: 0.995 },
  { parent: 12, child: 13, name: 'L arm low',  threshold: 0.995 },
  { parent: 1,  child: 2,  name: 'R leg up',   threshold: 0.990 },
  { parent: 2,  child: 3,  name: 'R leg low',  threshold: 0.990 },
  { parent: 4,  child: 5,  name: 'L leg up',   threshold: 0.990 },
  { parent: 5,  child: 6,  name: 'L leg low',  threshold: 0.990 },
  { parent: 0,  child: 7,  name: 'Bung',       threshold: 0.985 },
  { parent: 7,  child: 8,  name: 'Nguc',       threshold: 0.985 },
  { parent: 8,  child: 9,  name: 'Co duoi',    threshold: 0.980 },
  { parent: 9,  child: 10, name: 'Co tren',    threshold: 0.980 },
];

module.exports = { BONE_STRUCTURES, H36M_SKELETON, H36M_JOINT_NAMES };
