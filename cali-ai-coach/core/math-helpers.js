/**
 * Tính tích vô hướng (Dot Product) của 2 vector 3D
 */
function dotProduct(v1, v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

/**
 * Tính độ dài hình học (L2 Norm / Magnitude) của vector 3D
 */
function l2Norm(v) {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

/**
 * Đối sánh toàn bộ khung xương giữa Học viên và Người mẫu chuẩn
 * @param {Array} studentJoints - Mảng 12 khớp của học viên [{x,y,z}, ...]
 * @param {Array} modelJoints - Mảng 12 khớp của người mẫu [{x,y,z}, ...]
 * @param {Array} boneStructures - Cấu trúc xương định nghĩa trong pose-geometry
 * @param {Number} tolerance - Ngưỡng chấp nhận (mặc định 0.995 ~ lệch khoảng 5.7 độ)
 */
function compareH36MFrame(studentJoints, modelJoints, boneStructures, tolerance = 0.995) {
    let frameErrors = [];

    for (let [parentIdx, childIdx, boneName] of boneStructures) {
        let sp = studentJoints[parentIdx];
        let sc = studentJoints[childIdx];
        let mp = modelJoints[parentIdx];
        let mc = modelJoints[childIdx];

        if (!sp || !sc || !mp || !mc) continue;

        // Vector components (flat numbers — no object allocation)
        let svx = sc.x - sp.x, svy = sc.y - sp.y, svz = sc.z - sp.z;
        let mvx = mc.x - mp.x, mvy = mc.y - mp.y, mvz = mc.z - mp.z;

        // L2 norms (inlined, no l2Norm() call to avoid wrapping into objects)
        let normS = Math.sqrt(svx * svx + svy * svy + svz * svz);
        let normM = Math.sqrt(mvx * mvx + mvy * mvy + mvz * mvz);

        if (normS === 0 || normM === 0) continue;

        // Cosine similarity — inlined dot product, no objects
        let cosSim = (svx * mvx + svy * mvy + svz * mvz) / (normS * normM);
        cosSim = Math.min(Math.max(cosSim, -1.0), 1.0);

        if (cosSim < tolerance) {
            // Unit vector of model (flat numbers)
            let ux = mvx / normM, uy = mvy / normM, uz = mvz / normM;

            // Target child position (flat numbers, no object)
            let tx = sp.x + ux * normS;
            let ty = sp.y + uy * normS;
            let tz = sp.z + uz * normS;

            // Euclidean distance (flat numbers, no Math.pow)
            let dx = tx - sc.x, dy = ty - sc.y, dz = tz - sc.z;
            let moveDist = Math.sqrt(dx * dx + dy * dy + dz * dz);

            frameErrors.push({
                bone: boneName,
                cosSim: cosSim,
                moveDistance: moveDist
            });
        }
    }
    return frameErrors;
}

module.exports = { dotProduct, l2Norm, compareH36MFrame };
