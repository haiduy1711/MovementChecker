function dotProduct(v1, v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

function l2Norm(v) {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

function getBoneVector(joints, bone) {
    const p = joints[bone.parent];
    const c = joints[bone.child];
    if (!p || !c) return null;
    return { x: c.x - p.x, y: c.y - p.y, z: c.z - p.z };
}

function cosineSimilarity(a, b) {
    const dot = a.x * b.x + a.y * b.y + a.z * b.z;
    const na = Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    const nb = Math.sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
    if (na === 0 || nb === 0) return 0;
    return Math.min(Math.max(dot / (na * nb), -1.0), 1.0);
}

function compareH36MFrame(studentJoints, modelJoints, boneStructures, globalThreshold) {
    return boneStructures.map(bone => {
        const vS = getBoneVector(studentJoints, bone);
        const vM = getBoneVector(modelJoints, bone);
        if (!vS || !vM) {
            return { bone: bone.name, sim: 0, isCorrect: false };
        }
        const sim = cosineSimilarity(vS, vM);
        const threshold = bone.threshold || globalThreshold || 0.995;
        return { bone: bone.name, sim, isCorrect: sim >= threshold };
    }).filter(r => !r.isCorrect);
}

/**
 * Kiểm tra alignment tư thế Push-up dựa trên tọa độ H36M đã normalize.
 * Vai và cổ tay phải thẳng hàng dọc (sai lệch x nhỏ).
 * @param {Array} pose - Mảng 17 điểm H36M {x, y, z} đã normalize
 * @returns {Array} Mảng feedback { bone, message, severity, type }
 */
function checkPushUpAlignment(pose) {
    const feedback = [];
    const checks = [
        { sho: 14, wri: 16, elbow: 15, side: 'Phải', bone: 'R arm up' },
        { sho: 11, wri: 13, elbow: 12, side: 'Trái', bone: 'L arm up' },
    ];
    for (const { sho, wri, elbow, side, bone } of checks) {
        const s = pose[sho], w = pose[wri], e = pose[elbow];
        if (!s || !w) continue;

        // 1. Vai-Cổ tay thẳng hàng dọc
        const dx = Math.abs(s.x - w.x);
        if (dx > 0.05) {
            const sev = dx > 0.08 ? 'error' : 'warning';
            feedback.push({
                bone,
                message: `Tay ${side}: vai lệch ngang ${(dx * 100).toFixed(0)}% so với cổ tay. Đưa vai về phía trước!`,
                severity: sev,
                type: 'alignment'
            });
        }

        // 2. Khuỷu tay không bị bung rộng
        if (e) {
            const elbowFlare = Math.abs(e.x - s.x) / (Math.abs(s.x - w.x) + 1e-6);
            if (elbowFlare > 2.0) {
                feedback.push({
                    bone: bone === 'R arm up' ? 'R arm low' : 'L arm low',
                    message: `Tay ${side}: khuỷu tay bung rộng. Giữ khuỷu sát thân người!`,
                    severity: 'warning',
                    type: 'flare'
                });
            }
        }
    }

    // 3. Hông không bị trễ (sag) hay cong (pike)
    const lHip = pose[4], rHip = pose[1];
    const lSho = pose[11], rSho = pose[14];
    const lAnk = pose[6], rAnk = pose[3];
    if (lHip && lSho && lAnk) {
        const hipY = (lHip.y + rHip.y) / 2;
        const shoY = (lSho.y + rSho.y) / 2;
        const ankY = (lAnk.y + rAnk.y) / 2;
        const expectedY = (shoY + ankY) / 2;
        const sag = hipY - expectedY;
        if (sag > 0.08) {
            feedback.push({
                bone: 'Bung',
                message: `Hông bị trễ xuống ${(sag * 100).toFixed(0)}%. Siết cơ bụng để giữ thẳng lưng!`,
                severity: 'error',
                type: 'sag'
            });
        } else if (sag < -0.08) {
            feedback.push({
                bone: 'Nguc',
                message: `Mông đẩy cao quá ${(Math.abs(sag) * 100).toFixed(0)}%. Hạ hông xuống để giữ thẳng người!`,
                severity: 'error',
                type: 'pike'
            });
        }
    }

    return feedback;
}

/**
 * Tính điểm tổng thể cho một frame dựa trên lỗi bone và alignment.
 * @param {Array} frameErrors - Kết quả từ compareH36MFrame
 * @param {Array} alignmentFeedback - Kết quả từ checkPushUpAlignment
 * @returns {Object} { total, boneScore, alignmentScore, deductions }
 */
function scorePose(frameErrors, alignmentFeedback) {
    const boneScore = Math.max(0, 100 - frameErrors.length * 8.33);
    const alignmentDeductions = alignmentFeedback.reduce((sum, f) => {
        return sum + (f.severity === 'error' ? 10 : 5);
    }, 0);
    const alignmentScore = Math.max(0, 100 - alignmentDeductions);
    const total = Math.round((boneScore * 0.7 + alignmentScore * 0.3));
    return {
        total,
        boneScore: Math.round(boneScore),
        alignmentScore,
        deductions: {
            bones: frameErrors.length,
            alignments: alignmentFeedback.length
        }
    };
}

module.exports = { dotProduct, l2Norm, getBoneVector, cosineSimilarity, compareH36MFrame, checkPushUpAlignment, scorePose };
