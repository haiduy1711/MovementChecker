const fs = require('fs');
const { BONE_STRUCTURES } = require('./pose-geometry');
const { compareH36MFrame, cosineSimilarity, getBoneVector } = require('./math-helpers');
const { normalizePose } = require('./adapter');

/**
 * Tìm frame mẫu khớp nhất với frame học viên trong cửa sổ trượt
 */
function findBestMatch(studentPose, standardFrames, currentIdx, windowSize) {
    const start = Math.max(0, currentIdx - windowSize);
    const end = Math.min(standardFrames.length - 1, currentIdx + windowSize);

    let bestIdx = currentIdx;
    let bestAvgSim = -Infinity;

    for (let j = start; j <= end; j++) {
        let sumSim = 0, count = 0;
        for (const bone of BONE_STRUCTURES) {
            const vS = getBoneVector(studentPose, bone);
            const vM = getBoneVector(standardFrames[j], bone);
            if (vS && vM) {
                sumSim += cosineSimilarity(vS, vM);
                count++;
            }
        }
        const avgSim = count > 0 ? sumSim / count : -Infinity;
        if (avgSim > bestAvgSim) {
            bestAvgSim = avgSim;
            bestIdx = j;
        }
    }
    return { bestIdx, bestAvgSim };
}

/**
 * Chạy đường ống đối sánh dữ liệu giữa học viên và khuôn mẫu
 * @param {String} studentPath - Đường dẫn file JSON học viên
 * @param {String} standardPath - Đường dẫn file JSON mẫu chuẩn
 * @param {Object} options
 * @param {Number} options.slidingWindow - Kích thước cửa sổ trượt (mặc định 5)
 * @returns {Array} Mảng kết quả đánh giá chi tiết từng frame
 */
function runPipeline(studentPath, standardPath, options = {}) {
    const windowSize = options.slidingWindow || 5;

    const studentData = JSON.parse(fs.readFileSync(studentPath, 'utf-8'));
    const standardData = JSON.parse(fs.readFileSync(standardPath, 'utf-8'));

    const studentFrames = studentData.frames;
    const standardFrames = standardData.frames;

    const evaluationReport = [];

    console.log(`Pipeline bắt đầu: ${studentFrames.length} student frames, ${standardFrames.length} standard frames, window=${windowSize}`);

    // Pre-normalize all standard frames
    const normStandard = standardFrames.map(f => normalizePose(f.landmarks));

    for (let i = 0; i < studentFrames.length; i++) {
        const s_frame = studentFrames[i];
        let h36mStudent;

        try {
            h36mStudent = normalizePose(s_frame.landmarks);
        } catch (error) {
            evaluationReport.push({
                frame: s_frame.frame,
                phase: s_frame.phase || null,
                status: 'UNSTABLE_POSE',
                hasError: true,
                errors: [{ bone: 'SYSTEM', message: 'Không thể xác định tư thế - dữ liệu khớp không hợp lệ' }]
            });
            continue;
        }

        // Debug: in tọa độ Vai sau normalize
        if (h36mStudent[11]) {
            console.log(`[Frame ${i}] LShoulder[11] after norm: (${h36mStudent[11].x.toFixed(3)}, ${h36mStudent[11].y.toFixed(3)})`);
        }

        // Sliding window: tìm frame mẫu khớp nhất
        const matchedIdx = Math.min(i, normStandard.length - 1);
        const { bestIdx, bestAvgSim } = findBestMatch(h36mStudent, normStandard, matchedIdx, windowSize);

        const frameErrors = compareH36MFrame(h36mStudent, normStandard[bestIdx], BONE_STRUCTURES);

        evaluationReport.push({
            frame: s_frame.frame,
            phase: s_frame.phase || null,
            matchedStandardFrame: bestIdx,
            bestAvgSim: parseFloat(bestAvgSim.toFixed(4)),
            status: frameErrors.length === 0 ? 'CORRECT' : 'ERROR',
            hasError: frameErrors.length > 0,
            errors: frameErrors
        });
    }

    const totalErrors = evaluationReport.filter(r => r.hasError).length;
    console.log(`Pipeline hoàn tất: ${evaluationReport.length} frames, ${totalErrors} frames có lỗi`);
    return evaluationReport;
}

module.exports = { runPipeline };
