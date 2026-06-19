const fs = require('fs');
const { BONE_STRUCTURES } = require('./pose-geometry');
const { compareH36MFrame } = require('./math-helpers');
const { mediapipeToH36M } = require('./adapter');

/**
 * Chạy đường ống đối sánh dữ liệu giữa học viên và khuôn mẫu
 * @param {String} studentPath - Đường dẫn file JSON học viên
 * @param {String} standardPath - Đường dẫn file JSON mẫu chuẩn
 * @returns {Array} Mảng kết quả đánh giá chi tiết từng frame
 */
function runPipeline(studentPath, standardPath) {
    const studentData = JSON.parse(fs.readFileSync(studentPath, 'utf-8'));
    const standardData = JSON.parse(fs.readFileSync(standardPath, 'utf-8'));

    const studentFrames = studentData.frames;
    const standardFrames = standardData.frames;

    const totalFrames = Math.min(studentFrames.length, standardFrames.length);
    const evaluationReport = [];

    console.log(`Pipeline bắt đầu xử lý tích hợp: ${totalFrames} frames...`);

    for (let i = 0; i < totalFrames; i++) {
        const s_frame = studentFrames[i];
        const m_frame = standardFrames[i];

        try {
            const h36mStudent = mediapipeToH36M(s_frame.landmarks);
            const h36mStandard = mediapipeToH36M(m_frame.landmarks);

            const frameErrors = compareH36MFrame(h36mStudent, h36mStandard, BONE_STRUCTURES, 0.995);

            evaluationReport.push({
                frame: s_frame.frame,
                phase: s_frame.phase,
                hasError: frameErrors.length > 0,
                errors: frameErrors
            });

        } catch (error) {
            console.error(`Lỗi bỏ qua tai Frame ${i}:`, error.message);
            continue;
        }
    }

    console.log("Pipeline hoàn thanh! Da trich xuat xong bao cao hinh hoc.");
    return evaluationReport;
}

module.exports = { runPipeline };
