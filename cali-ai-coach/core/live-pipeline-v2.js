// core/live-pipeline-v2.js
const { compareH36MFrame } = require('./math-helpers');
const { BONE_STRUCTURES } = require('./pose-geometry');

class LivePoseEvaluator {
    constructor(kMeansKeyframes, toleranceThreshold = 0.995) {
        this.checkpoints = kMeansKeyframes; // 30 trạm mẫu lấy từ K-Means
        this.currentCheckpointIdx = -1;     // -1 nghĩa là chưa được định vị pha ban đầu
        this.tolerance = toleranceThreshold;
        this.totalReps = 0;
        this.isInitialized = false;
    }

    /**
     * Quét toàn cục qua 30 trạm xem tư thế hiện tại giống trạm nào nhất
     */
    findBestMatchingCheckpoint(liveStudentPose) {
        let bestIdx = 0;
        let minErrorsCount = Infinity;

        for (let i = 0; i < this.checkpoints.length; i++) {
            const targetPose = this.checkpoints[i];
            const errors = compareH36MFrame(liveStudentPose, targetPose, BONE_STRUCTURES, this.tolerance);

            if (errors.length < minErrorsCount) {
                minErrorsCount = errors.length;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    /**
     * Nhận vào 1 frame đơn lẻ và xử lý theo tư duy Rolling Stream
     */
    evaluateCurrentFrame(liveStudentPose) {
        // TỰ ĐỘNG ĐỊNH VỊ PHA: Nếu người tập vào tư thế giữa bài rồi mới bật web/test
        if (!this.isInitialized) {
            const matchedIdx = this.findBestMatchingCheckpoint(liveStudentPose);
            this.currentCheckpointIdx = matchedIdx;
            this.isInitialized = true;
            return {
                status: "INITIALIZED",
                matchedCheckpoint: matchedIdx,
                errors: []
            };
        }

        if (this.currentCheckpointIdx >= this.checkpoints.length) {
            this.totalReps++;
            this.currentCheckpointIdx = 0; // Vòng lặp rep mới quay về trạm 0
            return { status: "REP_COMPLETE", totalReps: this.totalReps, errors: [] };
        }

        const targetStandardPose = this.checkpoints[this.currentCheckpointIdx];
        const frameErrors = compareH36MFrame(liveStudentPose, targetStandardPose, BONE_STRUCTURES, this.tolerance);

        if (frameErrors.length === 0) {
            this.currentCheckpointIdx++; // Đạt yêu cầu hình học -> Cho phép vượt trạm
            return {
                status: "CHECKPOINT_PASSED",
                currentCheckpoint: this.currentCheckpointIdx,
                errors: []
            };
        }

        // Bị kẹt lại trạm cũ do tư thế chưa chuẩn hoặc chưa di chuyển tới pha tiếp theo
        return {
            status: "WAITING",
            currentCheckpoint: this.currentCheckpointIdx,
            errors: frameErrors
        };
    }
}

module.exports = { LivePoseEvaluator };
