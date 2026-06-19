// test_live_with_videos.js
const fs = require('fs');
const path = require('path');

// 1. Nhập các module lõi từ hệ thống core của bạn
const { RollingPoseFilter } = require('./core/live-filter');
const { mediapipeToH36M } = require('./core/adapter');
const { LivePoseEvaluator } = require('./core/live-pipeline-v2');

// 2. Định nghĩa đường dẫn tới dữ liệu mẫu JSON đã xuất từ Colab
const KEYFRAMES_SAMPLE_PATH = path.join(__dirname, 'data', 'push_up_keyframes.json'); // 30 trạm mẫu (17 khớp)
const STUDENT_STREAM_PATH = path.join(__dirname, 'data', 'student_full_stream.json');   // Video người tập (17 khớp)

// Kiểm tra sự tồn tại của file trước khi chạy
if (!fs.existsSync(KEYFRAMES_SAMPLE_PATH) || !fs.existsSync(STUDENT_STREAM_PATH)) {
    console.error("❌ Thất bại: Thiếu file dữ liệu thử nghiệm trong thư mục data/.");
    console.error("👉 Vui lòng đảm bảo bạn đã xuất 2 file JSON 17 khớp từ Colab và lưu vào thư mục data/.");
    process.exit(1);
}

// Đọc dữ liệu JSON vào bộ nhớ
const kMeansKeyframes = JSON.parse(fs.readFileSync(KEYFRAMES_SAMPLE_PATH, 'utf-8'));
const studentVideoData = JSON.parse(fs.readFileSync(STUDENT_STREAM_PATH, 'utf-8'));

// 3. Khởi tạo Pipeline (WindowSize = 5 để lọc mịn, Tolerance = 0.995 đạt chuẩn công nghiệp)
const poseFilter = new RollingPoseFilter(5); 
const evaluator = new LivePoseEvaluator(kMeansKeyframes, 0.995);

console.log("=================================================================");
console.log(`🎬 BẮT ĐẦU MÔ PHỎNG LUỒNG LIVE STREAM 17 KHỚP XƯƠNG`);
console.log(`- Tổng số khung hình người tập: ${studentVideoData.frames.length} frames`);
console.log(`- Số lượng đoạn xương đánh giá: 12 đoạn (bao gồm Trục Thân Người)`);
console.log(`- Ngưỡng toán học Cosine Similarity: 0.995`);
console.log("=================================================================\n");

let passCheckpointCount = 0;

// 4. Quét qua chuỗi khung hình để giả lập dòng chảy trực tiếp từ Camera
studentVideoData.frames.forEach((frameObj, idx) => {
    const rawLandmarks = frameObj.landmarks;

    // Bước A: Làm mịn tọa độ theo thời gian thực (Rolling Window)
    const smoothLandmarks = poseFilter.filter(rawLandmarks);

    // Bước B: Đồng bộ hệ quy chiếu dịch tâm Pelvis (Trừ khớp 0) cho toàn bộ 17 khớp
    const h36mPose = mediapipeToH36M(smoothLandmarks);

    // Bước C: Đẩy vào Máy trạng thái Live V2 để định vị pha và chấm điểm
    const result = evaluator.evaluateCurrentFrame(h36mPose);

    // Bước D: Điều phối và ghi nhận log kiểm thử
    if (result.status === "INITIALIZED") {
        console.log(`[Frame ${idx}] 🎯 LOG KHỞI ĐỘNG: Người tập vào tư thế trước! Hệ thống tự động định vị khớp tại trạm mẫu: ${result.matchedCheckpoint}/30`);
    } 
    
    else if (result.status === "CHECKPOINT_PASSED") {
        passCheckpointCount++;
        console.log(`[Frame ${idx}] ✨ VƯỢT TRẠM ĐẠT: Khớp form xương 17 điểm! Đang chờ tiến tới trạm: ${result.currentCheckpoint}/30`);
    } 
    
    else if (result.status === "REP_COMPLETE") {
        console.log(`\n🔥 [Frame ${idx}] --------------------------------------------------`);
        console.log(`🎉 KÍCH HOẠT THÀNH CÔNG: Đã hoàn thành trọn vẹn LƯỢT TẬP (REP) thứ: ${result.totalReps}`);
        console.log(`-------------------------------------------------------------------\n`);
    }
    
    // Log báo lỗi định kỳ sau mỗi 30 frames để không làm ngập màn hình Terminal
    else if (result.status === "WAITING" && idx % 30 === 0) {
        if (result.errors.length > 0) {
            console.log(`[Frame ${idx}] ⏳ Đang giữ vị trí ở trạm [${result.currentCheckpoint}]. Chi tiết lệch form:`);
            result.errors.forEach(err => {
                console.log(`   └─ Xương [${err.bone}]: cosSim=${err.cosSim.toFixed(5)} (Yêu cầu >= 0.995)`);
            });
        }
    }
});

console.log("\n=================================================================");
console.log("📊 TỔNG KẾT ĐÁNH GIÁ PIPELINE KIỂM THỬ MẪU:");
console.log(`- Tổng frame đã nạp xử lý: ${studentVideoData.frames.length}`);
console.log(`- Tổng số lần chuyển pha thành công: ${passCheckpointCount}`);
console.log(`- Tổng số lượt tập (Reps) ghi nhận: ${evaluator.totalReps}`);
console.log