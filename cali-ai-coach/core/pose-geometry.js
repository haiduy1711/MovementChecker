const BONE_STRUCTURES = [
    // [Khớp mẹ, Khớp con, Tên hiển thị]
    [14, 15, "Tay trên phải (Vai -> Cùi chỏ)"],
    [15, 16, "Tay dưới phải (Cùi chỏ -> Cổ tay)"],
    [11, 12, "Tay trên trái (Vai -> Cùi chỏ)"],
    [12, 13, "Tay dưới trái (Cùi chỏ -> Cổ tay)"],
    [1, 2, "Chân trên phải (Hông -> Đầu gối)"],
    [2, 3, "Chân dưới phải (Đầu gối -> Mắt cá)"],
    [4, 5, "Chân trên trái (Hông -> Đầu gối)"],
    [5, 6, "Chân dưới trái (Đầu gối -> Mắt cá)"],
    // Thân người
    [0, 7, "Bụng (Xương chậu -> Cột sống)"],
    [7, 8, "Ngực (Cột sống -> Ngực trên)"],
    [8, 9, "Cổ dưới (Ngực trên -> Gốc cổ)"],
    [9, 10, "Cổ trên (Gốc cổ -> Đầu)"],
];

module.exports = { BONE_STRUCTURES };
