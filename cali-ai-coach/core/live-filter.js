class RollingPoseFilter {
    constructor(windowSize = 5) {
        this.windowSize = windowSize;
        this.buffer = [];
    }
    filter(rawLandmarks) {
        this.buffer.push(rawLandmarks);
        if (this.buffer.length > this.windowSize) this.buffer.shift();
        if (this.buffer.length < 2) return rawLandmarks;

        const numJoints = rawLandmarks.length;
        const smoothed = new Array(numJoints);
        for (let j = 0; j < numJoints; j++) {
            let sumX = 0, sumY = 0;
            for (let f = 0; f < this.buffer.length; f++) {
                sumX += this.buffer[f][j].x;
                sumY += this.buffer[f][j].y;
            }
            smoothed[j] = { x: sumX / this.buffer.length, y: sumY / this.buffer.length, z: 0 };
        }
        return smoothed;
    }
}
module.exports = { RollingPoseFilter };
