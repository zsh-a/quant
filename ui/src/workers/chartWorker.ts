/**
 * Web Worker for downsampling large chart datasets
 * Uses Largest Triangle Three Buckets (LTTB) algorithm
 */

interface DataPoint {
    timestamp: string;
    value: number;
}

interface DownsampleRequest {
    type: 'downsample';
    data: DataPoint[];
    targetPoints: number;
}

interface DownsampleResponse {
    type: 'result';
    data: DataPoint[];
    originalCount: number;
    sampledCount: number;
}

/**
 * Largest Triangle Three Buckets (LTTB) downsampling algorithm
 * Preserves visual characteristics while reducing data points
 */
function downsampleLTTB(data: DataPoint[], threshold: number): DataPoint[] {
    if (data.length <= threshold) {
        return data;
    }

    const sampled: DataPoint[] = [];

    // Always include first point
    sampled.push(data[0]);

    const bucketSize = (data.length - 2) / (threshold - 2);

    let a = 0; // Initially a is the first point in the triangle

    for (let i = 0; i < threshold - 2; i++) {
        // Calculate point average for next bucket
        let avgX = 0;
        let avgY = 0;

        const avgRangeStart = Math.floor((i + 1) * bucketSize) + 1;
        const avgRangeEnd = Math.floor((i + 2) * bucketSize) + 1;
        const avgRangeLength = avgRangeEnd - avgRangeStart;

        for (let j = avgRangeStart; j < avgRangeEnd; j++) {
            avgX += new Date(data[j].timestamp).getTime();
            avgY += data[j].value;
        }
        avgX /= avgRangeLength;
        avgY /= avgRangeLength;

        // Get the range for this bucket
        const rangeOffs = Math.floor(i * bucketSize) + 1;
        const rangeTo = Math.floor((i + 1) * bucketSize) + 1;

        // Point a
        const pointAX = new Date(data[a].timestamp).getTime();
        const pointAY = data[a].value;

        let maxArea = -1;
        let maxAreaPoint = 0;

        for (let j = rangeOffs; j < rangeTo; j++) {
            const pointX = new Date(data[j].timestamp).getTime();
            const pointY = data[j].value;

            // Calculate triangle area
            const area = Math.abs(
                (pointAX - avgX) * (pointY - pointAY) -
                (pointAX - pointX) * (avgY - pointAY)
            ) * 0.5;

            if (area > maxArea) {
                maxArea = area;
                maxAreaPoint = j;
            }
        }

        sampled.push(data[maxAreaPoint]);
        a = maxAreaPoint;
    }

    // Always include last point
    sampled.push(data[data.length - 1]);

    return sampled;
}

/**
 * Simple downsampling - takes every Nth point
 * Faster but less accurate than LTTB
 */
function downsampleSimple(data: DataPoint[], targetPoints: number): DataPoint[] {
    if (data.length <= targetPoints) {
        return data;
    }

    const sampled: DataPoint[] = [];
    const step = data.length / targetPoints;

    for (let i = 0; i < targetPoints; i++) {
        const index = Math.floor(i * step);
        sampled.push(data[index]);
    }

    // Always include last point
    if (sampled[sampled.length - 1] !== data[data.length - 1]) {
        sampled.push(data[data.length - 1]);
    }

    return sampled;
}

// Listen for messages from main thread
self.onmessage = (e: MessageEvent<DownsampleRequest>) => {
    const { type, data, targetPoints } = e.data;

    if (type === 'downsample') {
        const startTime = performance.now();

        // Use LTTB for better quality, fall back to simple for very large datasets
        const sampled = data.length > 100000
            ? downsampleSimple(data, targetPoints)
            : downsampleLTTB(data, targetPoints);

        const endTime = performance.now();

        const response: DownsampleResponse = {
            type: 'result',
            data: sampled,
            originalCount: data.length,
            sampledCount: sampled.length
        };

        console.log(`Downsampled ${data.length} → ${sampled.length} points in ${(endTime - startTime).toFixed(2)}ms`);

        self.postMessage(response);
    }
};

export { };
