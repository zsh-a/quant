// Largest-Triangle-Three-Buckets (LTTB) downsampling algorithm
// Adapted for TypeScript

export function lttb(data: any[], threshold: number, valueKey: string = 'value'): any[] {
    const dataLength = data.length;
    if (threshold >= dataLength || threshold === 0) {
        return data; // Nothing to do
    }

    const sampled = [];
    let sampledIndex = 0;

    // Bucket size. Leave room for start and end data points
    const every = (dataLength - 2) / (threshold - 2);

    let a = 0;
    let maxAreaPoint: any, maxArea: number, area: number;

    sampled[sampledIndex++] = data[a];

    for (let i = 0; i < threshold - 2; i++) {
        // Calculate point average for next bucket (containing c)
        let avgX = 0;
        let avgY = 0;
        let avgRangeStart = Math.floor((i + 1) * every) + 1;
        let avgRangeEnd = Math.floor((i + 2) * every) + 1;
        avgRangeEnd = avgRangeEnd < dataLength ? avgRangeEnd : dataLength;

        const avgRangeLength = avgRangeEnd - avgRangeStart;

        for (; avgRangeStart < avgRangeEnd; avgRangeStart++) {
            avgX += avgRangeStart; // Using index as X for simplicity in generic data
            avgY += data[avgRangeStart][valueKey] || 0;
        }

        avgX /= avgRangeLength;
        avgY /= avgRangeLength;

        // Get the range for this bucket
        let rangeOffs = Math.floor((i + 0) * every) + 1;
        let rangeTo = Math.floor((i + 1) * every) + 1;

        // Point a
        const pointAX = a; // using index
        const pointAY = data[a][valueKey] || 0;

        maxArea = -1;
        let nextA = a;

        for (; rangeOffs < rangeTo; rangeOffs++) {
            // Calculate triangle area over three buckets
            area = Math.abs((pointAX - avgX) * ((data[rangeOffs][valueKey] || 0) - pointAY) -
                            (pointAX - rangeOffs) * (pointAY - avgY)) * 0.5;
            if (area > maxArea) {
                maxArea = area;
                maxAreaPoint = data[rangeOffs];
                nextA = rangeOffs; // Next a is this b
            }
        }

        sampled[sampledIndex++] = maxAreaPoint!;
        a = nextA; // This a is the next a (chosen b)
    }

    sampled[sampledIndex++] = data[dataLength - 1]; // Always add last

    return sampled;
}
