import { useEffect, useState, useRef } from 'react';

interface DataPoint {
    timestamp: string;
    value: number;
}

interface UseChartDataProps {
    data: DataPoint[];
    maxPoints?: number;
}

interface UseChartDataResult {
    sampledData: DataPoint[];
    isProcessing: boolean;
    compressionRatio: number;
}

/**
 * Hook for downsampling large chart datasets using Web Worker
 */
export const useChartData = ({
    data,
    maxPoints = 1000
}: UseChartDataProps): UseChartDataResult => {
    const [sampledData, setSampledData] = useState<DataPoint[]>(data);
    const [isProcessing, setIsProcessing] = useState(false);
    const [compressionRatio, setCompressionRatio] = useState(1);
    const workerRef = useRef<Worker | null>(null);

    useEffect(() => {
        // Initialize worker
        if (!workerRef.current && typeof Worker !== 'undefined') {
            try {
                workerRef.current = new Worker(
                    new URL('../workers/chartWorker.ts', import.meta.url),
                    { type: 'module' }
                );

                workerRef.current.onmessage = (e) => {
                    const { data: result, originalCount, sampledCount } = e.data;
                    setSampledData(result);
                    setCompressionRatio(originalCount / sampledCount);
                    setIsProcessing(false);
                };

                workerRef.current.onerror = (error) => {
                    console.error('Chart worker error:', error);
                    // Fallback to original data
                    setSampledData(data);
                    setIsProcessing(false);
                };
            } catch (error) {
                console.warn('Web Worker not supported, using original data');
                setSampledData(data);
            }
        }

        return () => {
            if (workerRef.current) {
                workerRef.current.terminate();
                workerRef.current = null;
            }
        };
    }, []);

    useEffect(() => {
        if (!data || data.length === 0) {
            setSampledData([]);
            return;
        }

        // If data is small enough, use it directly
        if (data.length <= maxPoints) {
            setSampledData(data);
            setCompressionRatio(1);
            return;
        }

        // Use worker for downsampling
        if (workerRef.current) {
            setIsProcessing(true);
            workerRef.current.postMessage({
                type: 'downsample',
                data,
                targetPoints: maxPoints
            });
        } else {
            // Fallback: simple sampling without worker
            const step = Math.ceil(data.length / maxPoints);
            const sampled = data.filter((_, index) => index % step === 0);
            setSampledData(sampled);
            setCompressionRatio(data.length / sampled.length);
        }
    }, [data, maxPoints]);

    return {
        sampledData,
        isProcessing,
        compressionRatio
    };
};
