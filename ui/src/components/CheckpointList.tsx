import React from 'react';

interface Checkpoint {
    id: string;
    timestamp: string;
    size: number;
    compressed_size: number;
    metadata: {
        progress?: number;
        equity?: number;
        positions?: number;
    };
}

interface CheckpointListProps {
    sessionId: string;
    onRestore?: (checkpointId: string) => void;
}

export const CheckpointList: React.FC<CheckpointListProps> = ({ sessionId, onRestore }) => {
    const [checkpoints, setCheckpoints] = React.useState<Checkpoint[]>([]);
    const [loading, setLoading] = React.useState(true);
    const [restoring, setRestoring] = React.useState<string | null>(null);

    React.useEffect(() => {
        fetchCheckpoints();
    }, [sessionId]);

    const fetchCheckpoints = async () => {
        try {
            const response = await fetch(`/api/session/${sessionId}/checkpoints`);
            if (response.ok) {
                const data = await response.json();
                setCheckpoints(data.checkpoints || []);
            }
        } catch (error) {
            console.error('Failed to fetch checkpoints:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleRestore = async (checkpointId: string) => {
        if (!confirm('Are you sure you want to restore this checkpoint? Current session will be stopped.')) {
            return;
        }

        setRestoring(checkpointId);
        try {
            const response = await fetch(`/api/session/${sessionId}/restore`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ checkpoint_id: checkpointId })
            });

            if (response.ok) {
                alert('Session restored successfully!');
                if (onRestore) {
                    onRestore(checkpointId);
                }
            } else {
                const error = await response.json();
                alert(`Failed to restore: ${error.detail || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Restore failed:', error);
            alert('Failed to restore checkpoint');
        } finally {
            setRestoring(null);
        }
    };

    const formatSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const formatTimestamp = (timestamp: string) => {
        const date = new Date(timestamp);
        return date.toLocaleString();
    };

    if (loading) {
        return (
            <div className="checkpoint-list loading">
                <div className="spinner">Loading checkpoints...</div>
            </div>
        );
    }

    if (checkpoints.length === 0) {
        return (
            <div className="checkpoint-list empty">
                <p>No checkpoints available for this session</p>
            </div>
        );
    }

    return (
        <div className="checkpoint-list">
            <div className="checkpoint-header">
                <h3>💾 Session Checkpoints</h3>
                <button onClick={fetchCheckpoints} className="refresh-btn">
                    🔄 Refresh
                </button>
            </div>

            <div className="checkpoints">
                {checkpoints.map((checkpoint) => (
                    <div key={checkpoint.id} className="checkpoint-item">
                        <div className="checkpoint-info">
                            <div className="checkpoint-time">
                                📅 {formatTimestamp(checkpoint.timestamp)}
                            </div>
                            <div className="checkpoint-details">
                                <span className="detail-item">
                                    Size: {formatSize(checkpoint.size)}
                                    {checkpoint.compressed_size && (
                                        <span className="compression">
                                            {' '}→ {formatSize(checkpoint.compressed_size)}
                                            {' '}({((1 - checkpoint.compressed_size / checkpoint.size) * 100).toFixed(0)}% saved)
                                        </span>
                                    )}
                                </span>
                                {checkpoint.metadata.progress !== undefined && (
                                    <span className="detail-item">
                                        Progress: {checkpoint.metadata.progress.toFixed(1)}%
                                    </span>
                                )}
                                {checkpoint.metadata.equity !== undefined && (
                                    <span className="detail-item">
                                        Equity: ${checkpoint.metadata.equity.toLocaleString()}
                                    </span>
                                )}
                                {checkpoint.metadata.positions !== undefined && (
                                    <span className="detail-item">
                                        Positions: {checkpoint.metadata.positions}
                                    </span>
                                )}
                            </div>
                        </div>
                        <button
                            onClick={() => handleRestore(checkpoint.id)}
                            disabled={restoring !== null}
                            className="restore-btn"
                        >
                            {restoring === checkpoint.id ? '⏳ Restoring...' : '↩️ Restore'}
                        </button>
                    </div>
                ))}
            </div>

            <style jsx>{`
        .checkpoint-list {
          background: #1a1a1a;
          border-radius: 12px;
          padding: 20px;
          color: #fff;
        }

        .checkpoint-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .checkpoint-header h3 {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
        }

        .refresh-btn {
          background: #2a2a2a;
          border: 1px solid #444;
          color: #fff;
          padding: 6px 12px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 13px;
          transition: all 0.2s;
        }

        .refresh-btn:hover {
          background: #333;
          border-color: #555;
        }

        .checkpoints {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .checkpoint-item {
          background: #2a2a2a;
          border-radius: 8px;
          padding: 16px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 16px;
          border: 1px solid #333;
          transition: all 0.2s;
        }

        .checkpoint-item:hover {
          border-color: #444;
          background: #2d2d2d;
        }

        .checkpoint-info {
          flex: 1;
        }

        .checkpoint-time {
          font-size: 14px;
          font-weight: 600;
          margin-bottom: 8px;
          color: #3498db;
        }

        .checkpoint-details {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          font-size: 12px;
          color: #888;
        }

        .detail-item {
          display: inline-block;
        }

        .compression {
          color: #2ecc71;
        }

        .restore-btn {
          background: #3498db;
          border: none;
          color: #fff;
          padding: 8px 16px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 13px;
          font-weight: 500;
          transition: all 0.2s;
          white-space: nowrap;
        }

        .restore-btn:hover:not(:disabled) {
          background: #2980b9;
          transform: translateY(-1px);
        }

        .restore-btn:disabled {
          background: #555;
          cursor: not-allowed;
          opacity: 0.6;
        }

        .loading, .empty {
          text-align: center;
          padding: 40px;
          color: #666;
        }

        .spinner {
          animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
        </div>
    );
};
