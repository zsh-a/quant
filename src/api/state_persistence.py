"""
Session state persistence and recovery module.
Handles serialization, checkpoint saving, and state restoration.
"""

import json
import sqlite3
import gzip
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class StatePersistence:
    """Manages session state persistence and recovery"""

    def __init__(self):
        self.enabled = True
        self.checkpoint_path = Path("data/checkpoints")
        self.checkpoint_interval = 60
        self.use_compression = True

        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite database
        self.db_path = self.checkpoint_path / "checkpoints.db"
        self._init_database()

        logger.info(
            f"State persistence initialized: path={self.checkpoint_path}, "
            f"interval={self.checkpoint_interval}s, compression={self.use_compression}"
        )

    def _init_database(self):
        """Initialize SQLite database for checkpoints"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                session_id TEXT NOT NULL,
                checkpoint_time TIMESTAMP NOT NULL,
                state_data BLOB NOT NULL,
                metadata TEXT,
                PRIMARY KEY (session_id, checkpoint_time)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_time 
            ON checkpoints(session_id, checkpoint_time DESC)
        """)

        conn.commit()
        conn.close()

        logger.info(f"Checkpoint database initialized: {self.db_path}")

    def serialize_state(self, state: Dict[str, Any]) -> bytes:
        """Serialize session state to bytes"""
        try:
            # Convert to JSON
            json_str = json.dumps(state, default=str)
            json_bytes = json_str.encode("utf-8")

            # Compress if enabled
            if self.use_compression:
                return gzip.compress(json_bytes)
            else:
                return json_bytes

        except Exception as e:
            logger.error(f"Failed to serialize state: {e}")
            raise

    def deserialize_state(self, data: bytes) -> Dict[str, Any]:
        """Deserialize session state from bytes"""
        try:
            # Decompress if needed
            if self.use_compression:
                json_bytes = gzip.decompress(data)
            else:
                json_bytes = data

            # Parse JSON
            json_str = json_bytes.decode("utf-8")
            return json.loads(json_str)

        except Exception as e:
            logger.error(f"Failed to deserialize state: {e}")
            raise

    def save_checkpoint(
        self,
        session_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save a checkpoint for a session"""
        if not self.enabled:
            return False

        try:
            # Serialize state
            state_data = self.serialize_state(state)
            checkpoint_time = datetime.now()

            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO checkpoints (session_id, checkpoint_time, state_data, metadata)
                VALUES (?, ?, ?, ?)
            """,
                (
                    session_id,
                    checkpoint_time,
                    state_data,
                    json.dumps(metadata) if metadata else None,
                ),
            )

            conn.commit()
            conn.close()

            logger.info(
                f"Checkpoint saved: session={session_id}, "
                f"size={len(state_data)} bytes, time={checkpoint_time}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_latest_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint for a session"""
        if not self.enabled:
            return None

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT state_data, checkpoint_time, metadata
                FROM checkpoints
                WHERE session_id = ?
                ORDER BY checkpoint_time DESC
                LIMIT 1
            """,
                (session_id,),
            )

            row = cursor.fetchone()
            conn.close()

            if not row:
                logger.info(f"No checkpoint found for session: {session_id}")
                return None

            state_data, checkpoint_time, metadata = row

            # Deserialize state
            state = self.deserialize_state(state_data)

            logger.info(
                f"Checkpoint loaded: session={session_id}, "
                f"time={checkpoint_time}, size={len(state_data)} bytes"
            )

            return {
                "state": state,
                "checkpoint_time": checkpoint_time,
                "metadata": json.loads(metadata) if metadata else None,
            }

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def list_checkpoints(self, session_id: str) -> list:
        """List all checkpoints for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT checkpoint_time, LENGTH(state_data) as size, metadata
                FROM checkpoints
                WHERE session_id = ?
                ORDER BY checkpoint_time DESC
            """,
                (session_id,),
            )

            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    "checkpoint_time": row[0],
                    "size_bytes": row[1],
                    "metadata": json.loads(row[2]) if row[2] else None,
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    def delete_old_checkpoints(self, session_id: str, keep_count: int = 5):
        """Delete old checkpoints, keeping only the latest N"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                DELETE FROM checkpoints
                WHERE session_id = ?
                AND checkpoint_time NOT IN (
                    SELECT checkpoint_time
                    FROM checkpoints
                    WHERE session_id = ?
                    ORDER BY checkpoint_time DESC
                    LIMIT ?
                )
            """,
                (session_id, session_id, keep_count),
            )

            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            if deleted > 0:
                logger.info(
                    f"Deleted {deleted} old checkpoints for session: {session_id}"
                )

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete old checkpoints: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT session_id) as session_count,
                    COUNT(*) as checkpoint_count,
                    SUM(LENGTH(state_data)) as total_size
                FROM checkpoints
            """)

            row = cursor.fetchone()
            conn.close()

            return {
                "session_count": row[0] or 0,
                "checkpoint_count": row[1] or 0,
                "total_size_bytes": row[2] or 0,
                "total_size_mb": (row[2] or 0) / 1024 / 1024,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


# Global instance
persistence = StatePersistence()
