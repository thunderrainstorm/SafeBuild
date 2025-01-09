import sqlite3
from datetime import datetime
from contextlib import contextmanager
import logging
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = 'helmet_check.db'):
        self.db_path = db_path
        self.initialize_db()
    
    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def initialize_db(self) -> None:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Simplified table structure to match code2's output format
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS helmet_status (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        status_text TEXT NOT NULL
                    )
                ''')
                conn.commit()
                logger.info("Database initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def log_status(self, status_text: str) -> bool:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute('''
                    INSERT INTO helmet_status (timestamp, status_text)
                    VALUES (?, ?)
                ''', (timestamp, status_text))
                conn.commit()
                logger.info(f"Status logged successfully: {status_text}")
                return True
        except sqlite3.Error as e:
            logger.error(f"Failed to log status: {e}")
            return False

    def fetch_status_logs(self) -> List[Dict]:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM helmet_status")
                rows = cursor.fetchall()
                # Format output to match code2's structure
                return [{'timestamp': row[1], 'status_text': row[2]} for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch status logs: {e}")
            return []

    def clear_status_log(self) -> bool:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM helmet_status')
                conn.commit()
                logger.info("Status logs cleared successfully")
                return True
        except sqlite3.Error as e:
            logger.error(f"Failed to clear status logs: {e}")
            return False
