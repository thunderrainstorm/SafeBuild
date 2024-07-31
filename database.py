import sqlite3
from datetime import datetime

def initialize_db():
    conn = sqlite3.connect('helmet_check.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS helmet_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    status_text TEXT
                )''')
    conn.commit()
    conn.close()

def log_status(status_text):
    conn = sqlite3.connect('helmet_check.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('INSERT INTO helmet_status (timestamp, status_text) VALUES (?, ?)', (timestamp, status_text))
    conn.commit()
    conn.close()

def clear_status_log():
    conn = sqlite3.connect('helmet_check.db')
    c = conn.cursor()
    c.execute('DELETE FROM helmet_status')
    conn.commit()
    conn.close()

def fetch_status_logs():
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('helmet_check.db')  # Use the correct database file name
        cursor = conn.cursor()
        
        # Fetch all records from the helmet_status table
        cursor.execute("SELECT * FROM helmet_status")  # Use the correct table name
        rows = cursor.fetchall()
        
        # Close the connection
        conn.close()
        
        # Format the logs for JSON response
        return [{'timestamp': row[1], 'status_text': row[2]} for row in rows]
    except sqlite3.Error as e:
        print(f"Error fetching data from SQLite: {e}")
        return []
