import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import random
import string
from dotenv import load_dotenv

load_dotenv()

# Get the database URL from your .env file
# Format should be: postgresql://user:password@localhost:5432/dbname
DB_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Creates a synchronous connection to PostgreSQL."""
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def init_db():
    """Creates the necessary tables according to SRS specifications."""
    conn = get_db_connection()
    if not conn:
        return
        
    try:
        with conn.cursor() as cur:
            # Table 1: Call Metadata
            cur.execute("""
                CREATE TABLE IF NOT EXISTS call_logs (
                    call_id VARCHAR(50) PRIMARY KEY,
                    stream_sid VARCHAR(100),
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_seconds INTEGER,
                    status VARCHAR(20)
                )
            """)
            
            # Table 2: Transcripts
            cur.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id SERIAL PRIMARY KEY,
                    call_id VARCHAR(50) REFERENCES call_logs(call_id),
                    speaker VARCHAR(20),
                    message TEXT,
                    timestamp TIMESTAMP
                )
            """)
            cur.execute("ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS call_sid VARCHAR(100);")
            cur.execute("ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS recording_url VARCHAR(255);")

        conn.commit()
        print("✅ PostgreSQL Database initialized and tables verified.")
    except Exception as e:
        print(f"Error creating tables: {e}")
    finally:
        conn.close()

def generate_call_id() -> str:
    """Generates the specific Call ID format required by BR-12 in the SRS."""
    date_str = datetime.now().strftime("%Y%md-%H%M%S")
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"CALL-{date_str}-{random_str}"

# --- UPDATE THIS FUNCTION ---
def save_call_data(stream_sid: str, call_sid: str, start_time: datetime, conversation_history: list):
    """Saves the metadata and the full transcript to Postgres when the call ends."""
    conn = get_db_connection()
    if not conn:
        return
        
    end_time = datetime.now()
    duration = int((end_time - start_time).total_seconds())
    call_id = generate_call_id()
    
    try:
        with conn.cursor() as cur:
            # 1. Save Call Metadata (Now includes call_sid)
            cur.execute("""
                INSERT INTO call_logs (call_id, stream_sid, call_sid, start_time, end_time, duration_seconds, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (call_id, stream_sid, call_sid, start_time, end_time, duration, "Completed"))
            
            
            # 2. Save the Transcript
            # We skip the very first message because it is the LLM System Prompt
            for msg in conversation_history[1:]:
                speaker = "Agent" if msg["role"] == "assistant" else "Customer"
                text = msg["content"]
                
                # If it's a RAG prompt, clean it up so we only log what the customer actually said
                if speaker == "Customer" and "Customer says:" in text:
                    text = text.split("Customer says:")[-1].strip()
                    
                cur.execute("""
                    INSERT INTO transcripts (call_id, speaker, message, timestamp)
                    VALUES (%s, %s, %s, %s)
                """, (call_id, speaker, text, datetime.now()))
                
        conn.commit()
        print(f"💾 Call {call_id} successfully saved to database!")
    except Exception as e:
        print(f"Error saving call data: {e}")
    finally:
        conn.close()

# Initialize tables when the module is imported
init_db()

# ... keep your existing db.py code ...

def get_all_calls():
    """Fetches all call logs from the database, newest first."""
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM call_logs ORDER BY start_time DESC")
            return cur.fetchall()
    except Exception as e:
        print(f"Error fetching calls: {e}")
        return []
    finally:
        conn.close()

def get_transcript(call_id: str):
    """Fetches the conversation transcript for a specific call."""
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT speaker, message, timestamp 
                FROM transcripts 
                WHERE call_id = %s 
                ORDER BY timestamp ASC
            """, (call_id,))
            return cur.fetchall()
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return []
    finally:
        conn.close()

def update_recording_url(call_sid: str, recording_url: str):
    """Saves the Twilio MP3 URL to the database once the recording finishes."""
    conn = get_db_connection()
    if not conn:
        return
        
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE call_logs SET recording_url = %s WHERE call_sid = %s", (recording_url, call_sid))
        conn.commit()
        print(f"🎵 Recording MP3 saved to database for call {call_sid}!")
    except Exception as e:
        print(f"Error updating recording URL: {e}")
    finally:
        conn.close()