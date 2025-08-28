import psycopg2, os
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv("SUPABASE_DB_URL")

try:
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    cursor.execute("SELECT NOW();")
    print("Connected! Current time:", cursor.fetchone())
    conn.close()
except Exception as e:
    print("Error:", e)
