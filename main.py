from fastapi import FastAPI, Request
from pydantic import BaseModel
import psycopg2
import os, json, uuid
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
# from langsmith import Client
# from langchain.callbacks import LangChainTracer
import logging
import sys
from datetime import datetime
from typing import List
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
import time
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# --- LangSmith Setup ---
# Temporarily disabled due to compatibility issues
langsmith_enabled = False
langsmith_client = None
print("⚠️  LangSmith temporarily disabled for compatibility")

# --- Setup ---
app = FastAPI()

# Request ID middleware - simplified for older FastAPI
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    logger.info(f"Request started: {request_id} - {request.method} {request.url}")
    
    response = await call_next(request)
    
    logger.info(f"Request completed: {request_id} - Status: {response.status_code}")
    return response

# Initialize LLM with LangSmith tracing
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0, 
    tags=["email-summarization"]
)

# --- Database Connection Management ---
import psycopg2.pool
from contextlib import contextmanager
import time

# Create connection pool (make it optional)
db_pool = None
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=os.getenv("SUPABASE_DB_URL")
    )
    print("✅ Database connection pool created successfully")
except Exception as e:
    print(f"⚠️  Database connection failed: {e}")
    db_pool = None

@contextmanager
def get_db_connection():
    """Get database connection from pool with automatic cleanup"""
    if not db_pool:
        raise Exception("Database not available")
    
    conn = None
    try:
        conn = db_pool.getconn()
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            db_pool.putconn(conn)

def check_db_health():
    """Check if database connection is healthy"""
    if not db_pool:
        return False
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
    except Exception:
        return False

# --- Input Schema ---
from typing import List
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
import time
from collections import defaultdict

# Rate limiting storage (in production, use Redis)
rate_limit_storage = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 60
MAX_MESSAGE_COUNT = 100
MAX_TEXT_LENGTH = 10000

class Message(BaseModel):
    sender: str
    text: str
    
    @classmethod
    def validate_text_length(cls, v):
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too long. Max length: {MAX_TEXT_LENGTH}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "sender": "john@example.com",
                "text": "Let's schedule a meeting for next week."
            }
        }

class SummarizeRequest(BaseModel):
    source: str
    messages: List[Message]
    
    @classmethod
    def validate_message_count(cls, v):
        if len(v) > MAX_MESSAGE_COUNT:
            raise ValueError(f"Too many messages. Max count: {MAX_MESSAGE_COUNT}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "source": "slack-channel",
                "messages": [
                    {"sender": "john@example.com", "text": "Let's schedule a meeting."},
                    {"sender": "jane@example.com", "text": "Great idea!"}
                ]
            }
        }

# Rate limiting dependency
def check_rate_limit(client_ip: str = Depends(lambda: "default")):
    """Check if client has exceeded rate limit"""
    current_time = time.time()
    minute_ago = current_time - 60
    
    # Clean old entries
    rate_limit_storage[client_ip] = [
        req_time for req_time in rate_limit_storage[client_ip] 
        if req_time > minute_ago
    ]
    
    # Check limit
    if len(rate_limit_storage[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Max 60 requests per minute."
        )
    
    # Add current request
    rate_limit_storage[client_ip].append(current_time)
    return True

# --- Prompt Template ---
prompt = PromptTemplate(
    input_variables=["messages"],
    template="""
    You are an assistant that summarizes batches of messages into clear categories.
    
    Messages:
    {messages}
    
    Summarize into:
    - A concise summary
    - Categorized JSON with keys: decisions, issues, reminders
    Return only JSON.
    """
)

# Initialize chain with LangSmith tracing
chain = LLMChain(
    llm=llm, 
    prompt=prompt,
    tags=["email-summarization-chain"]
)

# --- Endpoint ---
@app.post("/summarize")
async def summarize(
    req: SummarizeRequest, 
    request: Request,
    rate_limit: bool = Depends(check_rate_limit)
):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Starting summarization for {len(req.messages)} messages from {req.source}")
    
    # 1. Format messages for the LLM
    formatted = "\n".join([f"{m.sender}: {m.text}" for m in req.messages])
    logger.debug(f"[{request_id}] Formatted {len(formatted)} characters for LLM")

    # 2. Run summarization with LangSmith tracing
    try:
        start_time = time.time()
        result = chain.run(
            messages=formatted,
            tags=["summarize-endpoint"],
            metadata={
                "source": req.source,
                "message_count": len(req.messages),
                "endpoint": "/summarize",
                "request_id": request_id
            }
        )
        llm_time = time.time() - start_time
        logger.info(f"[{request_id}] LLM completed in {llm_time:.2f}s")
        
        # LangSmith feedback temporarily disabled
        pass
        
    except Exception as e:
        logger.error(f"[{request_id}] LLM error: {str(e)}")
        # LangSmith feedback temporarily disabled
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")

    try:
        parsed = json.loads(result)  # ensure it's valid JSON
        logger.debug(f"[{request_id}] Parsed LLM response successfully")
    except json.JSONDecodeError as e:
        logger.warning(f"[{request_id}] LLM response not valid JSON: {e}")
        parsed = {"summary": result, "categories": {}}

    # 3. Save to Supabase (messages + summary) - Optional
    message_ids = []
    if db_pool:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    for m in req.messages:
                        cursor.execute(
                            "INSERT INTO messages (source, sender, content) VALUES (%s, %s, %s) RETURNING id;",
                            (req.source, m.sender, m.text)
                        )
                        message_ids.append(cursor.fetchone()[0])

                    if message_ids:
                        cursor.execute(
                            "INSERT INTO summaries (message_ids, summary, categories) VALUES (%s::uuid[], %s, %s) RETURNING id;",
                            (message_ids, parsed.get("summary", ""), json.dumps(parsed.get("categories", {})))
                        )
                        conn.commit()
                        logger.info(f"[{request_id}] Saved {len(message_ids)} messages and summary to database")
                    else:
                        logger.warning(f"[{request_id}] No messages to save")
                        
        except Exception as db_error:
            logger.error(f"[{request_id}] Database error: {str(db_error)}")
            logger.warning(f"[{request_id}] Continuing without database save")
            message_ids = []
    else:
        logger.warning(f"[{request_id}] Database not available, skipping save")

    logger.info(f"[{request_id}] Summarization completed successfully")
    return {
        "summary": parsed.get("summary", ""),
        "categories": parsed.get("categories", {}),
        "saved": bool(message_ids),
        "request_id": request_id,
        "processing_time": llm_time
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/traces")
async def get_traces(limit: int = 10):
    """Get recent LangSmith traces for monitoring"""
    return {"error": "LangSmith temporarily disabled for compatibility"}

@app.get("/logs")
async def get_logs():
    """Get LangSmith project info and recent activity"""
    return {"error": "LangSmith temporarily disabled for compatibility"}
