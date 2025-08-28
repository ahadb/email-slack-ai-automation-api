from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
import psycopg2
import os, json, uuid, sys, time, logging
from datetime import datetime
from typing import List
from fastapi.security import HTTPBearer
from collections import defaultdict
from contextlib import contextmanager

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

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

# --- Setup ---
app = FastAPI()

# Placeholders for LLM + chain
llm = None
chain = None

# --- Request ID middleware ---
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    logger.info(f"Request started: {request_id} - {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Request completed: {request_id} - Status: {response.status_code}")
    return response

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

# --- Database Connection Management ---
import psycopg2.pool

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
    if not db_pool:
        return False
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
    except Exception:
        return False

# --- Input Schemas ---
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

class SummarizeRequest(BaseModel):
    source: str
    messages: List[Message]

    @classmethod
    def validate_message_count(cls, v):
        if len(v) > MAX_MESSAGE_COUNT:
            raise ValueError(f"Too many messages. Max count: {MAX_MESSAGE_COUNT}")
        return v

# --- Rate limiting ---
def check_rate_limit(client_ip: str = Depends(lambda: "default")):
    current_time = time.time()
    minute_ago = current_time - 60
    rate_limit_storage[client_ip] = [
        req_time for req_time in rate_limit_storage[client_ip]
        if req_time > minute_ago
    ]
    if len(rate_limit_storage[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 60 requests per minute."
        )
    rate_limit_storage[client_ip].append(current_time)
    return True

# --- Startup event: init LLM + chain ---
@app.on_event("startup")
async def startup_event():
    global llm, chain
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ OPENAI_API_KEY is missing in environment")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        tags=["email-summarization"]
    )
    chain = LLMChain(llm=llm, prompt=prompt, tags=["email-summarization-chain"])
    logger.info("✅ ChatOpenAI + LLMChain initialized successfully")

# --- Endpoints ---
@app.post("/summarize")
async def summarize(
    req: SummarizeRequest,
    request: Request,
    rate_limit: bool = Depends(check_rate_limit)
):
    if not chain:
        raise HTTPException(status_code=500, detail="LLM chain not initialized")

    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Starting summarization for {len(req.messages)} messages from {req.source}")

    formatted = "\n".join([f"{m.sender}: {m.text}" for m in req.messages])

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
    except Exception as e:
        logger.error(f"[{request_id}] LLM error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        parsed = {"summary": result, "categories": {}}

    return {
        "summary": parsed.get("summary", ""),
        "categories": parsed.get("categories", {}),
        "request_id": request_id,
        "processing_time": llm_time
    }

@app.get("/health")
async def health():
    return {"status": "ok", "db": check_db_health()}

@app.get("/debug/env")
async def debug_env():
    val = os.getenv("OPENAI_API_KEY")
    return {"OPENAI_API_KEY": "set" if val else None}
