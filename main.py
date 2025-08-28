from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
import os, json, uuid, sys, time, logging
from datetime import datetime
from typing import List
from fastapi.security import HTTPBearer
from collections import defaultdict

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from supabase import create_client, Client   # ✅ NEW

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

# Placeholders
llm = None
chain = None
supabase: Client | None = None

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

# --- Startup event: init LLM + Supabase client ---
@app.on_event("startup")
async def startup_event():
    global llm, chain, supabase

    # OpenAI
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

    # Supabase
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        logger.warning("⚠️ SUPABASE_URL or SUPABASE_ANON_KEY not set")
        supabase = None
    else:
        supabase = create_client(url, key)
        logger.info("✅ Supabase client initialized")

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

    # --- Save to Supabase ---
    saved = False
    message_ids = []
    if supabase:
        try:
            # Insert messages
            for m in req.messages:
                res = supabase.table("messages").insert({
                    "source": req.source,
                    "sender": m.sender,
                    "content": m.text
                }).execute()
                if res.data:
                    message_ids.append(res.data[0]["id"])

            # Insert summary
            if message_ids:
                supabase.table("summaries").insert({
                    "message_ids": message_ids,
                    "summary": parsed.get("summary", ""),
                    "categories": parsed.get("categories", {})
                }).execute()
                saved = True
                logger.info(f"[{request_id}] Saved {len(message_ids)} messages + summary to Supabase")
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to save to Supabase: {e}")

    return {
        "summary": parsed.get("summary", ""),
        "categories": parsed.get("categories", {}),
        "saved": saved,
        "request_id": request_id,
        "processing_time": llm_time
    }

@app.get("/health")
async def health():
    if supabase:
        try:
            supabase.table("messages").select("id").limit(1).execute()
            return {"status": "ok", "db": True}
        except Exception as e:
            return {"status": "ok", "db": False, "error": str(e)}
    return {"status": "ok", "db": False}

@app.get("/debug/env")
async def debug_env():
    val = os.getenv("OPENAI_API_KEY")
    return {"OPENAI_API_KEY": "set" if val else None}

@app.get("/")
async def root():
    return {"status": "alive"}
