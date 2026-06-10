import os
import uuid
import sys
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv

load_dotenv()

# Add src to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from rag_agent import run_agent
from rag_pipeline import build_rag_pipeline, get_rag_pipeline
from guardrails import run_with_guardrails


# ── Pydantic models ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is TechCorp's refund policy?",
                "customer_id": "CUST_001",
                "thread_id": "session-abc123"
            }
        }
    )
    question: str
    customer_id: str = "anonymous"
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    interaction_id: str
    thread_id: str
    customer_id: str
    blocked: bool
    block_reason: str | None
    confidence: float
    timestamp: str


class IngestRequest(BaseModel):
    content: str
    source_name: str = "uploaded_document"


class IngestResponse(BaseModel):
    status: str
    source_name: str
    message: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    components: dict


# ── Lifespan ──────────────────────────────────────────────────────
# Runs on startup and shutdown
# Replaces deprecated @app.on_event("startup")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — runs before server accepts requests
    print("Starting TechCorp AI Agent API...")
    print("Loading RAG pipeline...")
    get_rag_pipeline()
    print("Server ready to accept requests")
    yield
    # Shutdown — runs when server stops
    print("Server shutting down...")


# ── FastAPI app — ONE definition only ────────────────────────────
# lifespan defined BEFORE app
# routes defined AFTER app

app = FastAPI(
    title="TechCorp AI Research Assistant",
    description="""
    Production-grade AI agent API with:
    - RAG pipeline for internal document search
    - Web search via Tavily for current information
    - Guardrails for input/output safety
    - LangSmith observability for all interactions
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ── Routes ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Called by Kubernetes liveness and readiness probes.
    Returns 200 if healthy.
    """
    try:
        vector_store, _ = get_rag_pipeline()
        qdrant_status = "healthy"
    except Exception:
        qdrant_status = "unhealthy"

    llm_status = "healthy" if os.getenv("ANTHROPIC_API_KEY") else "unhealthy"
    tavily_status = "healthy" if os.getenv("TAVILY_API_KEY") else "unhealthy"

    overall_status = "healthy" if all([
        qdrant_status == "healthy",
        llm_status == "healthy",
        tavily_status == "healthy"
    ]) else "degraded"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        components={
            "qdrant": qdrant_status,
            "llm_api": llm_status,
            "tavily": tavily_status,
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Accepts a question and returns an AI-generated answer.
    Runs through input and output guardrails automatically.
    """
    interaction_id = str(uuid.uuid4())
    thread_id = request.thread_id or f"{request.customer_id}_{uuid.uuid4()}"

    try:
        result = run_with_guardrails(
            question=request.question,
            agent_func=run_agent,
            thread_id=thread_id
        )

        return ChatResponse(
            answer=result["answer"],
            interaction_id=interaction_id,
            thread_id=thread_id,
            customer_id=request.customer_id,
            blocked=result["blocked"],
            block_reason=result["block_reason"],
            confidence=result["confidence"],
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"Error processing request {interaction_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to process your request. Please try again."
        )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_text(request: IngestRequest):
    """
    Ingest new text content into the RAG knowledge base.
    Saves document and rebuilds the vector index.
    """
    try:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data"
        )
        file_path = os.path.join(data_dir, f"{request.source_name}.txt")

        with open(file_path, "w") as f:
            f.write(request.content)

        build_rag_pipeline(force_rebuild=True)

        return IngestResponse(
            status="success",
            source_name=request.source_name,
            message="Document ingested and indexed successfully",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"Ingest error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(e)}"
        )


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest a file upload into the RAG knowledge base.
    Accepts .txt and .pdf files.
    """
    allowed_types = ["text/plain", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not supported. Use .txt or .pdf"
        )

    try:
        content = await file.read()

        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data"
        )
        file_path = os.path.join(data_dir, file.filename)

        with open(file_path, "wb") as f:
            f.write(content)

        build_rag_pipeline(force_rebuild=True)

        return IngestResponse(
            status="success",
            source_name=file.filename,
            message=f"File {file.filename} ingested successfully",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"File ingest error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest file: {str(e)}"
        )


# ── Global error handler ──────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Catches any unhandled exception.
    Never exposes internal details to users.
    """
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred",
            "message": "Please contact support at support@techcorp.com"
        }
    )


# ── Run server ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False    # False prevents second process Qdrant conflict
    )