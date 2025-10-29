from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, constr
from typing import List, Optional, Dict, Any
import redis_cache
import vector_store
import document_loader
import uuid
from llm import get_llm_service
import asyncio
from datetime import datetime
import time
from functools import wraps
from metrics import REGISTRY, REQUESTS_TOTAL, RESPONSE_TIME, DOCUMENT_PROCESSING_TIME, LLM_INFERENCE_TIME, EMBEDDING_GENERATION_TIME
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from logger_config import logger
from config import Settings, get_settings
from prometheus_client import generate_latest

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app with configuration
app = FastAPI(
    title="RAG Assistant API",
    description="Production-ready RAG-based Q&A assistant API",
    version="1.0.0"
)

# Initialize Sentry for error tracking
def init_sentry(settings: Settings):
    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment,
            traces_sample_rate=0.1
        )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Sentry middleware
app.add_middleware(SentryAsgiMiddleware)

# Add rate limiting error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Request Models with validation
class QueryRequest(BaseModel):
    session_id: constr(min_length=1, max_length=100)
    query: constr(min_length=1, max_length=2000)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "query": "What are the key points in the document?"
            }
        }

class ChatHistoryRequest(BaseModel):
    session_id: constr(min_length=1, max_length=100)

# Response Models
class UploadResponse(BaseModel):
    message: str
    file_count: int
    session_id: str

class QueryResponse(BaseModel):
    response: str
    processing_time: float
    token_count: Optional[int]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

# Custom exception for file processing
class FileProcessingError(Exception):
    def __init__(self, message: str, detail: Optional[str] = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Record metrics using the new registry
    RESPONSE_TIME.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)
    
    REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Dependency for settings
async def get_app_settings():
    return get_settings()

@app.on_event("startup")
async def start_health_monitor():
    async def monitor():
        while True:
            try:
                health_status = await health_check()
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Health check failed: {e}")

    asyncio.create_task(monitor())

# File upload endpoint with improved error handling and validation
@app.post(
    "/upload/",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)

@limiter.limit("10/minute")  # Rate limiting for uploads
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: str = Form(...),
    settings: Settings = Depends(get_app_settings)
):
    """
    Upload and process documents for RAG pipeline.
    
    Args:
        files: List of files to upload (PDF, DOCX, TXT)
        session_id: Unique session identifier
        
    Returns:
        UploadResponse: Upload status and metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        logger.info(f"Starting file upload for session {session_id}")
        
        # Validate file count
        if len(files) > settings.max_files_per_upload:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {settings.max_files_per_upload} files allowed per upload"
            )
        
        extracted_text = ""
        processed_files = []
        
        # Process each file with proper error handling
        for file in files:
            try:
                processing_start = time.time()
                # Validate file size
                file_size = 0
                file_content = bytearray()
                
                # Read file in chunks to handle large files
                chunk_size = 1024 * 1024  # 1MB chunks
                while chunk := await file.read(chunk_size):
                    file_size += len(chunk)
                    if file_size > settings.max_file_size:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File {file.filename} exceeds maximum size of {settings.max_file_size} bytes"
                        )
                    file_content.extend(chunk)
                
                # Validate file extension
                file_extension = "." + file.filename.split(".")[-1].lower()
                if file_extension not in settings.allowed_extensions:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File type {file_extension} is not supported"
                    )
                
                # Extract text with timeout protection
                async with asyncio.timeout(settings.text_extraction_timeout):
                    text = document_loader.extract_text(bytes(file_content), file.filename)
                    extracted_text += text
                    processed_files.append(file.filename)
                
                DOCUMENT_PROCESSING_TIME.labels(
                    file_type=file_extension
                ).observe(time.time() - processing_start)
                
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=500,
                    detail=f"Text extraction timeout for file {file.filename}"
                )
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing file {file.filename}: {str(e)}"
                )
        
        # Store embeddings with retry logic
        try:
            embedding_start = time.time()
            async with asyncio.timeout(settings.embedding_timeout):
                vector_store.store_embeddings(session_id, extracted_text)
            EMBEDDING_GENERATION_TIME.observe(time.time() - embedding_start)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=500,
                detail="Embedding generation timeout"
            )
        
        # Save session data
        redis_cache.save_session(session_id, processed_files)
        
        logger.info(f"Successfully processed {len(processed_files)} files for session {session_id}")
        
        return UploadResponse(
            message="Files processed successfully",
            file_count=len(processed_files),
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during file processing"
        )

# Query endpoint with improved error handling and response validation
@app.post(
    "/query/",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
@limiter.limit("30/minute")  # Rate limiting for queries
async def query_llm(
    request: Request,
    query_request: QueryRequest,
    settings: Settings = Depends(get_app_settings)
):
    """
    Process user query and generate response using RAG pipeline.
    
    Args:
        query_request: Query details including session ID and query text
        
    Returns:
        QueryResponse: Generated response with metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing query for session {query_request.session_id}")
        
        # Validate session
        session_data = redis_cache.get_session(query_request.session_id)
        if not session_data:
            redis_cache.save_session(query_request.session_id, [])
        
        # Get chat history and relevant chunks with timeout protection
        async with asyncio.timeout(settings.context_retrieval_timeout):
            chat_history = redis_cache.get_chat_history(query_request.session_id)
            relevant_chunks = vector_store.retrieve_embeddings(
                query_request.session_id,
                query_request.query
            )
        
        context = relevant_chunks if relevant_chunks else None
        
        # Generate response with timeout protection
        async with asyncio.timeout(settings.llm_response_timeout):
            async with get_llm_service() as llm_service:
                llm_start = time.time()
                response = await llm_service.generate_response(
                    query=query_request.query,
                    context=context,
                    chat_history=chat_history
                )
                LLM_INFERENCE_TIME.labels(
                model_name=llm_service.config.llm_model_name
            ).observe(time.time() - llm_start)
        
        # Save chat history
        redis_cache.save_chat_history(
            query_request.session_id,
            query_request.query,
            response
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        return QueryResponse(
            response=response,
            processing_time=processing_time,
            token_count=len(response.split())  # Simple approximation
        )
        
    except asyncio.TimeoutError as e:
        logger.error(f"Timeout error during query processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Request timed out"
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query"
        )

# Chat history endpoint with improved error handling
@app.post(
    "/chat_history/",
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
@limiter.limit("60/minute")  # Rate limiting for chat history requests
async def fetch_chat_history(
    request: Request,
    history_request: ChatHistoryRequest
):
    """
    Retrieve chat history for a session.
    
    Args:
        history_request: Session ID for chat history retrieval
        
    Returns:
        Dict with chat history
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        logger.info(f"Fetching chat history for session {history_request.session_id}")
        
        chat_history = redis_cache.get_chat_history(history_request.session_id)
        
        return {
            "chat_history": chat_history if chat_history else [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving chat history"
        )

# Cleanup endpoint with improved error handling
@app.post(
    "/cleanup/",
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
@limiter.limit("10/minute")  # Rate limiting for cleanup requests
async def cleanup_session(
    request: Request,
    cleanup_request: ChatHistoryRequest
):
    """
    Clean up session data and embeddings.
    
    Args:
        cleanup_request: Session ID for cleanup
        
    Returns:
        Dict with cleanup status
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        logger.info(f"Cleaning up session {cleanup_request.session_id}")
        
        redis_cache.delete_session(cleanup_request.session_id)
        
        return {
            "message": f"Session {cleanup_request.session_id} cleaned up",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during session cleanup: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error cleaning up session"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    # Check critical service dependencies
    health_status = {
        "status": "healthy",
        "services": {
            "redis": check_redis_health(),
            "pinecone": check_pinecone_health(),
            "llm": check_llm_health()
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # If any service is unhealthy, return 503
    if not all(health_status["services"].values()):
        raise HTTPException(status_code=503, detail=health_status)
        
    return health_status

def check_redis_health():
    try:
        redis_cache.redis_client.ping()
        return True
    except:
        return False

def check_pinecone_health():
    try:
        vector_store.index.describe_index_stats()
        return True
    except:
        return False

def check_llm_health():
    try:
        "Not adding logic for llm health check as it takes time to perform this llm healtch check, might need to find a better approach"
        return True 
    except:
        return False

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics using the custom registry.
    """
    return generate_latest(REGISTRY)

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    init_sentry(settings)
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers
    )