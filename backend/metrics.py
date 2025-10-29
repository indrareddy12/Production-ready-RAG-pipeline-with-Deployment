from prometheus_client import Counter, Histogram, CollectorRegistry

# Create a custom registry
REGISTRY = CollectorRegistry(auto_describe=True)

# Define metrics with custom registry
REQUESTS_TOTAL = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)

RESPONSE_TIME = Histogram(
    'api_response_time_seconds',
    'Response time in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

DOCUMENT_PROCESSING_TIME = Histogram(
    'document_processing_seconds',
    'Time spent processing documents',
    ['file_type'],
    registry=REGISTRY
)

LLM_INFERENCE_TIME = Histogram(
    'llm_inference_seconds',
    'Time spent on LLM inference',
    ['model_name'],
    registry=REGISTRY
)

EMBEDDING_GENERATION_TIME = Histogram(
    'embedding_generation_seconds',
    'Time spent generating embeddings',
    registry=REGISTRY
)