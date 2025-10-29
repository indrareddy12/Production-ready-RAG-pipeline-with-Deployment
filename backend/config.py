from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from typing import Set, Dict, Any, Optional
from functools import lru_cache
import os
from pathlib import Path
import yaml
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class Settings(BaseSettings):
    """
    Configuration settings for the RAG application.
    Handles all configuration parameters with validation and documentation.
    """

    # Application Settings
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    app_name: str = Field(
        default="rag-assistant",
        description="Application name"
    )
    api_version: str = Field(
        default="v1",
        description="API version"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode flag"
    )

    # Server Settings
    host: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        description="Server port"
    )
    workers: int = Field(
        default=4,
        description="Number of worker processes"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    # Security Settings
    secret_key: str = Field(
        default="your-super-secret-key",
        description="Secret key for security"
    )
    allowed_origins: list = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="API key header name"
    )

    # Redis Settings
    redis_host: str = Field(
        default="redis",
        description="Redis host"
    )
    redis_port: int = Field(
        default=6379,
        description="Redis port"
    )
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password"
    )
    redis_db: int = Field(
        default=0,
        description="Redis database number"
    )
    session_ttl: int = Field(
        default=1800,
        description="Session TTL in seconds"
    )

    # Vector Store Settings
    vector_store_url: str = Field(
        default="http://localhost:8080",
        description="Vector store service URL"
    )
    pinecone_api_key: str = Field(
        default="your-pinecone-api-key",
        description="Pinecone API key"
    )
    pinecone_environment: str = Field(
        default="your-pinecone-environment",
        description="Pinecone environment"
    )
    pinecone_index_name: str = Field(
        default="rag-index",
        description="Pinecone index name"
    )
    embedding_dimension: int = Field(
        default=1024,
        description="Embedding dimension"
    )

    # LLM Settings (Updated with Missing Variables)
    llm_model_name: str = Field(
        default="deepseek-r1:1.5b",
        description="LLM model name"
    )
    llm_history_limit: int = Field(
        default=10,
        description="Limit for LLM conversation history"
    )
    llm_max_context_length: int = Field(
        default=97304,
        description="Maximum context length for LLM"
    )
    default_llm_context: str = Field(
        default="",
        description="Default system context for LLM interactions"
    )
    llm_system_prompt_tokens: int = Field(
        default=1000,
        description="Number of tokens allocated for system prompts"
    )
    llm_max_history_tokens: int = Field(
        default=16384,
        description="Maximum number of tokens for conversation history"
    )
    llm_max_response_tokens: int = Field(
        default=8192,
        description="Maximum number of tokens for LLM response"
    )
    llm_temperature: float = Field(
        default=0.6,
        description="LLM temperature parameter"
    )
    llm_retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for LLM API calls"
    )
    min_retry_wait: int = Field(
        default=4,
        description="Minimum wait time before retrying LLM calls"
    )
    max_retry_wait: int = Field(
        default=10,
        description="Maximum wait time before retrying LLM calls"
    )
    llm_concurrent_requests: int = Field(
        default=100,
        description="Maximum number of concurrent requests allowed for LLM"
    )

    # File Processing Settings
    allowed_extensions: Set[str] = Field(
        default={".pdf", ".docx", ".txt"},
        description="Allowed file extensions"
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum file size in bytes"
    )
    max_files_per_upload: int = Field(
        default=5,
        description="Maximum number of files per upload"
    )
    chunk_size: int = Field(
        default=512,
        description="Text chunk size for embeddings"
    )
    chunk_overlap: int = Field(
        default=100,
        description="Chunk overlap size"
    )

    # Timeout Settings
    text_extraction_timeout: int = Field(
        default=60,
        description="Timeout for text extraction in seconds"
    )
    embedding_timeout: int = Field(
        default=60,
        description="Timeout for embedding generation in seconds"
    )
    llm_response_timeout: int = Field(
        default=1000,
        description="Timeout for LLM response in seconds"
    )
    context_retrieval_timeout: int = Field(
        default=30,
        description="Timeout for context retrieval in seconds"
    )

    # Rate Limiting Settings
    rate_limit_uploads: str = Field(
        default="10/minute",
        description="Rate limit for file uploads"
    )
    rate_limit_queries: str = Field(
        default="30/minute",
        description="Rate limit for queries"
    )

    # Monitoring Settings
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    metrics_port: int = Field(
        default=9090,
        description="Prometheus metrics port"
    )

    # Model config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # Validators
    @validator("llm_temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

    @validator("workers")
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError("Number of workers must be at least 1")
        return v

    @validator("chunk_size")
    def validate_chunk_size(cls, v):
        if v < 1:
            raise ValueError("Chunk size must be positive")
        return v

    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v, values):
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v

class ConfigLoader:
    """
    Handles loading configuration from multiple sources
    with priority: environment variables > config file > defaults
    """
    def __init__(self):
        default_path = Path(__file__).parent / "config.yaml"
        self.config_path = Path(os.getenv("CONFIG_PATH", default_path)).resolve()
        self._config_data: Dict[str, Any] = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file if it exists"""
        yaml_config = {}
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
        return yaml_config

    def get_settings_data(self) -> Dict[str, Any]:
        """
        Get merged configuration from all sources.
        Priority: env vars > config file > defaults
        """
        # Get YAML config
        yaml_config = self._load_config()
        
        # Get environment variables that match Settings fields
        env_vars = {
            key.lower(): value 
            for key, value in os.environ.items() 
            if key.lower() in Settings.model_fields
        }
        
        # Merge with correct priority: yaml first, then env vars override
        merged_config = {
            **yaml_config,  # Lower priority
            **env_vars      # Higher priority
        }
        
        return merged_config

@lru_cache
def get_settings() -> Settings:
    """
    Get settings instance.
    Reads fresh configuration each time.
    """
    config_loader = ConfigLoader()
    config_data = config_loader.get_settings_data()
    return Settings(**config_data)

def get_environment_settings() -> Dict[str, Any]:
    """
    Get environment-specific settings.
    Useful for configuration that varies by environment.
    """
    settings = get_settings()
    env_settings = {
        Environment.DEVELOPMENT: {
            "debug": True,
            "log_level": "DEBUG",
            "allowed_origins": ["*"],
        },
        Environment.STAGING: {
            "debug": False,
            "log_level": "INFO",
            "allowed_origins": ["https://staging.yourdomain.com"],
        },
        Environment.PRODUCTION: {
            "debug": False,
            "log_level": "WARNING",
            "allowed_origins": ["https://yourdomain.com"],
            "workers": 8,
            "rate_limit_uploads": "5/minute",
            "rate_limit_queries": "20/minute",
        }
    }
    return env_settings.get(settings.environment, {})

# Example usage:
if __name__ == "__main__":
    settings = get_settings()
    print(f"Current environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")
    print(f"Redis host: {settings.redis_host}")