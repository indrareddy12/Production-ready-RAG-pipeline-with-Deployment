import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from typing import Optional

class CustomJSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
            
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure and return a logger instance with both console and file handlers
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        max_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler with custom JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomJSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(CustomJSONFormatter())
        logger.addHandler(file_handler)
    
    return logger

# Create default logger instance
logger = setup_logger(
    'backend',
    log_file='logs/backend_logs.log'
)