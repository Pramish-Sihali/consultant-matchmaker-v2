# app/utils/logger.py - Optimized Logging Configuration

import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pythonjsonlogger import jsonlogger
from app.config import settings

class CustomJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add service info
        log_record['service'] = 'consultant-matchmaker-v2'
        log_record['version'] = '2.0.0'
        log_record['environment'] = settings.environment
        
        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        
        # Add user context if available
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output in development"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)

def setup_logging():
    """Setup comprehensive logging for the application"""
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Choose formatter based on environment and configuration
    if settings.log_format.lower() == 'json':
        # JSON formatter for production
        json_formatter = CustomJSONFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        console_handler.setFormatter(json_formatter)
    else:
        # Colored text formatter for development
        if settings.is_development():
            text_formatter = ColoredFormatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            text_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        console_handler.setFormatter(text_formatter)
    
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if settings.log_file:
        try:
            file_handler = logging.FileHandler(settings.log_file)
            file_handler.setLevel(logging.INFO)
            
            # Always use JSON format for file logging
            file_json_formatter = CustomJSONFormatter(
                fmt='%(asctime)s %(name)s %(levelname)s %(message)s'
            )
            file_handler.setFormatter(file_json_formatter)
            
            root_logger.addHandler(file_handler)
            print(f"📁 Log file: {settings.log_file}")
            
        except Exception as e:
            print(f"⚠️ Could not setup file logging: {e}")
    
    # Configure external library logging levels
    external_loggers = {
        'httpx': logging.WARNING,
        'httpcore': logging.WARNING,
        'urllib3': logging.WARNING,
        'supabase': logging.WARNING,
        'asyncio': logging.WARNING,
        'uvicorn.access': logging.INFO if settings.is_development() else logging.WARNING
    }
    
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging configured: level={settings.log_level}, format={settings.log_format}")
    logger.info(f"🌐 Environment: {settings.environment}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)

class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding contextual information"""
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        # Add extra fields from context
        if 'extra' in kwargs:
            kwargs['extra'].update(self.extra)
        else:
            kwargs['extra'] = self.extra.copy()
        
        return msg, kwargs

def get_request_logger(request_id: str, user_id: Optional[str] = None) -> LoggerAdapter:
    """Get a logger with request context"""
    extra = {'request_id': request_id}
    if user_id:
        extra['user_id'] = user_id
    
    return LoggerAdapter(logging.getLogger('request'), extra)

def log_performance(
    logger: logging.Logger,
    operation: str,
    duration: float,
    metadata: Optional[Dict[str, Any]] = None
):
    """Log performance metrics"""
    perf_data = {
        'operation': operation,
        'duration_seconds': round(duration, 3),
        'performance': True
    }
    
    if metadata:
        perf_data.update(metadata)
    
    logger.info(f"⚡ {operation} completed in {duration:.3f}s", extra=perf_data)

def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None
):
    """Log error with contextual information"""
    error_data = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'error': True
    }
    
    if context:
        error_data.update(context)
    
    if user_id:
        error_data['user_id'] = user_id
    
    logger.error(f"❌ {type(error).__name__}: {str(error)}", extra=error_data, exc_info=True)

def log_ai_interaction(
    logger: logging.Logger,
    operation: str,
    model: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    duration: Optional[float] = None,
    success: bool = True
):
    """Log AI model interactions"""
    ai_data = {
        'ai_operation': operation,
        'ai_model': model,
        'ai_success': success,
        'ai_interaction': True
    }
    
    if input_tokens:
        ai_data['input_tokens'] = input_tokens
    if output_tokens:
        ai_data['output_tokens'] = output_tokens
    if duration:
        ai_data['duration_seconds'] = round(duration, 3)
    
    status = "✅" if success else "❌"
    logger.info(f"{status} AI {operation} with {model}", extra=ai_data)

def log_database_operation(
    logger: logging.Logger,
    operation: str,
    table: str,
    record_id: Optional[str] = None,
    duration: Optional[float] = None,
    success: bool = True
):
    """Log database operations"""
    db_data = {
        'db_operation': operation,
        'db_table': table,
        'db_success': success,
        'database': True
    }
    
    if record_id:
        db_data['record_id'] = record_id
    if duration:
        db_data['duration_seconds'] = round(duration, 3)
    
    status = "✅" if success else "❌"
    logger.info(f"{status} DB {operation} on {table}", extra=db_data)

def log_file_operation(
    logger: logging.Logger,
    operation: str,
    filename: str,
    file_size: Optional[int] = None,
    duration: Optional[float] = None,
    success: bool = True
):
    """Log file operations"""
    file_data = {
        'file_operation': operation,
        'filename': filename,
        'file_success': success,
        'file_processing': True
    }
    
    if file_size:
        file_data['file_size_bytes'] = file_size
    if duration:
        file_data['duration_seconds'] = round(duration, 3)
    
    status = "✅" if success else "❌"
    logger.info(f"{status} File {operation}: {filename}", extra=file_data)

def log_matching_operation(
    logger: logging.Logger,
    project_title: str,
    consultants_count: int,
    matches_found: int,
    duration: float,
    success: bool = True
):
    """Log project matching operations"""
    match_data = {
        'matching_operation': True,
        'project_title': project_title,
        'consultants_evaluated': consultants_count,
        'matches_found': matches_found,
        'duration_seconds': round(duration, 3),
        'matching_success': success
    }
    
    status = "✅" if success else "❌"
    logger.info(
        f"{status} Project matching: {matches_found} matches from {consultants_count} consultants in {duration:.2f}s",
        extra=match_data
    )

class StructuredLogger:
    """High-level structured logging interface"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str, **kwargs):
        """Log info with structured data"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning with structured data"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error with structured data"""
        if error:
            kwargs.update({
                'error_type': type(error).__name__,
                'error_message': str(error)
            })
        self.logger.error(message, extra=kwargs, exc_info=error is not None)
    
    def performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        kwargs.update({
            'performance': True,
            'operation': operation,
            'duration_seconds': round(duration, 3)
        })
        self.logger.info(f"⚡ {operation} completed in {duration:.3f}s", extra=kwargs)

# Export commonly used functions
__all__ = [
    'setup_logging',
    'get_logger',
    'get_request_logger',
    'log_performance',
    'log_error_with_context',
    'log_ai_interaction',
    'log_database_operation',
    'log_file_operation',
    'log_matching_operation',
    'StructuredLogger',
    'LoggerAdapter'
]