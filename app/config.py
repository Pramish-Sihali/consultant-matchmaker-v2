# app/config.py - Optimized Configuration Management

import os
import logging
from typing import Any, Dict, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Optimized application settings with environment variable support"""
    
    # ==========================================
    # SERVER CONFIGURATION
    # ==========================================
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="production", description="Environment (dev/staging/production)")
    
    # ==========================================
    # SUPABASE CONFIGURATION
    # ==========================================
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_service_key: str = Field(..., description="Supabase service role key")
    supabase_anon_key: Optional[str] = Field(default=None, description="Supabase anon key")
    supabase_bucket_name: str = Field(default="consultant-cvs", description="Storage bucket name")
    ai_model: str = "qwen2.5:7b"

    
    # ==========================================
    # AI MODEL CONFIGURATION (QWEN 2.5)
    # ==========================================
    ai_provider: str = Field(default="ollama", description="AI provider (ollama/openai)")
    
    # Ollama Configuration
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field(default="qwen2.5:7b", description="Qwen model to use")
    ollama_timeout: int = Field(default=120, description="Ollama timeout in seconds")
    
    # OpenAI Compatible API Configuration (alternative)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI base URL")
    openai_model: Optional[str] = Field(default="qwen2.5-7b", description="Model name for OpenAI API")
    
    
        # Enhanced AI timeout settings for Qwen 2.5
    ai_base_timeout: int = Field(default=180, description="Base timeout for AI calls (seconds)")
    ai_max_timeout: int = Field(default=600, description="Maximum timeout for AI calls (seconds)")
    ai_max_retries: int = Field(default=5, description="Maximum retries for AI calls")
    ai_health_check_timeout: int = Field(default=10, description="Timeout for AI health checks")
    
    # Worker timeout settings
    worker_ai_timeout: int = Field(default=900, description="Worker timeout for AI processing (15 minutes)")
    worker_phase_2_max_attempts: int = Field(default=3, description="Max attempts for Phase 2 before giving up")
    
    
    # ==========================================
    # FILE PROCESSING CONFIGURATION
    # ==========================================
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max file size (10MB)")
    allowed_file_types: List[str] = Field(
        default=[
            "application/pdf",
            "application/msword", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ],
        description="Allowed MIME types for CV uploads"
    )
    max_file_age_days: int = Field(default=30, description="Max age for uploaded files in days")
    
    # ==========================================
    # WORKER CONFIGURATION
    # ==========================================
    max_concurrent_jobs: int = Field(default=3, description="Max concurrent CV processing jobs")
    polling_interval: int = Field(default=5, description="Worker polling interval in seconds")
    max_retries: int = Field(default=3, description="Max retry attempts for failed processing")
    retry_delay: int = Field(default=30, description="Delay between retries in seconds")
    
    # ==========================================
    # REDIS CONFIGURATION
    # ==========================================
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL for caching and tasks")
    redis_db: int = Field(default=0, description="Redis database number")
    cache_ttl: int = Field(default=3600, description="Default cache TTL in seconds")
    
    # ==========================================
    # CORS CONFIGURATION
    # ==========================================
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:3001", 
            "http://localhost:5173",  # Vite dev server
            "https://*.vercel.app"
        ],
        description="Allowed CORS origins"
    )
    
    # ==========================================
    # LOGGING CONFIGURATION
    # ==========================================
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # ==========================================
    # PERFORMANCE CONFIGURATION
    # ==========================================
    enable_caching: bool = Field(default=True, description="Enable Redis caching")
    enable_metrics: bool = Field(default=False, description="Enable Prometheus metrics")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    
    # ==========================================
    # SECURITY CONFIGURATION
    # ==========================================
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator('ollama_model')
    def validate_ollama_model(cls, v):
        """Ensure we're using a supported Qwen model"""
        recommended_models = [
            "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b",
            "qwen2:7b", "qwen2:14b", "qwen2:32b"
        ]
        if v not in recommended_models:
            logger.warning(f"Model {v} not in recommended list: {recommended_models}")
        return v
    
    @validator('max_file_size')
    def validate_file_size(cls, v):
        """Ensure file size is reasonable"""
        if v > 50 * 1024 * 1024:  # 50MB
            raise ValueError("Max file size cannot exceed 50MB")
        return v
    
    @validator('cors_origins')
    def validate_cors_origins(cls, v):
        """Clean up CORS origins"""
        return [origin.rstrip('/') for origin in v if origin]
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get enhanced AI configuration with timeout settings"""
        config = {
            "provider": self.ai_provider,
            "model": self.ai_model,
            "base_timeout": self.ai_base_timeout,
            "max_timeout": self.ai_max_timeout,
            "max_retries": self.ai_max_retries,
            "health_check_timeout": self.ai_health_check_timeout
        }
        
        if self.ai_provider == "ollama":
            config.update({
                "url": self.ollama_url,
                "timeout": self.ai_max_timeout  # Use max timeout for Ollama
            })
        elif self.ai_provider == "openai":
            config.update({
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
                "timeout": self.ai_base_timeout  # OpenAI is typically faster
            })
        
        return config
    
    def validate_settings(self) -> bool:
        """Validate critical settings"""
        errors = []
        
        # Required settings
        if not self.supabase_url:
            errors.append("SUPABASE_URL is required")
        if not self.supabase_service_key:
            errors.append("SUPABASE_SERVICE_KEY is required")
        
        # AI provider validation
        if self.ai_provider == "ollama" and not self.ollama_url:
            errors.append("OLLAMA_URL is required when using Ollama")
        elif self.ai_provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required when using OpenAI")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True
    
    def get_database_url(self) -> str:
        """Get the complete database URL"""
        return f"{self.supabase_url}/rest/v1/"
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment.lower() in ["dev", "development", "local"]
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment.lower() == "production"

# Global settings instance
try:
    settings = Settings()
    settings.validate_settings()
    
    logger.info("✅ Configuration loaded successfully")
    logger.info(f"🌐 Environment: {settings.environment}")
    logger.info(f"🤖 AI Provider: {settings.ai_provider}")
    logger.info(f"🔗 Supabase URL: {settings.supabase_url}")
    
    if settings.ai_provider == "ollama":
        logger.info(f"🦙 Ollama URL: {settings.ollama_url}")
        logger.info(f"🧠 Model: {settings.ollama_model}")
    
except Exception as e:
    logger.error(f"❌ Configuration error: {e}")
    raise

# Export commonly used settings
__all__ = ["settings", "Settings"]