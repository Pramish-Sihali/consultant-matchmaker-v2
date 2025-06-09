# app/models/schemas.py - Optimized Pydantic Models and Schemas

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, EmailStr
from enum import Enum

# ==========================================
# ENUMS
# ==========================================

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class AIProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

# ==========================================
# BASE MODELS
# ==========================================

class ExperienceYears(BaseModel):
    """Experience years model with domain breakdown"""
    total: float = Field(default=0.0, ge=0, le=50, description="Total years of experience")
    total_months: Optional[int] = Field(default=None, ge=0, le=600, description="Total months of experience")
    domains: Dict[str, float] = Field(
        default_factory=lambda: {
            "software_development": 0.0,
            "research": 0.0,
            "engineering": 0.0,
            "agriculture": 0.0,
            "blockchain": 0.0
        },
        description="Experience by domain"
    )

class Qualifications(BaseModel):
    """Qualifications model"""
    degrees: List[str] = Field(default_factory=list, description="Academic degrees")
    certifications: List[str] = Field(default_factory=list, description="Professional certifications")
    licenses: List[str] = Field(default_factory=list, description="Professional licenses")

class Skills(BaseModel):
    """Skills model"""
    technical: List[str] = Field(default_factory=list, description="Technical skills")
    domain: List[str] = Field(default_factory=list, description="Domain-specific skills")

class ProcessingMetadata(BaseModel):
    """Processing metadata"""
    model: Optional[str] = None
    provider: Optional[str] = None
    processing_time: Optional[float] = None
    text_length: Optional[int] = None
    confidence_score: Optional[float] = None

class ExtractionErrors(BaseModel):
    """Error tracking for failed extractions"""
    error: Optional[str] = None
    final_error: Optional[str] = None
    retries_attempted: Optional[int] = None
    timestamp: Optional[datetime] = None
    attempt: Optional[int] = None
    traceback: Optional[str] = None

# ==========================================
# REQUEST MODELS
# ==========================================

class ConsultantCreateRequest(BaseModel):
    """Create consultant request"""
    prior_engagement: bool = Field(default=False, description="Has prior engagement with company")

class ConsultantUpdateRequest(BaseModel):
    """Update consultant request"""
    name: Optional[str] = Field(default=None, min_length=2, max_length=255)
    email: Optional[EmailStr] = None
    location: Optional[str] = Field(default=None, max_length=255)
    prior_engagement: Optional[bool] = None
    processing_status: Optional[ProcessingStatus] = None

class ProjectCreateRequest(BaseModel):
    """Create project request"""
    title: Optional[str] = Field(default=None, max_length=255, description="Project title")
    description: str = Field(..., min_length=10, max_length=10000, description="Project description")

class ProjectMatchRequest(BaseModel):
    """Project matching request"""
    description: str = Field(..., min_length=10, max_length=10000, description="Project description")
    title: Optional[str] = Field(default=None, max_length=255, description="Project title")
    max_matches: int = Field(default=10, ge=1, le=50, description="Maximum number of matches to return")
    min_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Minimum match score")

class AdminClearRequest(BaseModel):
    """Admin clear database request"""
    confirmation: str = Field(..., description="Must be 'DELETE_ALL_DATA' to confirm")
    
    @validator('confirmation')
    def validate_confirmation(cls, v):
        if v != 'DELETE_ALL_DATA':
            raise ValueError('Invalid confirmation. Operation cancelled for safety.')
        return v

# ==========================================
# RESPONSE MODELS
# ==========================================

class ConsultantResponse(BaseModel):
    """Consultant response model"""
    id: str
    name: str
    email: Optional[str] = None
    location: Optional[str] = None
    cv_file_path: Optional[str] = None
    experience_years: Optional[ExperienceYears] = None
    qualifications: Optional[Qualifications] = None
    skills: Optional[Skills] = None
    prior_engagement: bool
    processing_status: ProcessingStatus
    extraction_errors: Optional[ExtractionErrors] = None
    processing_metadata: Optional[ProcessingMetadata] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ConsultantStatusResponse(BaseModel):
    """Consultant status response"""
    success: bool
    consultant_id: str
    name: str
    processing_status: ProcessingStatus
    created_at: datetime
    updated_at: datetime
    extraction_errors: Optional[ExtractionErrors] = None
    has_data: bool
    email: Optional[str] = None
    location: Optional[str] = None
    experience_years: Optional[float] = None
    skills_count: int = 0
    qualifications_count: int = 0
    confidence_score: Optional[float] = None

class ProjectResponse(BaseModel):
    """Project response model"""
    id: str
    title: Optional[str] = None
    description: str
    requirements_extracted: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

class MatchReason(BaseModel):
    """Match reasoning model"""
    experience_match: str = Field(description="Experience matching explanation")
    qualification_match: List[str] = Field(description="Qualification matches")
    skill_match: List[str] = Field(description="Skill matches")
    prior_engagement_bonus: Optional[bool] = Field(default=None, description="Prior engagement bonus applied")
    overall_explanation: str = Field(description="Overall match explanation")
    strengths: Optional[List[str]] = Field(default_factory=list, description="Candidate strengths")
    gaps: Optional[List[str]] = Field(default_factory=list, description="Candidate gaps")
    component_scores: Optional[Dict[str, float]] = Field(default_factory=dict, description="Component scores")

class ConsultantMatch(BaseModel):
    """Consultant match model"""
    consultant: ConsultantResponse
    match_score: float = Field(ge=0, le=100, description="Match score percentage")
    match_reasons: MatchReason

class UploadResponse(BaseModel):
    """File upload response"""
    success: bool
    consultant_id: Optional[str] = None
    message: str
    file_info: Optional[Dict[str, Any]] = None

class MatchProjectResponse(BaseModel):
    """Project matching response"""
    success: bool
    matches: List[ConsultantMatch] = Field(default_factory=list)
    project_id: Optional[str] = None
    processing_time: Optional[float] = None
    total_consultants: int = 0
    message: str
    debug_info: Optional[Dict[str, Any]] = None

# ==========================================
# HEALTH & SYSTEM MODELS
# ==========================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    version: str
    environment: str
    ai_provider: str
    ai_model: str
    database_connected: bool
    ai_connected: bool
    cache_connected: bool
    response_time: float

class SystemStats(BaseModel):
    """System statistics"""
    consultants: Dict[str, int]
    storage: Dict[str, Any]
    ai_stats: Dict[str, Any]
    recent_consultants: List[Dict[str, Any]]
    system_metrics: Dict[str, Any]

class WorkerStatus(BaseModel):
    """Worker status response"""
    is_running: bool
    active_jobs: int
    max_concurrent_jobs: int
    polling_interval: int
    pending_consultants: int
    failed_consultants: int
    completed_today: int
    retry_queue_size: int

# ==========================================
# DATABASE MODELS
# ==========================================

class ConsultantDB(BaseModel):
    """Database model for consultant"""
    id: Optional[str] = None
    name: str
    email: Optional[str] = None
    location: Optional[str] = None
    cv_file_path: Optional[str] = None
    experience_years: Optional[Dict[str, Any]] = None
    qualifications: Optional[Dict[str, Any]] = None
    skills: Optional[Dict[str, Any]] = None
    prior_engagement: bool = False
    processing_status: str = "pending"
    extraction_errors: Optional[Dict[str, Any]] = None
    processing_metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ProjectDB(BaseModel):
    """Database model for project"""
    id: Optional[str] = None
    title: Optional[str] = None
    description: str
    requirements_extracted: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

# ==========================================
# VALIDATION HELPERS
# ==========================================

class ConsultantFilters(BaseModel):
    """Consultant filtering parameters"""
    status: Optional[ProcessingStatus] = None
    prior_engagement: Optional[bool] = None
    min_experience: Optional[float] = Field(default=None, ge=0, le=50)
    max_experience: Optional[float] = Field(default=None, ge=0, le=50)
    skills: Optional[List[str]] = None
    location: Optional[str] = None
    limit: Optional[int] = Field(default=100, ge=1, le=1000)
    offset: Optional[int] = Field(default=0, ge=0)

class ProjectRequirements(BaseModel):
    """Extracted project requirements"""
    experience_required: float = Field(default=3.0, ge=0, le=50, description="Required years of experience")
    qualifications_required: List[str] = Field(default_factory=list, description="Required qualifications")
    skills_required: List[str] = Field(default_factory=list, description="Required skills")
    domains: List[str] = Field(default_factory=list, description="Domain areas")
    priority_skills: List[str] = Field(default_factory=list, description="Priority skills")
    complexity_level: str = Field(default="medium", description="Project complexity")

# ==========================================
# UTILITY MODELS
# ==========================================

class APIResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int

class FileInfo(BaseModel):
    """File information model"""
    filename: str
    size: int
    content_type: str
    upload_time: datetime
    file_path: str

# ==========================================
# EXPORT
# ==========================================

__all__ = [
    # Enums
    "ProcessingStatus", "AIProvider",
    
    # Base models
    "ExperienceYears", "Qualifications", "Skills", "ProcessingMetadata", "ExtractionErrors",
    
    # Request models
    "ConsultantCreateRequest", "ConsultantUpdateRequest", "ProjectCreateRequest", 
    "ProjectMatchRequest", "AdminClearRequest",
    
    # Response models
    "ConsultantResponse", "ConsultantStatusResponse", "ProjectResponse", 
    "MatchReason", "ConsultantMatch", "UploadResponse", "MatchProjectResponse",
    
    # Health & system models
    "HealthResponse", "SystemStats", "WorkerStatus",
    
    # Database models
    "ConsultantDB", "ProjectDB",
    
    # Utility models
    "ConsultantFilters", "ProjectRequirements", "APIResponse", "PaginatedResponse", "FileInfo"
]