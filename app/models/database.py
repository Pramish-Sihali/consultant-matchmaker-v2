# app/models/database.py - Database Models and ORM-like Classes

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

# ==========================================
# ENUMS
# ==========================================

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ProjectStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class FileType(str, Enum):
    PDF = "application/pdf"
    DOC = "application/msword"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# ==========================================
# BASE MODEL CLASS
# ==========================================

@dataclass
class BaseModel:
    """Base model class with common functionality"""
    id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                elif isinstance(value, Enum):
                    result[key] = value.value
                elif hasattr(value, 'to_dict'):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model from dictionary"""
        if not data:
            return None
        
        # Convert datetime strings back to datetime objects
        for field_name in ['created_at', 'updated_at']:
            if field_name in data and isinstance(data[field_name], str):
                try:
                    data[field_name] = datetime.fromisoformat(data[field_name].replace('Z', '+00:00'))
                except ValueError:
                    data[field_name] = None
        
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def update_from_dict(self, data: Dict[str, Any]):
        """Update model fields from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

# ==========================================
# CONSULTANT MODELS
# ==========================================

@dataclass
class ExperienceYears:
    """Experience years model with domain breakdown"""
    total: float = 0.0
    total_months: Optional[int] = None
    domains: Dict[str, float] = field(default_factory=lambda: {
        "software_development": 0.0,
        "research": 0.0,
        "engineering": 0.0,
        "agriculture": 0.0,
        "blockchain": 0.0
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "total_months": self.total_months,
            "domains": self.domains
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not data:
            return cls()
        
        return cls(
            total=float(data.get("total", 0.0)),
            total_months=data.get("total_months"),
            domains=data.get("domains", {})
        )

@dataclass
class Qualifications:
    """Qualifications model"""
    degrees: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    licenses: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "degrees": self.degrees,
            "certifications": self.certifications,
            "licenses": self.licenses
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not data:
            return cls()
        
        return cls(
            degrees=data.get("degrees", []),
            certifications=data.get("certifications", []),
            licenses=data.get("licenses", [])
        )
    
    def get_total_count(self) -> int:
        """Get total number of qualifications"""
        return len(self.degrees) + len(self.certifications) + len(self.licenses)

@dataclass
class Skills:
    """Skills model"""
    technical: List[str] = field(default_factory=list)
    domain: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "technical": self.technical,
            "domain": self.domain
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not data:
            return cls()
        
        return cls(
            technical=data.get("technical", []),
            domain=data.get("domain", [])
        )
    
    def get_all_skills(self) -> List[str]:
        """Get all skills combined"""
        return self.technical + self.domain
    
    def get_total_count(self) -> int:
        """Get total number of skills"""
        return len(self.technical) + len(self.domain)

@dataclass
class ProcessingMetadata:
    """Processing metadata model"""
    model: Optional[str] = None
    provider: Optional[str] = None
    total_time: Optional[float] = None
    extraction_time: Optional[float] = None
    ai_time: Optional[float] = None
    validation_time: Optional[float] = None
    file_size: Optional[int] = None
    text_length: Optional[int] = None
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not data:
            return cls()
        
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

@dataclass
class ExtractionErrors:
    """Extraction errors model"""
    error: Optional[str] = None
    final_error: Optional[str] = None
    retries_attempted: Optional[int] = None
    timestamp: Optional[datetime] = None
    attempt: Optional[int] = None
    traceback: Optional[str] = None
    manual_retry: Optional[bool] = None
    retry_timestamp: Optional[str] = None
    previous_error: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not data:
            return cls()
        
        # Convert timestamp string back to datetime
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            try:
                data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                data['timestamp'] = None
        
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

@dataclass
class Consultant(BaseModel):
    """Consultant database model"""
    name: str = ""
    email: Optional[str] = None
    location: Optional[str] = None
    cv_file_path: Optional[str] = None
    experience_years: Optional[ExperienceYears] = None
    qualifications: Optional[Qualifications] = None
    skills: Optional[Skills] = None
    prior_engagement: bool = False
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    extraction_errors: Optional[ExtractionErrors] = None
    processing_metadata: Optional[ProcessingMetadata] = None
    
    def __post_init__(self):
        """Initialize nested objects if they're dictionaries"""
        if isinstance(self.experience_years, dict):
            self.experience_years = ExperienceYears.from_dict(self.experience_years)
        elif self.experience_years is None:
            self.experience_years = ExperienceYears()
        
        if isinstance(self.qualifications, dict):
            self.qualifications = Qualifications.from_dict(self.qualifications)
        elif self.qualifications is None:
            self.qualifications = Qualifications()
        
        if isinstance(self.skills, dict):
            self.skills = Skills.from_dict(self.skills)
        elif self.skills is None:
            self.skills = Skills()
        
        if isinstance(self.extraction_errors, dict):
            self.extraction_errors = ExtractionErrors.from_dict(self.extraction_errors)
        
        if isinstance(self.processing_metadata, dict):
            self.processing_metadata = ProcessingMetadata.from_dict(self.processing_metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert consultant to dictionary"""
        result = super().to_dict()
        
        # Convert nested objects
        if self.experience_years:
            result['experience_years'] = self.experience_years.to_dict()
        if self.qualifications:
            result['qualifications'] = self.qualifications.to_dict()
        if self.skills:
            result['skills'] = self.skills.to_dict()
        if self.extraction_errors:
            result['extraction_errors'] = self.extraction_errors.to_dict()
        if self.processing_metadata:
            result['processing_metadata'] = self.processing_metadata.to_dict()
        
        return result
    
    def is_completed(self) -> bool:
        """Check if consultant processing is completed"""
        return self.processing_status == ProcessingStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if consultant processing failed"""
        return self.processing_status == ProcessingStatus.FAILED
    
    def is_available_for_matching(self) -> bool:
        """Check if consultant is available for project matching"""
        return self.is_completed() and self.experience_years and self.skills
    
    def get_experience_total(self) -> float:
        """Get total years of experience"""
        return self.experience_years.total if self.experience_years else 0.0
    
    def get_skills_count(self) -> int:
        """Get total number of skills"""
        return self.skills.get_total_count() if self.skills else 0
    
    def get_qualifications_count(self) -> int:
        """Get total number of qualifications"""
        return self.qualifications.get_total_count() if self.qualifications else 0
    
    def has_skill(self, skill_name: str) -> bool:
        """Check if consultant has a specific skill"""
        if not self.skills:
            return False
        
        all_skills = [skill.lower() for skill in self.skills.get_all_skills()]
        return skill_name.lower() in all_skills
    
    def has_domain_experience(self, domain: str) -> float:
        """Get experience in specific domain"""
        if not self.experience_years or not self.experience_years.domains:
            return 0.0
        
        return self.experience_years.domains.get(domain, 0.0)

# ==========================================
# PROJECT MODELS
# ==========================================

@dataclass
class ProjectRequirements:
    """Project requirements model"""
    experience_required: float = 3.0
    qualifications_required: List[str] = field(default_factory=list)
    skills_required: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    priority_skills: List[str] = field(default_factory=list)
    complexity_level: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_required": self.experience_required,
            "qualifications_required": self.qualifications_required,
            "skills_required": self.skills_required,
            "domains": self.domains,
            "priority_skills": self.priority_skills,
            "complexity_level": self.complexity_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not data:
            return cls()
        
        return cls(
            experience_required=float(data.get("experience_required", 3.0)),
            qualifications_required=data.get("qualifications_required", []),
            skills_required=data.get("skills_required", []),
            domains=data.get("domains", []),
            priority_skills=data.get("priority_skills", []),
            complexity_level=data.get("complexity_level", "medium")
        )

@dataclass
class Project(BaseModel):
    """Project database model"""
    title: Optional[str] = None
    description: str = ""
    requirements_extracted: Optional[ProjectRequirements] = None
    status: ProjectStatus = ProjectStatus.ACTIVE
    
    def __post_init__(self):
        """Initialize nested objects if they're dictionaries"""
        if isinstance(self.requirements_extracted, dict):
            self.requirements_extracted = ProjectRequirements.from_dict(self.requirements_extracted)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert project to dictionary"""
        result = super().to_dict()
        
        if self.requirements_extracted:
            result['requirements_extracted'] = self.requirements_extracted.to_dict()
        
        return result
    
    def has_requirements(self) -> bool:
        """Check if project has extracted requirements"""
        return self.requirements_extracted is not None
    
    def get_required_skills(self) -> List[str]:
        """Get list of required skills"""
        if self.requirements_extracted:
            return self.requirements_extracted.skills_required
        return []
    
    def get_priority_skills(self) -> List[str]:
        """Get list of priority skills"""
        if self.requirements_extracted:
            return self.requirements_extracted.priority_skills
        return []

# ==========================================
# MATCHING MODELS
# ==========================================

@dataclass
class MatchScore:
    """Match score breakdown model"""
    total_score: float = 0.0
    experience_score: float = 0.0
    skills_score: float = 0.0
    qualifications_score: float = 0.0
    prior_engagement_bonus: float = 0.0
    ai_confidence_boost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": round(self.total_score, 1),
            "experience_score": round(self.experience_score, 1),
            "skills_score": round(self.skills_score, 1),
            "qualifications_score": round(self.qualifications_score, 1),
            "prior_engagement_bonus": round(self.prior_engagement_bonus, 1),
            "ai_confidence_boost": round(self.ai_confidence_boost, 1)
        }

@dataclass
class MatchReason:
    """Match reasoning model"""
    experience_assessment: str = ""
    skill_matches: List[str] = field(default_factory=list)
    qualification_matches: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_match": self.experience_assessment,
            "qualification_match": self.qualification_matches,
            "skill_match": self.skill_matches,
            "strengths": self.strengths,
            "gaps": self.gaps,
            "overall_explanation": self.recommendation
        }

@dataclass
class ConsultantMatch:
    """Consultant match result model"""
    consultant: Consultant
    match_score: MatchScore
    match_reasons: MatchReason
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "consultant": self.consultant.to_dict(),
            "match_score": self.match_score.total_score,
            "match_reasons": self.match_reasons.to_dict(),
            "component_scores": self.match_score.to_dict()
        }

# ==========================================
# FILE MODELS
# ==========================================

@dataclass
class FileInfo:
    """File information model"""
    filename: str
    file_path: str
    size: int
    content_type: str
    uploaded_at: datetime
    consultant_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "file_path": self.file_path,
            "size": self.size,
            "content_type": self.content_type,
            "uploaded_at": self.uploaded_at.isoformat(),
            "consultant_id": self.consultant_id
        }
    
    def get_file_extension(self) -> str:
        """Get file extension"""
        return self.filename.split('.')[-1].lower() if '.' in self.filename else ''
    
    def is_pdf(self) -> bool:
        """Check if file is PDF"""
        return self.content_type == FileType.PDF.value
    
    def is_word_document(self) -> bool:
        """Check if file is Word document"""
        return self.content_type in [FileType.DOC.value, FileType.DOCX.value]

# ==========================================
# STATISTICS MODELS
# ==========================================

@dataclass
class SystemStatistics:
    """System statistics model"""
    total_consultants: int = 0
    completed_consultants: int = 0
    failed_consultants: int = 0
    pending_consultants: int = 0
    processing_consultants: int = 0
    average_processing_time: float = 0.0
    success_rate: float = 0.0
    total_projects: int = 0
    storage_usage: int = 0  # in bytes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_consultants": self.total_consultants,
            "completed_consultants": self.completed_consultants,
            "failed_consultants": self.failed_consultants,
            "pending_consultants": self.pending_consultants,
            "processing_consultants": self.processing_consultants,
            "average_processing_time": round(self.average_processing_time, 2),
            "success_rate": round(self.success_rate, 1),
            "total_projects": self.total_projects,
            "storage_usage": self.storage_usage
        }

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def create_consultant_from_dict(data: Dict[str, Any]) -> Consultant:
    """Create Consultant object from dictionary"""
    return Consultant.from_dict(data)

def create_project_from_dict(data: Dict[str, Any]) -> Project:
    """Create Project object from dictionary"""
    return Project.from_dict(data)

def serialize_for_database(obj: Any) -> Any:
    """Serialize object for database storage"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_database(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize_for_database(v) for k, v in obj.items()}
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        return obj.value
    else:
        return obj

def deserialize_from_database(data: Any, model_class: type) -> Any:
    """Deserialize object from database"""
    if hasattr(model_class, 'from_dict') and isinstance(data, dict):
        return model_class.from_dict(data)
    else:
        return data

# ==========================================
# EXPORT
# ==========================================

__all__ = [
    # Enums
    'ProcessingStatus', 'ProjectStatus', 'FileType',
    
    # Base models
    'BaseModel',
    
    # Consultant models
    'ExperienceYears', 'Qualifications', 'Skills', 'ProcessingMetadata', 
    'ExtractionErrors', 'Consultant',
    
    # Project models
    'ProjectRequirements', 'Project',
    
    # Matching models
    'MatchScore', 'MatchReason', 'ConsultantMatch',
    
    # File models
    'FileInfo',
    
    # Statistics models
    'SystemStatistics',
    
    # Utility functions
    'create_consultant_from_dict', 'create_project_from_dict',
    'serialize_for_database', 'deserialize_from_database'
]