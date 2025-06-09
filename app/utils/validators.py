# app/utils/validators.py - Optimized Data Validation Utilities

import re
import logging
from typing import Dict, Any, List, Optional
from app.config import settings

logger = logging.getLogger(__name__)

def validate_extracted_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean extracted CV data with enhanced processing"""
    
    if not data or not isinstance(data, dict):
        raise ValueError('Invalid extracted data: must be a dictionary')
    
    logger.info(f"🔍 Validating extracted data for: {data.get('name', 'Unknown')}")
    
    validated = {
        "name": validate_name(data.get("name")),
        "email": validate_email(data.get("email")),
        "location": validate_location(data.get("location")),
        "experience_years": validate_experience_years(data.get("experience_years", {})),
        "qualifications": validate_qualifications(data.get("qualifications", {})),
        "skills": validate_skills(data.get("skills", {}))
    }
    
    # Log validation summary
    summary = _get_validation_summary(validated)
    logger.info(f"✅ Validation completed: {summary}")
    
    return validated

def validate_name(name: Any) -> str:
    """Validate and clean name field"""
    if not name or not isinstance(name, str):
        raise ValueError('Name is required and must be a string')
    
    clean_name = name.strip()
    
    if len(clean_name) < 2:
        raise ValueError('Name must be at least 2 characters long')
    
    if len(clean_name) > 100:
        logger.warning(f"Name truncated from {len(clean_name)} to 100 characters")
        clean_name = clean_name[:100]
    
    # Remove excessive whitespace and invalid characters
    sanitized_name = re.sub(r'\s+', ' ', clean_name)
    sanitized_name = re.sub(r'[^\w\s\-\.\'\u00C0-\u017F]', '', sanitized_name)
    
    return sanitized_name.strip()

def validate_email(email: Any) -> Optional[str]:
    """Validate email field with enhanced checking"""
    if not email or not isinstance(email, str):
        return None
    
    clean_email = email.strip().lower()
    
    # Enhanced email regex
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_regex, clean_email):
        logger.warning(f"Invalid email format: {email}")
        return None
    
    if len(clean_email) > 255:
        logger.warning(f"Email too long, discarded: {email}")
        return None
    
    # Check for common invalid patterns
    invalid_patterns = [
        r'\.{2,}',  # Multiple consecutive dots
        r'^\.|\.$',  # Starting or ending with dot
        r'@\.|\@$',  # Invalid @ placement
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, clean_email):
            logger.warning(f"Invalid email pattern detected: {email}")
            return None
    
    return clean_email

def validate_location(location: Any) -> Optional[str]:
    """Validate location field"""
    if not location or not isinstance(location, str):
        return None
    
    clean_location = location.strip()
    
    if len(clean_location) < 2:
        return None
    
    if len(clean_location) > 255:
        logger.warning(f"Location truncated from {len(clean_location)} to 255 characters")
        clean_location = clean_location[:255]
    
    # Remove excessive whitespace and clean up
    sanitized_location = re.sub(r'\s+', ' ', clean_location)
    sanitized_location = re.sub(r'[^\w\s\-\,\.\'\u00C0-\u017F]', '', sanitized_location)
    
    return sanitized_location.strip()

def validate_experience_years(experience: Any) -> Dict[str, Any]:
    """Validate experience years data - Enhanced version"""
    default_experience = {
        "total": 0.0,
        "total_months": None,
        "domains": {
            "software_development": 0.0,
            "research": 0.0,
            "engineering": 0.0,
            "agriculture": 0.0,
            "blockchain": 0.0
        }
    }
    
    if not experience or not isinstance(experience, dict):
        logger.info("🔧 No experience data provided, using default")
        return default_experience
    
    validated = dict(default_experience)
    
    # Validate total years - PRESERVE DECIMAL VALUES
    total = experience.get("total")
    logger.info(f"🔍 Validating total experience: {total} (type: {type(total)})")
    
    if isinstance(total, (int, float)) and total >= 0:
        validated["total"] = min(50.0, round(float(total), 1))
        logger.info(f"✅ Experience validated: {validated['total']} years")
    elif isinstance(total, str):
        try:
            parsed = float(total)
            if parsed >= 0:
                validated["total"] = min(50.0, round(parsed, 1))
                logger.info(f"✅ Experience parsed from string: {validated['total']} years")
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Could not parse experience string: {total}")
            validated["total"] = 0.0
    else:
        logger.warning(f"⚠️ Invalid experience type: {type(total)}, using 0")
        validated["total"] = 0.0
    
    # Validate total months
    total_months = experience.get("total_months")
    if isinstance(total_months, (int, float)) and total_months >= 0:
        validated["total_months"] = min(600, int(round(total_months)))
    
    # Validate domains - PRESERVE DECIMALS
    domains = experience.get("domains")
    if isinstance(domains, dict):
        validated_domains = dict(validated["domains"])  # Start with defaults
        
        for domain, years in domains.items():
            if isinstance(domain, str) and domain.strip():
                clean_domain = domain.strip().lower()[:100]
                
                # Map domain names to standard keys
                domain_mapping = {
                    "software_development": ["software", "development", "programming", "tech", "it"],
                    "research": ["research", "analysis", "academic", "policy"],
                    "engineering": ["engineering", "mechanical", "civil", "electrical"],
                    "agriculture": ["agriculture", "farming", "agtech", "supply"],
                    "blockchain": ["blockchain", "crypto", "web3", "defi", "bitcoin"]
                }
                
                # Find matching standard domain
                standard_domain = None
                for std_domain, keywords in domain_mapping.items():
                    if any(keyword in clean_domain for keyword in keywords):
                        standard_domain = std_domain
                        break
                
                if standard_domain:
                    valid_years = 0.0
                    if isinstance(years, (int, float)) and years >= 0:
                        valid_years = min(50.0, round(float(years), 1))
                    elif isinstance(years, str):
                        try:
                            parsed = float(years)
                            if parsed >= 0:
                                valid_years = min(50.0, round(parsed, 1))
                        except (ValueError, TypeError):
                            pass
                    
                    if valid_years > 0:
                        validated_domains[standard_domain] = max(
                            validated_domains[standard_domain], valid_years
                        )
        
        validated["domains"] = validated_domains
    
    logger.info(f"✅ Final validated experience: {validated}")
    return validated

def validate_qualifications(qualifications: Any) -> Dict[str, List[str]]:
    """Validate qualifications data with enhanced cleaning"""
    default_qualifications = {
        "degrees": [],
        "certifications": [],
        "licenses": []
    }
    
    if not qualifications or not isinstance(qualifications, dict):
        return default_qualifications
    
    validated = dict(default_qualifications)
    
    # Validate degrees
    degrees = qualifications.get("degrees")
    if isinstance(degrees, list):
        validated["degrees"] = validate_string_array(degrees, max_length=200, max_count=10)
    
    # Validate certifications
    certifications = qualifications.get("certifications")
    if isinstance(certifications, list):
        validated["certifications"] = validate_string_array(certifications, max_length=200, max_count=20)
    
    # Validate licenses
    licenses = qualifications.get("licenses")
    if isinstance(licenses, list):
        validated["licenses"] = validate_string_array(licenses, max_length=200, max_count=10)
    
    return validated

def validate_skills(skills: Any) -> Dict[str, List[str]]:
    """Validate skills data with enhanced categorization"""
    default_skills = {
        "technical": [],
        "domain": []
    }
    
    if not skills or not isinstance(skills, dict):
        return default_skills
    
    validated = dict(default_skills)
    
    # Validate technical skills
    technical = skills.get("technical")
    if isinstance(technical, list):
        validated["technical"] = validate_string_array(technical, max_length=100, max_count=50)
    
    # Validate domain skills
    domain = skills.get("domain")
    if isinstance(domain, list):
        validated["domain"] = validate_string_array(domain, max_length=100, max_count=50)
    
    # Auto-categorize if skills are mixed up
    all_skills = validated["technical"] + validated["domain"]
    if all_skills:
        categorized = categorize_skills(all_skills)
        validated["technical"] = categorized["technical"]
        validated["domain"] = categorized["domain"]
    
    return validated

def categorize_skills(skills: List[str]) -> Dict[str, List[str]]:
    """Automatically categorize skills into technical and domain"""
    
    technical_keywords = [
        # Programming languages
        "python", "javascript", "java", "c++", "c#", "go", "rust", "php", "ruby", "swift",
        # Frameworks/Tools
        "react", "angular", "vue", "node", "django", "flask", "spring", "docker", "kubernetes",
        # Databases
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        # Cloud/DevOps
        "aws", "azure", "gcp", "terraform", "jenkins", "git", "github", "gitlab",
        # Other technical
        "api", "rest", "graphql", "microservices", "linux", "unix", "html", "css"
    ]
    
    domain_keywords = [
        # Business/Management
        "project management", "agile", "scrum", "leadership", "strategy", "consulting",
        # Industry domains
        "finance", "healthcare", "education", "marketing", "sales", "hr", "legal",
        # Methodologies
        "machine learning", "data analysis", "research", "design thinking", "user experience"
    ]
    
    categorized = {"technical": [], "domain": []}
    
    for skill in skills:
        skill_lower = skill.lower()
        is_technical = any(keyword in skill_lower for keyword in technical_keywords)
        is_domain = any(keyword in skill_lower for keyword in domain_keywords)
        
        if is_technical and not is_domain:
            categorized["technical"].append(skill)
        elif is_domain and not is_technical:
            categorized["domain"].append(skill)
        else:
            # If ambiguous, put in domain
            categorized["domain"].append(skill)
    
    return categorized

def validate_string_array(
    array: List[Any], 
    max_length: int = 255, 
    max_count: int = 50,
    min_length: int = 2
) -> List[str]:
    """Validate an array of strings with enhanced cleaning"""
    if not isinstance(array, list):
        return []
    
    validated = []
    seen = set()  # For deduplication
    
    for i, item in enumerate(array[:max_count]):  # Limit number of items
        if isinstance(item, str):
            clean_item = item.strip()
            
            if len(clean_item) >= min_length:  # Minimum length check
                # Truncate if too long
                if len(clean_item) > max_length:
                    clean_item = clean_item[:max_length]
                
                # Clean up the string
                clean_item = re.sub(r'\s+', ' ', clean_item)  # Multiple spaces to single
                clean_item = re.sub(r'[^\w\s\-\.\+\#\(\)\/]', '', clean_item)  # Remove special chars
                
                # Deduplicate (case insensitive)
                if clean_item.lower() not in seen:
                    validated.append(clean_item)
                    seen.add(clean_item.lower())
    
    return validated

def validate_file_upload(file_size: int, filename: str, mime_type: str) -> Dict[str, Any]:
    """Validate file upload parameters with enhanced checks"""
    errors = []
    warnings = []
    
    # Check file size
    if file_size > settings.max_file_size:
        errors.append(
            f"File size ({_format_file_size(file_size)}) exceeds maximum allowed "
            f"({_format_file_size(settings.max_file_size)})"
        )
    elif file_size > settings.max_file_size * 0.8:
        warnings.append("File is large and may take longer to process")
    
    if file_size == 0:
        errors.append("File is empty")
    
    # Check file type
    if mime_type not in settings.allowed_file_types:
        errors.append(f"File type '{mime_type}' not allowed. Supported types: PDF, DOC, DOCX")
    
    # Check filename
    if not filename:
        errors.append('Filename is required')
    elif len(filename) > 255:
        errors.append('Filename is too long (max 255 characters)')
    elif not re.match(r'^[a-zA-Z0-9\-_\.\s]+$', filename):
        warnings.append('Filename contains special characters that may cause issues')
    
    # Check file extension
    if filename:
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        expected_extensions = {
            'application/pdf': 'pdf',
            'application/msword': 'doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx'
        }
        
        expected_ext = expected_extensions.get(mime_type)
        if expected_ext and extension != expected_ext:
            warnings.append(f"File extension '{extension}' doesn't match content type")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "file_info": {
            "size": file_size,
            "size_formatted": _format_file_size(file_size),
            "type": mime_type,
            "filename": filename
        }
    }

def validate_project_match_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Validate project matching request"""
    errors = []
    warnings = []
    
    if not request or not isinstance(request, dict):
        errors.append('Request must be a dictionary')
        return {"valid": False, "errors": errors}
    
    # Validate description
    description = request.get("description")
    if not description or not isinstance(description, str):
        errors.append('Description is required and must be a string')
    else:
        description = description.strip()
        if len(description) < 10:
            errors.append('Description must be at least 10 characters long')
        elif len(description) > 10000:
            errors.append('Description must be less than 10,000 characters')
        elif len(description) < 50:
            warnings.append('Short descriptions may result in less accurate matching')
    
    # Validate title (optional)
    title = request.get("title")
    if title and not isinstance(title, str):
        errors.append('Title must be a string')
    elif title and len(title) > 255:
        errors.append('Title must be less than 255 characters')
    
    # Validate matching parameters
    max_matches = request.get("max_matches", 10)
    if not isinstance(max_matches, int) or max_matches < 1 or max_matches > 100:
        errors.append('max_matches must be an integer between 1 and 100')
    
    min_score = request.get("min_score", 0.0)
    if not isinstance(min_score, (int, float)) or min_score < 0 or min_score > 100:
        errors.append('min_score must be a number between 0 and 100')
    
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    return {
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "data": {
            "description": description,
            "title": title.strip() if title else None,
            "max_matches": max_matches,
            "min_score": min_score
        }
    }

def sanitize_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize data for safe logging"""
    if not data or not isinstance(data, dict):
        return data
    
    sanitized = dict(data)
    
    # Remove or mask sensitive fields
    sensitive_fields = ["email", "password", "token", "key", "secret"]
    
    for field in sensitive_fields:
        if field in sanitized:
            if field == "email" and "@" in str(sanitized[field]):
                email = str(sanitized[field])
                local, domain = email.split("@", 1)
                sanitized[field] = f"{local[:2]}***@{domain}"
            else:
                sanitized[field] = "***"
    
    # Truncate long fields
    for field, value in sanitized.items():
        if isinstance(value, str) and len(value) > 100:
            sanitized[field] = value[:100] + "..."
    
    return sanitized

def _get_validation_summary(validated: Dict[str, Any]) -> Dict[str, Any]:
    """Get validation summary for logging"""
    return {
        "name": bool(validated.get("name")),
        "email": bool(validated.get("email")),
        "location": bool(validated.get("location")),
        "experience_total": validated.get("experience_years", {}).get("total", 0),
        "qualifications_count": _get_total_qualifications(validated.get("qualifications", {})),
        "skills_count": _get_total_skills(validated.get("skills", {}))
    }

def _get_total_qualifications(qualifications: Dict[str, List[str]]) -> int:
    """Get total number of qualifications"""
    return (
        len(qualifications.get("degrees", [])) +
        len(qualifications.get("certifications", [])) +
        len(qualifications.get("licenses", []))
    )

def _get_total_skills(skills: Dict[str, List[str]]) -> int:
    """Get total number of skills"""
    return (
        len(skills.get("technical", [])) +
        len(skills.get("domain", []))
    )

def _format_file_size(bytes_size: int) -> str:
    """Format file size in human readable format"""
    if bytes_size == 0:
        return '0 Bytes'
    
    k = 1024
    sizes = ['Bytes', 'KB', 'MB', 'GB']
    i = min(len(sizes) - 1, int((bytes_size.bit_length() - 1) // 10))
    
    return f"{bytes_size / (k ** i):.1f} {sizes[i]}"