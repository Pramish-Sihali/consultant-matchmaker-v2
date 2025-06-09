# app/utils/helpers.py - Utility Helper Functions

import re
import hashlib
import uuid
import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ==========================================
# STRING UTILITIES
# ==========================================

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters
    text = re.sub(r'[^\w\s\-\.\@\#\+\(\)\/]', '', text)
    
    return text

def extract_email_from_text(text: str) -> Optional[str]:
    """Extract email address from text"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, text)
    return matches[0] if matches else None

def extract_phone_from_text(text: str) -> Optional[str]:
    """Extract phone number from text"""
    phone_patterns = [
        r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
        r'\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{3}',        # International
        r'\(\d{3}\)\s?\d{3}-\d{4}',                               # (123) 456-7890
    ]
    
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
    
    return None

def normalize_skill_name(skill: str) -> str:
    """Normalize skill names for better matching"""
    if not skill:
        return ""
    
    # Common skill normalizations
    normalizations = {
        'js': 'javascript',
        'ts': 'typescript',
        'py': 'python',
        'ai': 'artificial intelligence',
        'ml': 'machine learning',
        'aws': 'amazon web services',
        'gcp': 'google cloud platform',
        'k8s': 'kubernetes',
        'docker': 'containerization',
        'react.js': 'react',
        'vue.js': 'vue',
        'node.js': 'nodejs'
    }
    
    skill_lower = skill.lower().strip()
    return normalizations.get(skill_lower, skill_lower)

def similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity score between two texts (0-1)"""
    if not text1 or not text2:
        return 0.0
    
    # Simple Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

# ==========================================
# DATE & TIME UTILITIES
# ==========================================

def parse_date_range(date_text: str) -> Optional[Dict[str, Any]]:
    """Parse date ranges from CV text"""
    if not date_text:
        return None
    
    # Common date patterns
    patterns = [
        # "Jan 2020 - Present"
        r'(\w{3})\s+(\d{4})\s*[-–]\s*(?:present|current)',
        # "2020 - 2023"
        r'(\d{4})\s*[-–]\s*(\d{4})',
        # "01/2020 - 12/2023"
        r'(\d{1,2})/(\d{4})\s*[-–]\s*(\d{1,2})/(\d{4})',
        # "January 2020 - December 2023"
        r'(\w+)\s+(\d{4})\s*[-–]\s*(\w+)\s+(\d{4})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, date_text, re.IGNORECASE)
        if match:
            groups = match.groups()
            # Process based on pattern
            # This is a simplified version - you could expand this
            return {
                "start_date": groups[0] if len(groups) > 0 else None,
                "end_date": groups[-1] if len(groups) > 1 else "present",
                "raw_text": date_text
            }
    
    return None

def calculate_months_between_dates(start_date: str, end_date: str = "present") -> int:
    """Calculate months between two dates"""
    try:
        # Simplified calculation - you could expand this
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        if "present" in end_date.lower():
            end_year = current_year
            end_month = current_month
        else:
            # Parse end date
            end_year = int(re.search(r'\d{4}', end_date).group()) if re.search(r'\d{4}', end_date) else current_year
            end_month = 12  # Default to December
        
        # Parse start date
        start_year = int(re.search(r'\d{4}', start_date).group()) if re.search(r'\d{4}', start_date) else current_year
        start_month = 1  # Default to January
        
        return max(0, (end_year - start_year) * 12 + (end_month - start_month))
        
    except Exception as e:
        logger.warning(f"Error calculating months: {e}")
        return 0

def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"

# ==========================================
# DATA PROCESSING UTILITIES
# ==========================================

def safe_get_nested(data: Dict, *keys, default=None) -> Any:
    """Safely get nested dictionary values"""
    current = data
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError, IndexError):
        return default

def merge_dicts_deep(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_deep(result[key], value)
        else:
            result[key] = value
    
    return result

def filter_dict_by_keys(data: Dict, allowed_keys: List[str]) -> Dict:
    """Filter dictionary to only include allowed keys"""
    return {k: v for k, v in data.items() if k in allowed_keys}

def flatten_dict(data: Dict, separator: str = '.') -> Dict:
    """Flatten nested dictionary"""
    def _flatten(obj, parent_key='', sep='.'):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            return {parent_key: obj}
        return dict(items)
    
    return _flatten(data, '', separator)

# ==========================================
# VALIDATION UTILITIES
# ==========================================

def is_valid_uuid(uuid_string: str) -> bool:
    """Check if string is a valid UUID"""
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False

def is_valid_email(email: str) -> bool:
    """Validate email address"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def is_valid_url(url: str) -> bool:
    """Validate URL"""
    pattern = r'^https?:\/\/(?:[-\w.])+(?:\:[0-9]+)?(?:\/(?:[\w\/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    return bool(re.match(pattern, url))

def validate_experience_years(experience: Any) -> bool:
    """Validate experience years value"""
    if isinstance(experience, (int, float)):
        return 0 <= experience <= 50
    elif isinstance(experience, str):
        try:
            exp_float = float(experience)
            return 0 <= exp_float <= 50
        except ValueError:
            return False
    return False

# ==========================================
# FILE & PATH UTILITIES
# ==========================================

def generate_unique_filename(original_filename: str, consultant_id: str) -> str:
    """Generate unique filename for uploaded files"""
    timestamp = int(time.time())
    file_extension = Path(original_filename).suffix
    safe_filename = re.sub(r'[^a-zA-Z0-9\-_.]', '_', original_filename)
    
    return f"{consultant_id}/{timestamp}_{safe_filename}"

def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return Path(filename).suffix.lower()

def get_mime_type_from_extension(extension: str) -> str:
    """Get MIME type from file extension"""
    mime_types = {
        '.pdf': 'application/pdf',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.txt': 'text/plain',
        '.csv': 'text/csv',
        '.json': 'application/json'
    }
    return mime_types.get(extension.lower(), 'application/octet-stream')

def format_file_size(bytes_size: int) -> str:
    """Format file size in human readable format"""
    if bytes_size == 0:
        return '0 B'
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    
    while bytes_size >= 1024 and i < len(units) - 1:
        bytes_size /= 1024
        i += 1
    
    return f"{bytes_size:.1f} {units[i]}"

# ==========================================
# ASYNC UTILITIES
# ==========================================

async def run_with_timeout(coro, timeout_seconds: float):
    """Run coroutine with timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

async def batch_process(items: List, batch_size: int, process_func: Callable) -> List:
    """Process items in batches"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_func(item) for item in batch],
            return_exceptions=True
        )
        results.extend(batch_results)
    
    return results

async def retry_async(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0
) -> Any:
    """Retry async function with exponential backoff"""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = delay * (backoff_factor ** attempt)
                await asyncio.sleep(wait_time)
            else:
                break
    
    raise last_exception

# ==========================================
# HASHING & ENCRYPTION UTILITIES
# ==========================================

def generate_hash(text: str, algorithm: str = 'sha256') -> str:
    """Generate hash of text"""
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()

def generate_short_id(length: int = 8) -> str:
    """Generate short random ID"""
    return str(uuid.uuid4()).replace('-', '')[:length]

def generate_cache_key(*args) -> str:
    """Generate cache key from arguments"""
    key_parts = []
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(json.dumps(arg, sort_keys=True))
        else:
            key_parts.append(str(arg))
    
    combined = ':'.join(key_parts)
    return generate_hash(combined)[:16]

# ==========================================
# PERFORMANCE UTILITIES
# ==========================================

class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"⚡ {self.name} took {format_duration(duration)}")
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

def measure_performance(func):
    """Decorator to measure function performance"""
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"⚡ {func.__name__} took {format_duration(duration)}")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {format_duration(duration)}: {e}")
            raise
    
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"⚡ {func.__name__} took {format_duration(duration)}")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {format_duration(duration)}: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# ==========================================
# JSON UTILITIES
# ==========================================

def safe_json_loads(json_string: str, default=None) -> Any:
    """Safely parse JSON string"""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, default=None) -> str:
    """Safely serialize object to JSON"""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(default) if default is not None else "{}"

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON object from text"""
    try:
        # Find JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            json_text = text[start:end + 1]
            return json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None

# ==========================================
# ENVIRONMENT UTILITIES
# ==========================================

def get_environment_info() -> Dict[str, Any]:
    """Get environment information"""
    import platform
    import sys
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release()
    }

def is_development_environment() -> bool:
    """Check if running in development environment"""
    from app.config import settings
    return settings.is_development()

# ==========================================
# DATA CONVERSION UTILITIES
# ==========================================

def convert_to_bool(value: Any, default: bool = False) -> bool:
    """Convert various types to boolean"""
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    elif isinstance(value, (int, float)):
        return value != 0
    else:
        return default

def convert_to_float(value: Any, default: float = 0.0) -> float:
    """Convert various types to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def convert_to_int(value: Any, default: int = 0) -> int:
    """Convert various types to int"""
    try:
        return int(float(value))  # Convert through float to handle strings like "5.0"
    except (ValueError, TypeError):
        return default

# ==========================================
# EXPORT ALL UTILITIES
# ==========================================

__all__ = [
    # String utilities
    'clean_text', 'extract_email_from_text', 'extract_phone_from_text', 
    'normalize_skill_name', 'similarity_score',
    
    # Date & time utilities
    'parse_date_range', 'calculate_months_between_dates', 'format_duration',
    
    # Data processing utilities
    'safe_get_nested', 'merge_dicts_deep', 'filter_dict_by_keys', 'flatten_dict',
    
    # Validation utilities
    'is_valid_uuid', 'is_valid_email', 'is_valid_url', 'validate_experience_years',
    
    # File & path utilities
    'generate_unique_filename', 'get_file_extension', 'get_mime_type_from_extension', 'format_file_size',
    
    # Async utilities
    'run_with_timeout', 'batch_process', 'retry_async',
    
    # Hashing & encryption utilities
    'generate_hash', 'generate_short_id', 'generate_cache_key',
    
    # Performance utilities
    'Timer', 'measure_performance',
    
    # JSON utilities
    'safe_json_loads', 'safe_json_dumps', 'extract_json_from_text',
    
    # Environment utilities
    'get_environment_info', 'is_development_environment',
    
    # Data conversion utilities
    'convert_to_bool', 'convert_to_float', 'convert_to_int'
]