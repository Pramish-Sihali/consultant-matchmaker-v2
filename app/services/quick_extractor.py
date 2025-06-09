# app/services/quick_extractor.py - Enhanced Fast Rule-Based Basic Info Extraction

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class QuickExtractorService:
    """Fast rule-based extraction for immediate basic info"""
    
    def __init__(self):
        # Common patterns for quick extraction
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_patterns = [
            re.compile(r'\+?[\d\s\-\(\)]{10,15}'),  # International format
            re.compile(r'\b\d{10}\b'),               # 10 digit numbers
            re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b')  # US format
        ]
        self.date_patterns = [
            re.compile(r'\b(\d{4})\s*[-–—]\s*(\d{4}|\bpresent\b|\bcurrent\b)', re.IGNORECASE),
            re.compile(r'\b(\w{3,9})\s+(\d{4})\s*[-–—]\s*(\w{3,9}\s+\d{4}|\bpresent\b|\bcurrent\b)', re.IGNORECASE),
            re.compile(r'\b(\d{1,2})/(\d{4})\s*[-–—]\s*(\d{1,2}/\d{4}|\bpresent\b|\bcurrent\b)', re.IGNORECASE)
        ]
        
        # Common job titles for quick identification
        self.job_title_keywords = [
            'engineer', 'developer', 'manager', 'analyst', 'consultant', 'specialist',
            'lead', 'senior', 'junior', 'intern', 'director', 'coordinator', 
            'administrator', 'designer', 'architect', 'scientist', 'researcher',
            'programmer', 'technician', 'supervisor', 'associate', 'assistant'
        ]
        
        # Location keywords/patterns
        self.location_patterns = [
            re.compile(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b'),  # City, Country
            re.compile(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b'),      # City, State
            re.compile(r'\b\d{5}(?:-\d{4})?\b')              # ZIP codes
        ]
        
        # Common skills for quick extraction
        self.common_skills = [
            # Programming languages
            'python', 'javascript', 'java', 'c++', 'c#', 'typescript', 'react', 'angular',
            'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'laravel',
            # Technologies
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'linux', 'sql',
            'mongodb', 'postgresql', 'mysql', 'redis', 'html', 'css', 'sass',
            # Frameworks & Tools
            'next.js', 'nuxt.js', 'webpack', 'babel', 'jest', 'cypress', 'figma',
            'photoshop', 'illustrator', 'sketch', 'tensorflow', 'pytorch'
        ]
    
    async def extract_basic_info(
        self, 
        extracted_text: str, 
        filename: str = None
    ) -> Dict[str, Any]:
        """
        Quick extraction of basic information from CV text
        Target: Complete in 2-5 seconds
        """
        
        start_time = datetime.now()
        
        try:
            logger.info(f"🚀 Starting quick extraction on {len(extracted_text)} characters")
            
            # Clean text for better processing
            clean_text = self._clean_text(extracted_text)
            lines = clean_text.split('\n')
            
            # Extract basic information
            basic_info = {
                'name': self._extract_name(lines),
                'email': self._extract_email(clean_text),
                'phone': self._extract_phone(clean_text),
                'location': self._extract_location(clean_text),
                'basic_experience': self._extract_basic_experience(clean_text, lines),
                'quick_skills': self._extract_quick_skills(clean_text),
                'education_hints': self._extract_education_hints(clean_text),
                'extraction_metadata': {
                    'extraction_time': (datetime.now() - start_time).total_seconds(),
                    'text_length': len(extracted_text),
                    'lines_processed': len(lines),
                    'filename': filename,
                    'extraction_method': 'rule_based_quick'
                }
            }
            
            # Calculate confidence score
            basic_info['extraction_confidence'] = self._calculate_confidence(basic_info)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Quick extraction completed in {processing_time:.2f}s")
            logger.info(f"📋 Extracted: {basic_info['name']} | {basic_info['email']} | {basic_info['basic_experience']['total_years']} years exp")
            
            return basic_info
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Quick extraction failed after {processing_time:.2f}s: {e}")
            return self._create_fallback_basic_info(filename)
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better processing"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\n\-\.\@\#\+\(\)\/\,\:]', ' ', text)
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        return text.strip()
    
    def _extract_name(self, lines: List[str]) -> str:
        """Enhanced name extraction from first few lines with better patterns"""
        try:
            # Common name prefixes to skip
            skip_patterns = [
                'resume', 'cv', 'curriculum vitae', 'curriculum', 'bio', 'biography',
                'profile', 'contact', 'page', 'tel:', 'email:', 'phone:', 'mobile:',
                'address:', 'location:', 'linkedin:', 'github:', 'website:', 'portfolio:',
                'objective', 'summary', 'experience', 'education', 'skills',
                'personal information', 'personal details', 'contact information'
            ]
            
            logger.info(f"🔍 Searching for name in {len(lines)} lines")
            
            # Check first 10 lines for names
            for i, line in enumerate(lines[:10]):
                line = line.strip()
                
                # Skip empty or very short lines
                if not line or len(line) < 2:
                    continue
                    
                # Skip lines with skip patterns
                if any(pattern in line.lower() for pattern in skip_patterns):
                    logger.info(f"📋 Skipping line {i}: '{line[:50]}...' (contains skip pattern)")
                    continue
                
                # Skip lines with email addresses (likely contact info)
                if '@' in line:
                    logger.info(f"📋 Skipping line {i}: '{line[:50]}...' (contains email)")
                    continue
                    
                # Skip lines with too many numbers (likely phone/date)
                number_count = sum(1 for char in line if char.isdigit())
                if number_count > len(line) * 0.3:  # More than 30% numbers
                    logger.info(f"📋 Skipping line {i}: '{line[:50]}...' (too many numbers)")
                    continue
                
                # Clean the line
                cleaned_line = self._clean_name_line(line)
                if not cleaned_line:
                    continue
                    
                # Check if this looks like a name
                if self._is_likely_name(cleaned_line):
                    extracted_name = self._format_name(cleaned_line)
                    logger.info(f"✅ Found name on line {i}: '{extracted_name}' from '{line}'")
                    return extracted_name
                else:
                    logger.info(f"📋 Rejected line {i}: '{cleaned_line}' (doesn't match name pattern)")
            
            # Fallback: Look for patterns like "Name: John Doe" in first 15 lines
            for i, line in enumerate(lines[:15]):
                name_match = re.search(r'(?:name|candidate|applicant)[\s:]*([A-Za-z\s\-\'\.]{2,50})', line, re.IGNORECASE)
                if name_match:
                    potential_name = name_match.group(1).strip()
                    if self._is_likely_name(potential_name):
                        extracted_name = self._format_name(potential_name)
                        logger.info(f"✅ Found name via pattern on line {i}: '{extracted_name}'")
                        return extracted_name
            
            # Last resort: Look for capitalized words that could be names
            for i, line in enumerate(lines[:8]):
                words = line.split()
                if len(words) >= 2 and len(words) <= 4:
                    # Check if first 2-3 words are capitalized and look like names
                    potential_names = words[:3]
                    if all(word[0].isupper() and word.isalpha() and len(word) >= 2 for word in potential_names[:2]):
                        name_candidate = ' '.join(potential_names[:2])
                        if len(name_candidate) >= 4 and not any(pattern in name_candidate.lower() for pattern in skip_patterns):
                            extracted_name = self._format_name(name_candidate)
                            logger.info(f"✅ Found name via capitalization on line {i}: '{extracted_name}'")
                            return extracted_name
            
            logger.warning("⚠️ Could not find name in CV text")
            return "Unknown"
            
        except Exception as e:
            logger.error(f"❌ Name extraction failed: {e}")
            return "Unknown"

    def _clean_name_line(self, line: str) -> str:
        """Clean a line to extract potential name"""
        # Remove common prefixes/suffixes
        line = re.sub(r'^(mr\.?|mrs\.?|ms\.?|dr\.?|prof\.?)\s*', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\s*(jr\.?|sr\.?|ii|iii|iv)$', '', line, flags=re.IGNORECASE)
        
        # Remove extra whitespace and special characters (but keep hyphens, apostrophes)
        line = re.sub(r'[^\w\s\-\'\.]', ' ', line)
        line = re.sub(r'\s+', ' ', line).strip()
        
        # Remove trailing dots
        line = line.rstrip('.')
        
        return line

    def _is_likely_name(self, text: str) -> bool:
        """Check if text looks like a person's name"""
        if not text or len(text) < 2:
            return False
            
        words = text.split()
        
        # Must have 1-4 words
        if len(words) < 1 or len(words) > 4:
            return False
        
        # Each word should be 2+ characters
        if any(len(word) < 2 for word in words):
            return False
            
        # Should contain only letters, spaces, hyphens, apostrophes
        if not re.match(r'^[A-Za-z\s\-\'\.]+$', text):
            return False
            
        # First word should start with capital letter
        if not words[0][0].isupper():
            return False
            
        # For multiple words, at least first 2 should be capitalized
        if len(words) >= 2 and not words[1][0].isupper():
            return False
            
        # Shouldn't be too long
        if len(text) > 50:
            return False
            
        # Common false positives
        false_positives = [
            'unknown', 'name', 'full name', 'first name', 'last name',
            'page', 'contact', 'email', 'phone', 'mobile', 'address',
            'objective', 'summary', 'profile', 'about me', 'personal',
            'experience', 'education', 'skills', 'work', 'employment'
        ]
        
        if text.lower() in false_positives:
            return False
            
        return True

    def _format_name(self, name: str) -> str:
        """Format name properly"""
        # Remove extra spaces and format
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Title case
        name = name.title()
        
        # Fix common abbreviations
        name = re.sub(r'\bMc([a-z])', r'Mc\1', name)  # McDonald -> McDonald
        name = re.sub(r'\bO\'([a-z])', r"O'\1", name)  # O'connor -> O'Connor
        
        return name
    
    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address"""
        try:
            matches = self.email_pattern.findall(text)
            if matches:
                # Return the first valid email
                for email in matches:
                    email = email.lower()
                    # Basic validation
                    if len(email) > 5 and '.' in email.split('@')[1]:
                        return email
            return None
        except Exception as e:
            logger.warning(f"Email extraction failed: {e}")
            return None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number"""
        try:
            for pattern in self.phone_patterns:
                matches = pattern.findall(text)
                if matches:
                    # Clean and return first match
                    phone = re.sub(r'[^\d\+]', '', matches[0])
                    if len(phone) >= 10:  # Minimum valid phone length
                        return matches[0]  # Return original format
            return None
        except Exception as e:
            logger.warning(f"Phone extraction failed: {e}")
            return None
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location information"""
        try:
            # Try location patterns
            for pattern in self.location_patterns:
                matches = pattern.findall(text)
                if matches:
                    return matches[0]
            
            # Look for common location keywords
            location_keywords = ['address', 'location', 'based in', 'from']
            for keyword in location_keywords:
                pattern = re.compile(f'{keyword}:?\s*([A-Za-z\s,]+)', re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    location = match.group(1).strip()
                    if len(location) > 3 and len(location) < 50:
                        return location
            
            return None
        except Exception as e:
            logger.warning(f"Location extraction failed: {e}")
            return None
    
    def _extract_basic_experience(self, text: str, lines: List[str]) -> Dict[str, Any]:
        """Extract basic experience information"""
        try:
            experience_data = {
                'total_years': 0.0,
                'job_titles': [],
                'companies': [],
                'date_ranges': []
            }
            
            # Find date ranges
            current_year = datetime.now().year
            total_experience = 0.0
            
            for pattern in self.date_patterns:
                matches = pattern.findall(text)
                for match in matches:
                    try:
                        start_year = int(match[0]) if match[0].isdigit() else current_year - 5
                        
                        if len(match) > 2 and match[2].lower() in ['present', 'current']:
                            end_year = current_year
                        elif len(match) > 1:
                            end_year = int(match[1]) if match[1].isdigit() else start_year + 1
                        else:
                            end_year = start_year + 1
                        
                        years = max(0, min(end_year - start_year, 15))  # Cap at 15 years per position
                        total_experience += years
                        experience_data['date_ranges'].append(f"{start_year}-{end_year}")
                        
                    except (ValueError, IndexError):
                        continue
            
            # Estimate experience if no dates found
            if total_experience == 0:
                # Count job-related keywords as proxy
                job_indicators = len([line for line in lines if any(keyword in line.lower() for keyword in self.job_title_keywords)])
                total_experience = min(job_indicators * 1.5, 10)  # Rough estimate
            
            experience_data['total_years'] = round(min(total_experience, 30), 1)  # Cap at 30 years
            
            # Extract job titles (simple approach)
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in self.job_title_keywords):
                    # Clean and add job title
                    title = line.strip()
                    if len(title) > 5 and len(title) < 100:
                        experience_data['job_titles'].append(title)
            
            # Remove duplicates and limit
            experience_data['job_titles'] = list(set(experience_data['job_titles']))[:5]
            
            return experience_data
            
        except Exception as e:
            logger.warning(f"Experience extraction failed: {e}")
            return {'total_years': 0.0, 'job_titles': [], 'companies': [], 'date_ranges': []}
    
    def _extract_quick_skills(self, text: str) -> List[str]:
        """Extract commonly mentioned skills"""
        try:
            text_lower = text.lower()
            found_skills = []
            
            for skill in self.common_skills:
                # Check for whole word matches
                if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                    found_skills.append(skill)
            
            # Limit to top 10 most relevant
            return found_skills[:10]
            
        except Exception as e:
            logger.warning(f"Skills extraction failed: {e}")
            return []
    
    def _extract_education_hints(self, text: str) -> List[str]:
        """Quick extraction of education hints"""
        try:
            education_keywords = [
                'bachelor', 'master', 'phd', 'degree', 'university', 'college',
                'bsc', 'msc', 'ba', 'ma', 'btech', 'mtech', 'certification'
            ]
            
            education_hints = []
            lines = text.lower().split('\n')
            
            for line in lines:
                if any(keyword in line for keyword in education_keywords):
                    # Clean and add education line
                    clean_line = line.strip()
                    if len(clean_line) > 10 and len(clean_line) < 200:
                        education_hints.append(clean_line.title())
            
            return education_hints[:3]  # Top 3 education hints
            
        except Exception as e:
            logger.warning(f"Education extraction failed: {e}")
            return []
    
    def _calculate_confidence(self, basic_info: Dict[str, Any]) -> float:
        """Calculate extraction confidence score"""
        try:
            confidence = 0.0
            
            # Name confidence
            if basic_info['name'] and basic_info['name'] != 'Unknown':
                confidence += 0.3
            
            # Email confidence
            if basic_info['email']:
                confidence += 0.2
            
            # Experience confidence
            if basic_info['basic_experience']['total_years'] > 0:
                confidence += 0.2
            
            # Skills confidence
            if basic_info['quick_skills']:
                confidence += 0.15
            
            # Contact info confidence
            if basic_info['phone'] or basic_info['location']:
                confidence += 0.1
            
            # Education hints confidence
            if basic_info['education_hints']:
                confidence += 0.05
            
            return round(min(confidence, 1.0), 2)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _create_fallback_basic_info(self, filename: str = None) -> Dict[str, Any]:
        """Create fallback response when extraction fails"""
        return {
            'name': 'Unknown',
            'email': None,
            'phone': None,
            'location': None,
            'basic_experience': {
                'total_years': 0.0,
                'job_titles': [],
                'companies': [],
                'date_ranges': []
            },
            'quick_skills': [],
            'education_hints': [],
            'extraction_confidence': 0.1,
            'extraction_metadata': {
                'extraction_time': 0.0,
                'text_length': 0,
                'lines_processed': 0,
                'filename': filename,
                'extraction_method': 'fallback',
                'error': 'Quick extraction failed'
            }
        }

# Global quick extractor instance
quick_extractor = QuickExtractorService()