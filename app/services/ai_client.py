# app/services/ai_client.py - Complete Optimized Qwen 2.5 AI Client

import asyncio
import json
import logging
import re
import time
from typing import Dict, Any, Optional, Callable, Union
import httpx
from cachetools import TTLCache
from app.config import settings

logger = logging.getLogger(__name__)

class QwenAIClient:
    """Enhanced AI client with better timeout handling and complete CV analysis"""
    
    def __init__(self):
        self.config = settings.get_ai_config()
        self.provider = self.config["provider"]
        self.cache = TTLCache(maxsize=100, ttl=settings.cache_ttl) if settings.enable_caching else None
        
        # Enhanced timeout and retry settings for Qwen 2.5
        self.max_retries = 5  # Increased from 3
        self.base_timeout = 180  # 3 minutes base timeout for Qwen 2.5
        self.max_timeout = 600   # 10 minutes maximum
        self.ollama_health_check_timeout = 10  # Quick health check
        
        logger.info(f"🤖 Enhanced AI Client initialized: {self.provider} with {self.config.get('model', 'unknown model')}")
        logger.info(f"⏱️ Timeout settings: base={self.base_timeout}s, max={self.max_timeout}s, retries={self.max_retries}")

    async def test_connection(self) -> Dict[str, Any]:
        """Enhanced connection test with timeout handling"""
        start_time = time.time()
        
        try:
            if self.provider == "ollama":
                return await self._test_ollama_connection_enhanced()
            elif self.provider == "openai":
                return await self._test_openai_connection()
            else:
                raise Exception(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"❌ Enhanced AI connection test failed: {e}")
            return {
                "success": False,
                "provider": self.provider,
                "model": self.config.get("model"),
                "error": str(e),
                "response_time": response_time,
                "recommendation": "Check Ollama service and model availability"
            }

    async def _test_ollama_connection_enhanced(self) -> Dict[str, Any]:
        """Enhanced Ollama connection test with health checks"""
        start_time = time.time()
        
        try:
            # Step 1: Quick health check
            logger.info("🔍 Step 1: Checking Ollama service health...")
            async with httpx.AsyncClient(timeout=self.ollama_health_check_timeout) as client:
                health_response = await client.get(f"{self.config['url']}/api/version")
                health_response.raise_for_status()
                version_data = health_response.json()
                logger.info(f"✅ Ollama service healthy: v{version_data.get('version', 'unknown')}")
            
            # Step 2: Check model availability
            logger.info("🔍 Step 2: Checking model availability...")
            async with httpx.AsyncClient(timeout=self.ollama_health_check_timeout) as client:
                models_response = await client.get(f"{self.config['url']}/api/tags")
                models_response.raise_for_status()
                models_data = models_response.json()
                
                available_models = [model['name'] for model in models_data.get('models', [])]
                model_available = self.config['model'] in available_models
                
                if not model_available:
                    return {
                        "success": False,
                        "provider": "ollama",
                        "model": self.config['model'],
                        "error": f"Model {self.config['model']} not found",
                        "available_models": available_models[:5],
                        "recommendation": f"Run: ollama pull {self.config['model']}"
                    }
                
                logger.info(f"✅ Model {self.config['model']} is available")
            
            # Step 3: Test generation with progressive timeout
            logger.info("🔍 Step 3: Testing generation capabilities...")
            test_start = time.time()
            
            test_response = await self._call_ollama_with_progressive_timeout(
                'Respond with exactly: "Connection test successful"',
                temperature=0,
                max_tokens=50
            )
            
            test_time = time.time() - test_start
            response_time = time.time() - start_time
            
            return {
                "success": True,
                "provider": "ollama",
                "model": self.config['model'],
                "model_available": True,
                "available_models": available_models[:5],
                "ollama_version": version_data.get('version'),
                "test_response": test_response.strip()[:100],  # Truncate for logging
                "response_time": response_time,
                "generation_time": test_time,
                "status": "healthy",
                "recommendation": "AI system ready for processing"
            }
            
        except httpx.TimeoutException:
            return {
                "success": False,
                "provider": "ollama", 
                "model": self.config['model'],
                "error": "Ollama service timeout - check if Ollama is running",
                "response_time": time.time() - start_time,
                "recommendation": "Start Ollama service: ollama serve"
            }
        except httpx.ConnectError:
            return {
                "success": False,
                "provider": "ollama",
                "model": self.config['model'], 
                "error": "Cannot connect to Ollama service",
                "response_time": time.time() - start_time,
                "recommendation": "Check Ollama URL and ensure service is running"
            }
        except Exception as e:
            return {
                "success": False,
                "provider": "ollama",
                "model": self.config['model'],
                "error": str(e),
                "response_time": time.time() - start_time,
                "recommendation": "Check Ollama logs for detailed error information"
            }

    async def _test_openai_connection(self) -> Dict[str, Any]:
        """Test OpenAI-compatible API connection"""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config['model'],
            "messages": [{"role": "user", "content": 'Respond with exactly: "Connection test successful"'}],
            "temperature": 0,
            "max_tokens": 50
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.config['base_url']}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            response_time = time.time() - start_time
            usage = result.get('usage', {})
            
            return {
                "success": True,
                "provider": "openai",
                "model": self.config['model'],
                "test_response": result['choices'][0]['message']['content'].strip(),
                "response_time": response_time,
                "token_usage": usage,
                "tokens_per_second": round(usage.get('total_tokens', 0) / response_time, 2) if response_time > 0 else 0
            }

    async def _call_ollama_with_progressive_timeout(
        self, 
        prompt: str, 
        temperature: float = 0.1,
        max_tokens: int = 4000
    ) -> str:
        """Enhanced Ollama call with progressive timeout and better retry logic"""
        
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Progressive timeout: start with base, increase each attempt
                current_timeout = min(
                    self.base_timeout * (1.5 ** (attempt - 1)),
                    self.max_timeout
                )
                
                logger.info(f"🤖 Ollama attempt {attempt}/{self.max_retries} (timeout: {current_timeout:.0f}s)")
                
                # Check Ollama health before making the call
                if attempt > 1:  # Skip health check on first attempt
                    await self._quick_ollama_health_check()
                
                payload = {
                    "model": self.config["model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": max_tokens,
                        "repeat_penalty": 1.1,
                        "stop": ["</analysis>", "END_RESPONSE"],
                        # Optimization for Qwen 2.5
                        "num_ctx": 4096,  # Context window
                        "num_batch": 512,  # Batch size
                        "num_gpu": 1 if settings.environment == "production" else 0,
                        "low_vram": True,  # Memory optimization
                        "f16_kv": True,    # Memory optimization
                    }
                }
                
                start_time = time.time()
                
                async with httpx.AsyncClient(timeout=current_timeout) as client:
                    response = await client.post(
                        f"{self.config['url']}/api/generate",
                        json=payload
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if not result.get('response'):
                        raise Exception("Empty response from Ollama")
                    
                    generation_time = time.time() - start_time
                    response_text = result['response']
                    
                    logger.info(f"✅ Ollama success: {len(response_text)} chars in {generation_time:.1f}s")
                    return response_text
                    
            except httpx.TimeoutException as e:
                last_exception = e
                wait_time = min(10 * attempt, 60)  # Progressive backoff, max 60s
                logger.warning(f"⏰ Ollama timeout (attempt {attempt}), retrying in {wait_time}s...")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(wait_time)
                    continue
                    
            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code == 503:  # Service unavailable
                    wait_time = min(5 * attempt, 30)
                    logger.warning(f"⚠️ Ollama service busy (attempt {attempt}), retrying in {wait_time}s...")
                    
                    if attempt < self.max_retries:
                        await asyncio.sleep(wait_time)
                        continue
                else:
                    # Other HTTP errors shouldn't be retried
                    raise Exception(f"Ollama HTTP error {e.response.status_code}: {e.response.text}")
                    
            except Exception as e:
                last_exception = e
                if "connection" in str(e).lower():
                    wait_time = min(5 * attempt, 30)
                    logger.warning(f"🔌 Ollama connection issue (attempt {attempt}), retrying in {wait_time}s...")
                    
                    if attempt < self.max_retries:
                        await asyncio.sleep(wait_time)
                        continue
                else:
                    # Non-connection errors shouldn't be retried as much
                    if attempt >= 2:
                        raise Exception(f"Ollama error: {str(e)}")
        
        # All retries exhausted
        error_msg = f"Ollama failed after {self.max_retries} attempts"
        if last_exception:
            if isinstance(last_exception, httpx.TimeoutException):
                error_msg += f" (final error: timeout after {self.max_timeout}s)"
            else:
                error_msg += f" (final error: {str(last_exception)})"
        
        raise Exception(error_msg)

    async def _call_openai(
        self, 
        prompt: str, 
        temperature: float = 0.1,
        max_tokens: int = 4000
    ) -> str:
        """Optimized OpenAI-compatible API call"""
        
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "stop": ["</analysis>", "END_RESPONSE"]
        }
        
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.config["timeout"]) as client:
                    response = await client.post(
                        f"{self.config['base_url']}/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if not result.get('choices') or not result['choices'][0].get('message', {}).get('content'):
                        raise Exception("Empty response from API")
                    
                    return result['choices'][0]['message']['content']
                    
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"⏰ API call failed (attempt {attempt}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"API call failed after {self.max_retries} attempts: {str(e)}")
            except Exception as e:
                raise Exception(f"API call error: {str(e)}")

    async def _quick_ollama_health_check(self):
        """Quick health check before making Ollama calls"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.config['url']}/api/version")
                response.raise_for_status()
                logger.debug("✅ Ollama health check passed")
        except Exception as e:
            logger.warning(f"⚠️ Ollama health check failed: {e}")
            raise Exception(f"Ollama service not healthy: {e}")

    async def analyze_cv_text(
        self, 
        cv_text: str, 
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Enhanced CV analysis with better timeout handling"""
        
        # Check cache first
        cache_key = f"cv_analysis_{hash(cv_text)}"
        if self.cache and cache_key in self.cache:
            logger.info("📋 Using cached CV analysis")
            return self.cache[cache_key]
        
        try:
            if progress_callback:
                progress_callback("ai_analysis", 10, "Preprocessing CV text...")
            
            # Preprocess CV text
            processed_text = self._preprocess_cv_text(cv_text)
            
            if progress_callback:
                progress_callback("ai_analysis", 30, "Generating optimized AI prompt...")
            
            # Create optimized prompt
            prompt = self._create_qwen_cv_prompt(processed_text)
            
            if progress_callback:
                progress_callback("ai_analysis", 50, f"Calling {self.config['model']} (this may take 2-5 minutes)...")
            
            # Call AI model with enhanced error handling
            start_time = time.time()
            logger.info(f"🤖 Starting AI analysis with {self.config['model']}...")
            logger.info(f"📝 Prompt length: {len(prompt)} chars | Text length: {len(cv_text)} chars")
            
            if self.provider == "ollama":
                response = await self._call_ollama_with_progressive_timeout(prompt, temperature=0.1)
            else:
                response = await self._call_openai(prompt, temperature=0.1)
            
            ai_time = time.time() - start_time
            
            if progress_callback:
                progress_callback("ai_analysis", 80, "Parsing and validating AI response...")
            
            # Parse and validate response
            parsed_data = self._parse_qwen_response(response)
            validated_data = self._validate_and_enhance_data(parsed_data, cv_text)
            
            # Add metadata
            validated_data["_metadata"] = {
                "model": self.config.get("model"),
                "provider": self.provider,
                "processing_time": ai_time,
                "text_length": len(cv_text),
                "confidence_score": self._calculate_confidence_score(validated_data),
                "timeout_used": self.base_timeout,
                "response_length": len(response)
            }
            
            # Cache result
            if self.cache:
                self.cache[cache_key] = validated_data
            
            if progress_callback:
                progress_callback("ai_analysis", 100, "CV analysis completed successfully")
            
            logger.info(f"✅ AI analysis completed in {ai_time:.1f}s for: {validated_data.get('name', 'Unknown')}")
            logger.info(f"📊 Experience: {validated_data.get('experience_years', {}).get('total', 0)}y | "
                       f"Skills: {len(validated_data.get('skills', {}).get('technical', []))} | "
                       f"Confidence: {validated_data['_metadata']['confidence_score']}")
            
            return validated_data
            
        except Exception as e:
            ai_time = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"❌ AI analysis failed after {ai_time:.1f}s: {e}")
            
            if progress_callback:
                progress_callback("ai_analysis", 0, f"AI analysis failed: {str(e)[:100]}")
            
            # Provide more specific error messages
            if "timeout" in str(e).lower():
                detailed_error = f"AI analysis timed out after {ai_time:.1f}s. The model may be overloaded or the prompt too complex."
            elif "connection" in str(e).lower():
                detailed_error = f"Connection to AI service failed. Check if Ollama is running and accessible."
            elif "model" in str(e).lower():
                detailed_error = f"AI model error. Check if {self.config['model']} is properly loaded in Ollama."
            else:
                detailed_error = f"AI analysis error: {str(e)}"
            
            raise Exception(detailed_error)

    def _preprocess_cv_text(self, cv_text: str) -> str:
        """Preprocess CV text for better Qwen 2.5 understanding"""
        
        # Clean up text
        text = re.sub(r'\s+', ' ', cv_text.strip())
        
        # Standardize date formats for better parsing
        text = re.sub(r'(\d{4})/(\d{1,2})\s*[–-]\s*(\d{4})/(\d{1,2})', r'\1/\2 to \3/\4', text)
        text = re.sub(r'(\d{1,2})/(\d{4})\s*[–-]\s*(\d{1,2})/(\d{4})', r'\1/\2 to \3/\4', text)
        text = re.sub(r'(\d{4})\s*[–-]\s*(\d{4})', r'\1 to \2', text)
        
        # Add context markers
        if re.search(r'student|pursuing|expected.*graduation', text, re.IGNORECASE):
            text = "[STUDENT_PROFILE] " + text
        
        if re.search(r'202[4-5]|present', text, re.IGNORECASE):
            text = "[RECENT_DATES] " + text
            
        return text

    def _create_qwen_cv_prompt(self, cv_text: str) -> str:
        """Create optimized prompt specifically for Qwen 2.5 models"""
        
        return f"""You are an expert CV analysis assistant using Qwen 2.5. Analyze this CV and extract structured information accurately.

<cv_text>
{cv_text}
</cv_text>

CRITICAL INSTRUCTIONS:

1. **EXPERIENCE CALCULATION**: 
   - Calculate total work experience by adding ALL professional work periods
   - Include: full-time jobs, internships, part-time work, consulting, freelance
   - Convert months to years (12 months = 1 year, 6 months = 0.5 years)
   - For students with only internships/projects: calculate actual work months
   - If no work experience found: set total to 0

2. **DATE PARSING EXAMPLES**:
   - "2023/01 to 2025/06" = 2.4 years
   - "Jan 2024 - Present" = ~1.5 years (current: 2025)
   - "Summer 2024" = 0.25 years (3 months)
   - Multiple roles: sum non-overlapping periods

3. **DOMAIN EXPERIENCE**: Only count actual work experience in each domain
   - software_development: programming, development, tech roles
   - research: academic research, policy analysis, data research
   - engineering: civil, mechanical, electrical engineering
   - agriculture: farming, agtech, supply chain
   - blockchain: crypto, DeFi, web3, smart contracts

4. **OUTPUT FORMAT**: Respond with ONLY valid JSON:

{{
  "name": "Full name",
  "email": "email@example.com or null",
  "location": "City, Country or null",
  "experience_years": {{
    "total": 0.0,
    "total_months": 0,
    "domains": {{
      "software_development": 0.0,
      "research": 0.0,
      "engineering": 0.0,
      "agriculture": 0.0,
      "blockchain": 0.0
    }}
  }},
  "qualifications": {{
    "degrees": ["Bachelor of Science in...", "Master of..."],
    "certifications": ["AWS Certified...", "PMP..."],
    "licenses": ["Professional Engineer", "CPA"]
  }},
  "skills": {{
    "technical": ["Python", "JavaScript", "AWS", "Docker"],
    "domain": ["Machine Learning", "Data Analysis", "Project Management"]
  }}
}}

VALIDATION RULES:
- Be conservative with experience calculation
- Only include confirmed qualifications and skills
- Use exact numbers for experience (0.5, 1.5, 2.0, etc.)
- Domains should only reflect actual work experience, not just education

<analysis>
"""

    def _parse_qwen_response(self, response: str) -> Dict[str, Any]:
        """Parse Qwen 2.5 response with enhanced error handling"""
        
        try:
            # Clean the response
            response = response.strip()
            
            # Find JSON between <analysis> tags or standalone
            if '<analysis>' in response:
                start = response.find('<analysis>') + len('<analysis>')
                end = response.find('</analysis>')
                if end != -1:
                    response = response[start:end].strip()
            
            # Find JSON object
            json_start = response.find('{')
            json_end = response.rfind('}')
            
            if json_start == -1 or json_end == -1:
                raise ValueError("No JSON object found in response")
            
            json_str = response[json_start:json_end + 1]
            parsed_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['name', 'experience_years', 'qualifications', 'skills']
            for field in required_fields:
                if field not in parsed_data:
                    logger.warning(f"Missing required field: {field}")
                    parsed_data[field] = self._get_default_value(field)
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response sample: {response[:200]}...")
            return self._create_fallback_response()
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return self._create_fallback_response()

    def _validate_and_enhance_data(self, data: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Validate and enhance extracted data"""
        
        # Validate experience
        exp_data = data.get('experience_years', {})
        total_exp = exp_data.get('total', 0)
        
        # Sanity checks
        if isinstance(total_exp, str):
            try:
                total_exp = float(total_exp)
            except ValueError:
                total_exp = 0
        
        # Cap unrealistic values
        total_exp = max(0, min(50, total_exp))
        
        # Student validation
        if '[STUDENT_PROFILE]' in original_text and total_exp > 2:
            logger.info(f"Adjusting student experience from {total_exp} to max 2 years")
            total_exp = min(2, total_exp)
        
        # Update experience data
        exp_data['total'] = round(total_exp, 1)
        data['experience_years'] = exp_data
        
        # Validate and clean other fields
        data['name'] = str(data.get('name', 'Unknown')).strip()[:100]
        
        email = data.get('email')
        if email and '@' not in str(email):
            data['email'] = None
        
        # Ensure lists are actually lists
        for section in ['degrees', 'certifications', 'licenses']:
            quals = data.get('qualifications', {})
            if section in quals and not isinstance(quals[section], list):
                quals[section] = []
        
        for section in ['technical', 'domain']:
            skills = data.get('skills', {})
            if section in skills and not isinstance(skills[section], list):
                skills[section] = []
        
        return data

    def _calculate_confidence_score(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score for the extracted data"""
        
        score = 0.0
        
        # Name confidence
        if data.get('name') and data['name'] != 'Unknown':
            score += 0.2
        
        # Email confidence
        if data.get('email') and '@' in str(data['email']):
            score += 0.1
        
        # Experience confidence
        exp_total = data.get('experience_years', {}).get('total', 0)
        if exp_total > 0:
            score += 0.3
        
        # Qualifications confidence
        quals = data.get('qualifications', {})
        if any(quals.get(k, []) for k in ['degrees', 'certifications', 'licenses']):
            score += 0.2
        
        # Skills confidence
        skills = data.get('skills', {})
        if any(skills.get(k, []) for k in ['technical', 'domain']):
            score += 0.2
        
        return round(min(1.0, score), 2)

    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create fallback response when parsing fails"""
        return {
            "name": "Unknown",
            "email": None,
            "location": None,
            "experience_years": {
                "total": 0.0,
                "total_months": 0,
                "domains": {
                    "software_development": 0.0,
                    "research": 0.0,
                    "engineering": 0.0,
                    "agriculture": 0.0,
                    "blockchain": 0.0
                }
            },
            "qualifications": {
                "degrees": [],
                "certifications": [],
                "licenses": []
            },
            "skills": {
                "technical": [],
                "domain": []
            }
        }

    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields"""
        defaults = {
            "name": "Unknown",
            "email": None,
            "location": None,
            "experience_years": {"total": 0.0, "total_months": 0, "domains": {}},
            "qualifications": {"degrees": [], "certifications": [], "licenses": []},
            "skills": {"technical": [], "domain": []}
        }
        return defaults.get(field, None)

# Global AI client instance
ai_client = QwenAIClient()