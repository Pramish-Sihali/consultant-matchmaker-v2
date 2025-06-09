# app/services/project_matcher.py - Optimized Project Matching Service with Qwen 2.5

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from app.services.ai_client import ai_client
from app.config import settings
from cachetools import TTLCache

logger = logging.getLogger(__name__)

class ProjectMatcherService:
    """Intelligent project-consultant matching using Qwen 2.5"""
    
    def __init__(self):
        self.batch_size = 8  # Optimized for Qwen 2.5
        self.cache = TTLCache(maxsize=50, ttl=1800) if settings.enable_caching else None  # 30min cache
        
    async def match_project(
        self, 
        project_data: Dict[str, Any], 
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Main project matching function optimized for Qwen 2.5"""
        
        description = project_data["description"]
        title = project_data.get("title", "Untitled Project")
        consultants = project_data["consultants"]
        max_matches = project_data.get("max_matches", 10)
        min_score = project_data.get("min_score", 0.0)
        
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Starting project matching: {title}")
            logger.info(f"📄 Description: {len(description)} chars, {len(consultants)} consultants")
            
            # Check cache first
            cache_key = f"project_match_{hash(description)}_{len(consultants)}"
            if self.cache and cache_key in self.cache:
                logger.info("📋 Using cached project matching results")
                cached_result = self.cache[cache_key]
                
                # Filter cached results by current parameters
                filtered_matches = [
                    match for match in cached_result["matches"]
                    if match["match_score"] >= min_score
                ][:max_matches]
                
                return {
                    "matches": filtered_matches,
                    "project_requirements": cached_result.get("project_requirements", {}),
                    "from_cache": True
                }
            
            if progress_callback:
                progress_callback("analysis", 10, "Analyzing project requirements with Qwen 2.5...")
            
            # Step 1: Extract project requirements using Qwen 2.5
            requirements_start = time.time()
            project_requirements = await self._extract_project_requirements(description, title)
            requirements_time = time.time() - requirements_start
            
            logger.info(f"📋 Requirements extracted in {requirements_time:.2f}s")
            logger.info(f"   Skills needed: {len(project_requirements.get('skills_required', []))}")
            logger.info(f"   Experience: {project_requirements.get('experience_required', 0)} years")
            
            if progress_callback:
                progress_callback("matching", 30, "Starting consultant evaluation...")
            
            # Step 2: Process consultants in optimized batches
            matching_start = time.time()
            all_matches = []
            
            for i in range(0, len(consultants), self.batch_size):
                batch = consultants[i:i + self.batch_size]
                batch_progress = 30 + (i / len(consultants)) * 60
                
                if progress_callback:
                    progress_callback(
                        "matching", 
                        batch_progress, 
                        f"Evaluating batch {i//self.batch_size + 1}/{(len(consultants)-1)//self.batch_size + 1}..."
                    )
                
                batch_matches = await self._process_consultant_batch(
                    batch, project_requirements, description, title
                )
                all_matches.extend(batch_matches)
            
            matching_time = time.time() - matching_start
            
            if progress_callback:
                progress_callback("ranking", 90, "Ranking and finalizing matches...")
            
            # Step 3: Rank and filter results
            ranking_start = time.time()
            
            # Sort by match score
            ranked_matches = sorted(all_matches, key=lambda x: x["match_score"], reverse=True)
            
            # Apply minimum score filter
            filtered_matches = [
                match for match in ranked_matches 
                if match["match_score"] >= min_score
            ]
            
            # Apply max matches limit
            final_matches = filtered_matches[:max_matches]
            
            # Round scores for consistency
            for match in final_matches:
                match["match_score"] = round(match["match_score"], 1)
            
            ranking_time = time.time() - ranking_start
            total_time = time.time() - start_time
            
            result = {
                "matches": final_matches,
                "project_requirements": project_requirements,
                "processing_stats": {
                    "total_time": total_time,
                    "requirements_time": requirements_time,
                    "matching_time": matching_time,
                    "ranking_time": ranking_time,
                    "total_consultants": len(consultants),
                    "qualified_matches": len(filtered_matches),
                    "returned_matches": len(final_matches)
                }
            }
            
            # Cache result
            if self.cache:
                self.cache[cache_key] = result
            
            if progress_callback:
                progress_callback("complete", 100, "Matching completed")
            
            logger.info(f"✅ Matching completed in {total_time:.2f}s")
            logger.info(f"🎯 Results: {len(final_matches)} matches from {len(consultants)} consultants")
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Project matching error after {total_time:.2f}s: {e}")
            if progress_callback:
                progress_callback("error", 0, f"Error: {str(e)}")
            raise Exception(f"Project matching failed: {str(e)}")
    
    async def _extract_project_requirements(
        self, 
        description: str, 
        title: str
    ) -> Dict[str, Any]:
        """Extract project requirements using Qwen 2.5"""
        
        prompt = self._create_requirements_prompt(description, title)
        
        try:
            if settings.ai_provider == "ollama":
                response = await ai_client._call_ollama(prompt, temperature=0.1, max_tokens=2000)
            else:
                response = await ai_client._call_openai(prompt, temperature=0.1, max_tokens=2000)
            
            requirements = self._parse_requirements_response(response)
            
            logger.info(f"📋 Requirements extracted:")
            logger.info(f"   Experience required: {requirements.get('experience_required', 0)} years")
            logger.info(f"   Skills required: {len(requirements.get('skills_required', []))}")
            logger.info(f"   Domains: {requirements.get('domains', [])}")
            
            return requirements
            
        except Exception as e:
            logger.error(f"❌ Requirements extraction failed: {e}")
            # Return basic fallback requirements
            return {
                "experience_required": 3.0,
                "qualifications_required": [],
                "skills_required": [],
                "domains": [],
                "priority_skills": [],
                "complexity_level": "medium"
            }
    
    async def _process_consultant_batch(
        self, 
        consultant_batch: List[Dict], 
        project_requirements: Dict, 
        project_description: str,
        project_title: str
    ) -> List[Dict]:
        """Process a batch of consultants with optimized concurrent evaluation"""
        
        # Create semaphore to control concurrency for Qwen 2.5
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent AI calls
        
        async def evaluate_with_semaphore(consultant):
            async with semaphore:
                return await self._evaluate_consultant(
                    consultant, project_requirements, project_description, project_title
                )
        
        # Process consultants concurrently
        tasks = [
            evaluate_with_semaphore(consultant)
            for consultant in consultant_batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        matches = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Error evaluating {consultant_batch[i].get('name', 'Unknown')}: {result}")
                # Create fallback match with low score
                matches.append(self._create_fallback_match(consultant_batch[i]))
            else:
                matches.append(result)
        
        return matches
    
    async def _evaluate_consultant(
        self, 
        consultant: Dict, 
        project_requirements: Dict, 
        project_description: str,
        project_title: str
    ) -> Dict:
        """Evaluate a single consultant against project using Qwen 2.5"""
        
        try:
            consultant_name = consultant.get('name', 'Unknown')
            logger.info(f"🔍 Evaluating: {consultant_name}")
            
            # Calculate rule-based scores for baseline
            experience_score = self._calculate_experience_score(consultant, project_requirements)
            skills_score = self._calculate_skills_score(consultant, project_requirements)
            qualifications_score = self._calculate_qualifications_score(consultant, project_requirements)
            
            logger.info(f"📊 {consultant_name}: exp={experience_score:.1f}, skills={skills_score:.1f}, quals={qualifications_score:.1f}")
            
            # Get detailed AI analysis using Qwen 2.5
            try:
                ai_analysis = await self._get_qwen_analysis(
                    consultant, project_description, project_title, project_requirements
                )
                logger.info(f"🤖 AI analysis completed for {consultant_name}")
            except Exception as ai_error:
                logger.error(f"❌ AI analysis failed for {consultant_name}: {ai_error}")
                # Use fallback analysis
                ai_analysis = self._create_fallback_analysis(consultant)
            
            # Calculate weighted final score
            # Weights: Skills (40%) + Qualifications (35%) + Experience (20%) + Prior Engagement (5%)
            base_score = (skills_score * 0.4) + (qualifications_score * 0.35) + (experience_score * 0.2)
            
            # Prior engagement bonus
            prior_bonus = 5.0 if consultant.get("prior_engagement") else 0.0
            
            # AI confidence boost (can add up to 10 points)
            ai_confidence = ai_analysis.get("confidence_boost", 0.0)
            
            final_score = min(100.0, base_score + prior_bonus + ai_confidence)
            
            logger.info(f"🎯 Final score for {consultant_name}: {final_score:.1f}")
            
            # Transform match reasons to expected format
            match_reasons = {
                "experience": ai_analysis.get("experience_assessment", f"{experience_score:.0f}% experience match"),
                "qualifications": ai_analysis.get("qualification_matches", []),
                "skills": ai_analysis.get("skill_matches", []),
                "strengths": ai_analysis.get("strengths", []),
                "gaps": ai_analysis.get("gaps", []),
                "recommendation": ai_analysis.get("recommendation", f"Match score: {final_score:.0f}%"),
                "prior_engagement_bonus": consultant.get("prior_engagement", False),
                "component_scores": {
                    "experience": round(experience_score, 1),
                    "skills": round(skills_score, 1),
                    "qualifications": round(qualifications_score, 1),
                    "ai_boost": round(ai_confidence, 1)
                }
            }
            
            return {
                "consultant_id": consultant["id"],
                "match_score": round(final_score, 1),
                "match_reasons": match_reasons
            }
            
        except Exception as e:
            logger.error(f"❌ Evaluation error for {consultant.get('name', 'Unknown')}: {e}")
            return self._create_fallback_match(consultant)
    
    async def _get_qwen_analysis(
        self, 
        consultant: Dict, 
        project_description: str,
        project_title: str,
        requirements: Dict
    ) -> Dict:
        """Get detailed analysis from Qwen 2.5"""
        
        prompt = self._create_consultant_analysis_prompt(
            consultant, project_description, project_title, requirements
        )
        
        try:
            if settings.ai_provider == "ollama":
                response = await ai_client._call_ollama(prompt, temperature=0.2, max_tokens=1500)
            else:
                response = await ai_client._call_openai(prompt, temperature=0.2, max_tokens=1500)
            
            return self._parse_analysis_response(response)
            
        except Exception as e:
            logger.error(f"❌ Qwen analysis failed: {e}")
            raise e
    
    def _calculate_experience_score(self, consultant: Dict, requirements: Dict) -> float:
        """Calculate experience match score"""
        try:
            consultant_exp = consultant.get("experience_years", {}).get("total", 0)
            required_exp = requirements.get("experience_required", 3.0)
            
            if consultant_exp == 0:
                return 20.0  # Some points for any professional background
            
            if consultant_exp >= required_exp * 1.5:
                return 100.0  # Significantly exceeds requirements
            elif consultant_exp >= required_exp:
                return 90.0   # Meets requirements well
            elif consultant_exp >= required_exp * 0.8:
                return 75.0   # Close to requirements
            elif consultant_exp >= required_exp * 0.5:
                return 60.0   # Some relevant experience
            else:
                return max(30.0, (consultant_exp / required_exp) * 60.0)
            
        except Exception as e:
            logger.error(f"❌ Experience score calculation error: {e}")
            return 50.0
    
    def _calculate_skills_score(self, consultant: Dict, requirements: Dict) -> float:
        """Calculate skills match score"""
        try:
            # Get consultant skills
            consultant_skills = []
            skills_data = consultant.get("skills", {})
            if skills_data:
                consultant_skills.extend(skills_data.get("technical", []))
                consultant_skills.extend(skills_data.get("domain", []))
            
            consultant_skills = [skill.lower().strip() for skill in consultant_skills if skill]
            
            # Get required skills
            required_skills = [skill.lower().strip() for skill in requirements.get("skills_required", []) if skill]
            priority_skills = [skill.lower().strip() for skill in requirements.get("priority_skills", []) if skill]
            
            if not required_skills:
                return 75.0  # No specific skills required
            
            # Calculate matches
            matched_skills = 0
            priority_matches = 0
            
            for req_skill in required_skills:
                has_skill = any(
                    req_skill in cons_skill or cons_skill in req_skill or 
                    self._skills_similarity(req_skill, cons_skill) > 0.8
                    for cons_skill in consultant_skills
                )
                
                if has_skill:
                    matched_skills += 1
                    if req_skill in priority_skills:
                        priority_matches += 1
            
            # Base score from matches
            match_percentage = matched_skills / len(required_skills)
            base_score = match_percentage * 80.0
            
            # Priority skill bonus
            if priority_skills:
                priority_bonus = (priority_matches / len(priority_skills)) * 20.0
                base_score += priority_bonus
            
            # Skill abundance bonus (having many relevant skills)
            abundance_bonus = min(10.0, len(consultant_skills) * 0.5)
            
            final_score = min(100.0, base_score + abundance_bonus)
            
            return final_score
            
        except Exception as e:
            logger.error(f"❌ Skills score calculation error: {e}")
            return 50.0
    
    def _calculate_qualifications_score(self, consultant: Dict, requirements: Dict) -> float:
        """Calculate qualifications match score"""
        try:
            # Get consultant qualifications
            consultant_quals = []
            quals_data = consultant.get("qualifications", {})
            if quals_data:
                consultant_quals.extend(quals_data.get("degrees", []))
                consultant_quals.extend(quals_data.get("certifications", []))
                consultant_quals.extend(quals_data.get("licenses", []))
            
            consultant_quals = [qual.lower().strip() for qual in consultant_quals if qual]
            
            # Get required qualifications
            required_quals = [qual.lower().strip() for qual in requirements.get("qualifications_required", []) if qual]
            
            if not required_quals:
                # Award points based on having any qualifications
                if len(consultant_quals) >= 3:
                    return 85.0  # Well qualified
                elif len(consultant_quals) >= 1:
                    return 70.0  # Some qualifications
                else:
                    return 60.0  # No specific qualifications required
            
            # Calculate matches
            matched_quals = 0
            for req_qual in required_quals:
                has_qual = any(
                    req_qual in cons_qual or cons_qual in req_qual
                    for cons_qual in consultant_quals
                )
                if has_qual:
                    matched_quals += 1
            
            # Base score from matches
            if matched_quals == len(required_quals):
                base_score = 100.0  # All requirements met
            elif matched_quals > 0:
                base_score = (matched_quals / len(required_quals)) * 90.0
            else:
                base_score = 40.0  # No specific matches but may have other qualifications
            
            # Bonus for additional relevant qualifications
            additional_bonus = min(10.0, len(consultant_quals) * 2.0)
            
            final_score = min(100.0, base_score + additional_bonus)
            
            return final_score
            
        except Exception as e:
            logger.error(f"❌ Qualifications score calculation error: {e}")
            return 50.0
    
    def _skills_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity between two skills"""
        # Simple similarity check
        if skill1 == skill2:
            return 1.0
        
        # Check for common abbreviations/variations
        skill_mappings = {
            "js": "javascript",
            "ts": "typescript", 
            "py": "python",
            "ai": "artificial intelligence",
            "ml": "machine learning",
            "aws": "amazon web services",
            "gcp": "google cloud platform"
        }
        
        skill1_normalized = skill_mappings.get(skill1, skill1)
        skill2_normalized = skill_mappings.get(skill2, skill2)
        
        if skill1_normalized == skill2_normalized:
            return 0.9
        
        # Check if one is contained in the other
        if skill1 in skill2 or skill2 in skill1:
            return 0.8
        
        # Check for common words
        words1 = set(skill1.split())
        words2 = set(skill2.split())
        common_words = words1.intersection(words2)
        
        if common_words and len(common_words) >= len(words1) * 0.5:
            return 0.7
        
        return 0.0
    
    def _create_requirements_prompt(self, description: str, title: str) -> str:
        """Create prompt for project requirements extraction"""
        
        return f"""You are a project requirements analyst using Qwen 2.5. Extract key requirements from this project description.

PROJECT: {title}

DESCRIPTION:
{description}

Extract and respond with ONLY valid JSON:

{{
  "experience_required": 0.0,
  "qualifications_required": ["specific degrees, certifications, licenses needed"],
  "skills_required": ["technical skills, tools, programming languages"],
  "domains": ["industry domains, business areas"],
  "priority_skills": ["most critical 2-3 skills from skills_required"],
  "complexity_level": "low|medium|high"
}}

EXTRACTION RULES:
1. Be specific about technical requirements
2. Estimate experience level from project complexity
3. Include both hard skills and domain knowledge
4. Identify 2-3 most critical skills as priority
5. Set complexity based on technical sophistication

Respond with ONLY the JSON object:"""
    
    def _create_consultant_analysis_prompt(
        self, 
        consultant: Dict, 
        project_description: str,
        project_title: str,
        requirements: Dict
    ) -> str:
        """Create prompt for consultant analysis"""
        
        # Prepare consultant summary
        consultant_skills = []
        skills_data = consultant.get("skills", {})
        if skills_data:
            consultant_skills.extend(skills_data.get("technical", []))
            consultant_skills.extend(skills_data.get("domain", []))
        
        consultant_quals = []
        quals_data = consultant.get("qualifications", {})
        if quals_data:
            consultant_quals.extend(quals_data.get("degrees", []))
            consultant_quals.extend(quals_data.get("certifications", []))
            consultant_quals.extend(quals_data.get("licenses", []))
        
        experience_total = consultant.get('experience_years', {}).get('total', 0)
        
        return f"""You are a consultant matching expert using Qwen 2.5. Evaluate this consultant for the project.

CONSULTANT: {consultant.get('name', 'Unknown')}
Experience: {experience_total} years
Skills: {', '.join(consultant_skills[:10])}  
Qualifications: {', '.join(consultant_quals[:5])}
Prior Engagement: {consultant.get('prior_engagement', False)}

PROJECT: {project_title}
DESCRIPTION: {project_description}

REQUIREMENTS:
{json.dumps(requirements, indent=2)}

Analyze the match and respond with ONLY valid JSON:

{{
  "experience_assessment": "brief evaluation of experience fit",
  "skill_matches": ["specific skills that match"],
  "qualification_matches": ["specific qualifications that match"],
  "strengths": ["key strengths for this project"],
  "gaps": ["potential gaps or concerns"],
  "recommendation": "overall recommendation summary",
  "confidence_boost": 0.0
}}

ANALYSIS RULES:
1. confidence_boost: 0-10 points for exceptional fit beyond baseline scores
2. Be specific about actual matches found
3. Highlight unique strengths and potential concerns
4. Focus on project-specific relevance

Respond with ONLY the JSON object:"""
    
    def _parse_requirements_response(self, response: str) -> Dict[str, Any]:
        """Parse requirements extraction response"""
        try:
            # Clean and extract JSON
            response = response.strip()
            json_start = response.find('{')
            json_end = response.rfind('}')
            
            if json_start == -1 or json_end == -1:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end + 1]
            parsed = json.loads(json_str)
            
            # Validate and set defaults
            return {
                "experience_required": float(parsed.get("experience_required", 3.0)),
                "qualifications_required": parsed.get("qualifications_required", []),
                "skills_required": parsed.get("skills_required", []),
                "domains": parsed.get("domains", []),
                "priority_skills": parsed.get("priority_skills", []),
                "complexity_level": parsed.get("complexity_level", "medium")
            }
            
        except Exception as e:
            logger.error(f"❌ Requirements parsing error: {e}")
            return {
                "experience_required": 3.0,
                "qualifications_required": [],
                "skills_required": [],
                "domains": [],
                "priority_skills": [],
                "complexity_level": "medium"
            }
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse consultant analysis response"""
        try:
            # Clean and extract JSON
            response = response.strip()
            json_start = response.find('{')
            json_end = response.rfind('}')
            
            if json_start == -1 or json_end == -1:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end + 1]
            parsed = json.loads(json_str)
            
            return {
                "experience_assessment": parsed.get("experience_assessment", "Experience evaluated"),
                "skill_matches": parsed.get("skill_matches", []),
                "qualification_matches": parsed.get("qualification_matches", []),
                "strengths": parsed.get("strengths", []),
                "gaps": parsed.get("gaps", []),
                "recommendation": parsed.get("recommendation", "Analysis completed"),
                "confidence_boost": max(0.0, min(10.0, float(parsed.get("confidence_boost", 0.0))))
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis parsing error: {e}")
            return self._create_fallback_analysis({})
    
    def _create_fallback_analysis(self, consultant: Dict) -> Dict[str, Any]:
        """Create fallback analysis when AI fails"""
        
        experience_years = consultant.get('experience_years', {}).get('total', 0)
        skills_data = consultant.get('skills', {})
        quals_data = consultant.get('qualifications', {})
        
        total_skills = len(skills_data.get('technical', [])) + len(skills_data.get('domain', []))
        total_quals = (len(quals_data.get('degrees', [])) + 
                      len(quals_data.get('certifications', [])) + 
                      len(quals_data.get('licenses', [])))
        
        return {
            "experience_assessment": f"{experience_years} years of professional experience",
            "skill_matches": [f"{total_skills} documented skills"],
            "qualification_matches": [f"{total_quals} qualifications"],
            "strengths": ["Professional background", "Documented experience"],
            "gaps": ["Detailed analysis unavailable"],
            "recommendation": "Consider for further evaluation",
            "confidence_boost": 0.0
        }
    
    def _create_fallback_match(self, consultant: Dict) -> Dict:
        """Create fallback match for failed evaluations"""
        
        return {
            "consultant_id": consultant.get("id", "unknown"),
            "match_score": 25.0,  # Low but not zero
            "match_reasons": {
                "experience": "Evaluation failed",
                "qualifications": ["Evaluation failed"],
                "skills": ["Evaluation failed"],
                "strengths": ["Requires manual review"],
                "gaps": ["Automated evaluation unavailable"],
                "recommendation": "Manual evaluation required",
                "prior_engagement_bonus": consultant.get("prior_engagement", False),
                "component_scores": {
                    "experience": 0.0,
                    "skills": 0.0,
                    "qualifications": 0.0,
                    "ai_boost": 0.0
                }
            }
        }

# Global project matcher instance
project_matcher = ProjectMatcherService()