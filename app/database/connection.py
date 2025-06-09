# app/database/connection.py - Enhanced Database Operations for Phased Processing

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
from supabase import create_client, Client
from app.config import settings
from app.models.schemas import ConsultantFilters, ProcessingStatus

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Enhanced database manager with phased processing support"""
    
    def __init__(self):
        self.client: Client = create_client(
            settings.supabase_url, 
            settings.supabase_service_key
        )
        logger.info("✅ Enhanced Database client initialized")
    
    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            result = self.client.table('consultants').select('id').limit(1).execute()
            logger.info("✅ Database connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    # ==========================================
    # ENHANCED CONSULTANT OPERATIONS
    # ==========================================
    
    async def create_consultant(self, consultant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new consultant record with phased processing support"""
        try:
            current_time = datetime.utcnow().isoformat()
            
            data = {
                "name": consultant_data.get("name", ""),
                "email": consultant_data.get("email"),
                "location": consultant_data.get("location"),
                "cv_file_path": consultant_data.get("cv_file_path"),
                "experience_years": consultant_data.get("experience_years"),
                "qualifications": consultant_data.get("qualifications"),
                "skills": consultant_data.get("skills"),
                "prior_engagement": consultant_data.get("prior_engagement", False),
                "processing_status": consultant_data.get("processing_status", "pending"),
                "processing_phase": consultant_data.get("processing_phase", "pending"),
                "extracted_text": consultant_data.get("extracted_text"),
                "basic_info": consultant_data.get("basic_info"),
                "full_analysis": consultant_data.get("full_analysis"),
                "extraction_errors": consultant_data.get("extraction_errors"),
                "processing_metadata": consultant_data.get("processing_metadata"),
                "phase_timestamps": consultant_data.get("phase_timestamps", {}),
                "extraction_metadata": consultant_data.get("extraction_metadata")
            }
            
            result = self.client.table('consultants').insert(data).execute()
            
            if result.data:
                consultant = result.data[0]
                logger.info(f"✅ Created consultant: {consultant['name']} ({consultant['id'][:8]}...) - Phase: {consultant['processing_phase']}")
                return consultant
            else:
                raise Exception("No data returned from insert")
                
        except Exception as e:
            logger.error(f"❌ Error creating consultant: {e}")
            raise Exception(f"Failed to create consultant: {str(e)}")
    
    async def get_consultant_by_id(self, consultant_id: str) -> Optional[Dict[str, Any]]:
        """Get consultant by ID with all phased processing fields"""
        try:
            result = self.client.table('consultants')\
                .select('*')\
                .eq('id', consultant_id)\
                .single()\
                .execute()
            
            return result.data if result.data else None
            
        except Exception as e:
            logger.error(f"❌ Error fetching consultant {consultant_id}: {e}")
            return None
    
    async def update_consultant_phase(
        self, 
        consultant_id: str, 
        phase_updates: Dict[str, Any],
        preserve_timestamps: bool = True
    ) -> Dict[str, Any]:
        """Update consultant with phase-specific data"""
        try:
            current_time = datetime.utcnow().isoformat()
            
            # Prepare update data
            updates = {
                'updated_at': current_time
            }
            updates.update(phase_updates)
            
            # Handle phase timestamps
            if preserve_timestamps and 'phase_timestamps' in phase_updates:
                # Get existing timestamps first
                existing = await self.get_consultant_by_id(consultant_id)
                if existing and existing.get('phase_timestamps'):
                    existing_timestamps = existing['phase_timestamps']
                    new_timestamps = phase_updates['phase_timestamps']
                    # Merge timestamps
                    merged_timestamps = {**existing_timestamps, **new_timestamps}
                    updates['phase_timestamps'] = merged_timestamps
            
            result = self.client.table('consultants')\
                .update(updates)\
                .eq('id', consultant_id)\
                .execute()
            
            if result.data:
                consultant = result.data[0]
                phase = consultant.get('processing_phase', 'unknown')
                status = consultant.get('processing_status', 'unknown')
                logger.info(f"✅ Updated consultant: {consultant.get('name', 'Unknown')} ({consultant['id'][:8]}...) - {phase}/{status}")
                return consultant
            else:
                raise Exception("No data returned from update")
                
        except Exception as e:
            logger.error(f"❌ Error updating consultant {consultant_id}: {e}")
            raise Exception(f"Failed to update consultant: {str(e)}")
    
    async def update_consultant(self, consultant_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatibility: Update consultant record"""
        return await self.update_consultant_phase(consultant_id, updates)
    
    async def get_consultants_by_phase(self, phase: str) -> List[Dict[str, Any]]:
        """Get consultants by processing phase"""
        try:
            result = self.client.table('consultants')\
                .select('*')\
                .eq('processing_phase', phase)\
                .order('created_at', desc=False)\
                .execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"❌ Error fetching consultants by phase {phase}: {e}")
            return []
    
    async def get_consultants_needing_phase_2(self) -> List[Dict[str, Any]]:
        """Get consultants ready for AI analysis (Phase 2)"""
        try:
            result = self.client.table('consultants')\
                .select('*')\
                .in_('processing_phase', ['partially_processed', 'analyzing'])\
                .not_.is_('extracted_text', 'null')\
                .order('updated_at', desc=False)\
                .execute()
            
            consultants = result.data or []
            logger.info(f"📋 Found {len(consultants)} consultants ready for Phase 2")
            return consultants
            
        except Exception as e:
            logger.error(f"❌ Error fetching Phase 2 candidates: {e}")
            return []
    
    async def get_consultant_status_summary(self, consultant_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive consultant status including phased processing info"""
        try:
            consultant = await self.get_consultant_by_id(consultant_id)
            if not consultant:
                return None
            
            # Calculate derived fields
            has_basic_data = bool(consultant.get("basic_info"))
            has_full_data = bool(
                consultant.get("experience_years") or 
                consultant.get("skills") or 
                consultant.get("qualifications")
            )
            
            basic_info = consultant.get("basic_info") or {}
            experience_total = 0.0
            if has_full_data:
                exp_data = consultant.get("experience_years")
                if isinstance(exp_data, dict):
                    experience_total = exp_data.get("total", 0.0)
            elif has_basic_data:
                basic_exp = basic_info.get("basic_experience", {})
                experience_total = basic_exp.get("total_years", 0.0)
            
            # Skills count
            skills_count = 0
            if has_full_data:
                skills_data = consultant.get("skills")
                if isinstance(skills_data, dict):
                    technical_skills = skills_data.get("technical", [])
                    domain_skills = skills_data.get("domain", [])
                    skills_count = len(technical_skills) + len(domain_skills)
            elif has_basic_data:
                quick_skills = basic_info.get("quick_skills", [])
                skills_count = len(quick_skills)
            
            # Qualifications count
            qualifications_count = 0
            if has_full_data:
                quals_data = consultant.get("qualifications")
                if isinstance(quals_data, dict):
                    degrees = quals_data.get("degrees", [])
                    certs = quals_data.get("certifications", [])
                    licenses = quals_data.get("licenses", [])
                    qualifications_count = len(degrees) + len(certs) + len(licenses)
            elif has_basic_data:
                education_hints = basic_info.get("education_hints", [])
                qualifications_count = len(education_hints)
            
            # Confidence score
            confidence_score = None
            if has_full_data:
                metadata = consultant.get("processing_metadata")
                if isinstance(metadata, dict):
                    confidence_score = metadata.get("confidence_score")
            elif has_basic_data:
                confidence_score = basic_info.get("extraction_confidence", 0.0)
            
            # Processing progress
            phase = consultant.get("processing_phase", "pending")
            progress_percentage = {
                "pending": 0,
                "extracting": 25,
                "partially_processed": 50,
                "analyzing": 75,
                "completed": 100,
                "failed": 0
            }.get(phase, 0)
            
            return {
                "success": True,
                "consultant_id": consultant_id,
                "name": consultant.get("name", "Unknown"),
                "processing_status": consultant["processing_status"],
                "processing_phase": phase,
                "progress_percentage": progress_percentage,
                "created_at": consultant["created_at"],
                "updated_at": consultant["updated_at"],
                "extraction_errors": consultant.get("extraction_errors"),
                "has_basic_data": has_basic_data,
                "has_full_data": has_full_data,
                "email": consultant.get("email") or basic_info.get("email"),
                "location": consultant.get("location") or basic_info.get("location"),
                "experience_years": experience_total,
                "skills_count": skills_count,
                "qualifications_count": qualifications_count,
                "confidence_score": confidence_score,
                "phase_timestamps": consultant.get("phase_timestamps", {}),
                "basic_info": basic_info if has_basic_data else None,
                "quick_preview": {
                    "phone": basic_info.get("phone"),
                    "quick_skills": basic_info.get("quick_skills", [])[:5],  # First 5 skills
                    "job_titles": basic_info.get("basic_experience", {}).get("job_titles", [])[:3]  # First 3 titles
                } if has_basic_data else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting consultant status: {e}")
            return None
    
    async def get_phased_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about phased processing"""
        try:
            consultants = await self.get_all_consultants()
            
            stats = {
                "total": len(consultants),
                "by_phase": {
                    "pending": 0,
                    "extracting": 0,
                    "partially_processed": 0,
                    "analyzing": 0,
                    "completed": 0,
                    "failed": 0
                },
                "by_status": {
                    "pending": 0,
                    "processing": 0,
                    "completed": 0,
                    "failed": 0
                },
                "with_basic_info": 0,
                "with_full_analysis": 0,
                "average_phase_1_time": 0.0,
                "average_phase_2_time": 0.0
            }
            
            phase_1_times = []
            phase_2_times = []
            
            for consultant in consultants:
                # Count by phase
                phase = consultant.get('processing_phase', 'pending')
                if phase in stats["by_phase"]:
                    stats["by_phase"][phase] += 1
                
                # Count by status
                status = consultant.get('processing_status', 'pending')
                if status in stats["by_status"]:
                    stats["by_status"][status] += 1
                
                # Count data availability
                if consultant.get('basic_info'):
                    stats["with_basic_info"] += 1
                
                if consultant.get('full_analysis'):
                    stats["with_full_analysis"] += 1
                
                # Collect timing data
                metadata = consultant.get('extraction_metadata', {})
                if isinstance(metadata, dict):
                    phase_1_time = metadata.get('phase_1_time')
                    if phase_1_time and isinstance(phase_1_time, (int, float)):
                        phase_1_times.append(phase_1_time)
                
                proc_metadata = consultant.get('processing_metadata', {})
                if isinstance(proc_metadata, dict):
                    phase_2_time = proc_metadata.get('phase_2_time')
                    if phase_2_time and isinstance(phase_2_time, (int, float)):
                        phase_2_times.append(phase_2_time)
            
            # Calculate averages
            if phase_1_times:
                stats["average_phase_1_time"] = round(sum(phase_1_times) / len(phase_1_times), 2)
            
            if phase_2_times:
                stats["average_phase_2_time"] = round(sum(phase_2_times) / len(phase_2_times), 2)
            
            # Calculate efficiency metrics
            stats["phase_1_success_rate"] = round(
                (stats["with_basic_info"] / stats["total"]) * 100, 1
            ) if stats["total"] > 0 else 0
            
            stats["phase_2_success_rate"] = round(
                (stats["with_full_analysis"] / stats["total"]) * 100, 1
            ) if stats["total"] > 0 else 0
            
            stats["overall_success_rate"] = round(
                (stats["by_status"]["completed"] / stats["total"]) * 100, 1
            ) if stats["total"] > 0 else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting phased processing stats: {e}")
            return {}
    
    async def cleanup_failed_phase_records(self) -> Dict[str, Any]:
        """Clean up consultants stuck in processing phases"""
        try:
            logger.info("🧹 Starting cleanup of stuck processing records...")
            
            consultants = await self.get_all_consultants()
            
            # Find consultants stuck in processing phases for too long
            from datetime import datetime, timedelta
            cutoff_time = datetime.utcnow() - timedelta(hours=2)  # 2 hours timeout
            
            stuck_consultants = []
            for consultant in consultants:
                updated_at_str = consultant.get('updated_at')
                if updated_at_str:
                    try:
                        updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                        if updated_at < cutoff_time:
                            phase = consultant.get('processing_phase', '')
                            if phase in ['extracting', 'analyzing']:
                                stuck_consultants.append(consultant)
                    except ValueError:
                        continue
            
            logger.info(f"📋 Found {len(stuck_consultants)} consultants stuck in processing")
            
            cleaned = 0
            errors = []
            
            for consultant in stuck_consultants:
                try:
                    consultant_id = consultant['id']
                    phase = consultant.get('processing_phase', 'unknown')
                    
                    # Reset to appropriate retry state
                    if phase == 'extracting':
                        # Reset to pending for Phase 1 retry
                        await self.update_consultant_phase(consultant_id, {
                            "processing_status": "pending",
                            "processing_phase": "pending",
                            "extraction_errors": {
                                "cleanup_reason": "Stuck in extracting phase",
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        })
                    elif phase == 'analyzing':
                        # Reset to partially_processed for Phase 2 retry
                        await self.update_consultant_phase(consultant_id, {
                            "processing_status": "processing",
                            "processing_phase": "partially_processed",
                            "extraction_errors": {
                                "cleanup_reason": "Stuck in analyzing phase",
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        })
                    
                    cleaned += 1
                    logger.info(f"🔄 Reset stuck consultant: {consultant['name']} ({consultant_id[:8]}...)")
                    
                except Exception as error:
                    error_msg = f"Could not reset consultant {consultant['id']}: {str(error)}"
                    logger.error(f"❌ {error_msg}")
                    errors.append(error_msg)
            
            result = {
                "success": True,
                "message": f"Cleanup completed: {cleaned} stuck consultants reset",
                "details": {
                    "stuck_consultants": len(stuck_consultants),
                    "reset_count": cleaned,
                    "errors_count": len(errors)
                }
            }
            
            if errors:
                result["errors"] = errors
                result["message"] += f" ({len(errors)} errors occurred)"
            
            logger.info(f"✅ Cleanup completed: {result['details']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Cleanup operation failed: {e}")
            return {
                "success": False,
                "message": f"Cleanup operation failed: {str(e)}"
            }
    
    # ==========================================
    # EXISTING METHODS (Enhanced for compatibility)
    # ==========================================
    
    async def get_consultants(self, filters: ConsultantFilters) -> List[Dict[str, Any]]:
        """Get consultants with filters (enhanced for phased processing)"""
        try:
            query = self.client.table('consultants').select('*')
            
            # Apply filters
            if filters.status:
                query = query.eq('processing_status', filters.status.value)
            
            if filters.prior_engagement is not None:
                query = query.eq('prior_engagement', filters.prior_engagement)
            
            if filters.location:
                query = query.ilike('location', f'%{filters.location}%')
            
            # Experience filters (check both full_analysis and basic_info)
            if filters.min_experience is not None:
                # This is complex with the new structure, so we'll filter in Python
                pass  # Will filter after retrieval
            
            # Ordering and pagination
            query = query.order('created_at', desc=True)
            
            if filters.offset:
                query = query.range(filters.offset, filters.offset + filters.limit - 1)
            else:
                query = query.limit(filters.limit)
            
            result = query.execute()
            consultants = result.data or []
            
            # Post-process filtering for experience (since we can't do complex queries easily)
            if filters.min_experience is not None or filters.max_experience is not None:
                filtered_consultants = []
                for consultant in consultants:
                    exp_total = 0.0
                    
                    # Try full analysis first
                    exp_data = consultant.get('experience_years')
                    if isinstance(exp_data, dict):
                        exp_total = exp_data.get('total', 0.0)
                    else:
                        # Try basic info
                        basic_info = consultant.get('basic_info')
                        if isinstance(basic_info, dict):
                            basic_exp = basic_info.get('basic_experience', {})
                            exp_total = basic_exp.get('total_years', 0.0)
                    
                    # Apply filters
                    if filters.min_experience is not None and exp_total < filters.min_experience:
                        continue
                    if filters.max_experience is not None and exp_total > filters.max_experience:
                        continue
                    
                    filtered_consultants.append(consultant)
                
                consultants = filtered_consultants
            
            return consultants
            
        except Exception as e:
            logger.error(f"❌ Error fetching consultants: {e}")
            raise Exception(f"Failed to fetch consultants: {str(e)}")
    
    async def get_consultants_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get consultants by processing status"""
        try:
            result = self.client.table('consultants')\
                .select('*')\
                .eq('processing_status', status)\
                .order('created_at', desc=False)\
                .execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"❌ Error fetching consultants by status {status}: {e}")
            return []
    
    async def get_all_consultants(self) -> List[Dict[str, Any]]:
        """Get all consultants"""
        try:
            result = self.client.table('consultants')\
                .select('*')\
                .order('created_at', desc=True)\
                .execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"❌ Error fetching all consultants: {e}")
            return []
    
    async def get_available_consultants(self) -> List[Dict[str, Any]]:
        """Get consultants available for project matching (completed only)"""
        try:
            result = self.client.table('consultants')\
                .select('*')\
                .eq('processing_status', 'completed')\
                .eq('processing_phase', 'completed')\
                .order('prior_engagement', desc=True)\
                .order('updated_at', desc=True)\
                .execute()
            
            consultants = result.data or []
            logger.info(f"📋 Found {len(consultants)} consultants available for matching")
            return consultants
            
        except Exception as e:
            logger.error(f"❌ Error fetching available consultants: {e}")
            return []
    
    async def delete_consultant(self, consultant_id: str) -> bool:
        """Delete consultant record"""
        try:
            result = self.client.table('consultants')\
                .delete()\
                .eq('id', consultant_id)\
                .execute()
            
            logger.info(f"🗑️ Deleted consultant: {consultant_id[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting consultant {consultant_id}: {e}")
            raise Exception(f"Failed to delete consultant: {str(e)}")
    
    async def delete_all_consultants(self) -> int:
        """Delete all consultant records"""
        try:
            # First get count
            count_result = self.client.table('consultants').select('id').execute()
            count = len(count_result.data) if count_result.data else 0
            
            # Delete all
            self.client.table('consultants').delete().neq('id', 'impossible-id').execute()
            
            logger.info(f"🗑️ Deleted {count} consultant records")
            return count
            
        except Exception as e:
            logger.error(f"❌ Error deleting all consultants: {e}")
            raise Exception(f"Failed to delete all consultants: {str(e)}")
    
    # ==========================================
    # PROJECT OPERATIONS (Unchanged)
    # ==========================================
    
    async def create_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new project record"""
        try:
            data = {
                "title": project_data.get("title"),
                "description": project_data["description"],
                "requirements_extracted": project_data.get("requirements_extracted")
            }
            
            result = self.client.table('projects').insert(data).execute()
            
            if result.data:
                project = result.data[0]
                logger.info(f"✅ Created project: {project.get('title', 'Untitled')} ({project['id'][:8]}...)")
                return project
            else:
                raise Exception("No data returned from insert")
                
        except Exception as e:
            logger.error(f"❌ Error creating project: {e}")
            raise Exception(f"Failed to create project: {str(e)}")
    
    async def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project by ID"""
        try:
            result = self.client.table('projects')\
                .select('*')\
                .eq('id', project_id)\
                .single()\
                .execute()
            
            return result.data if result.data else None
            
        except Exception as e:
            logger.error(f"❌ Error fetching project {project_id}: {e}")
            return None
    
    async def update_project_requirements(self, project_id: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Update project with extracted requirements"""
        try:
            result = self.client.table('projects')\
                .update({"requirements_extracted": requirements})\
                .eq('id', project_id)\
                .execute()
            
            if result.data:
                return result.data[0]
            else:
                raise Exception("No data returned from update")
                
        except Exception as e:
            logger.error(f"❌ Error updating project requirements {project_id}: {e}")
            raise Exception(f"Failed to update project requirements: {str(e)}")
    
    async def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get all projects"""
        try:
            result = self.client.table('projects')\
                .select('*')\
                .order('created_at', desc=True)\
                .execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"❌ Error fetching all projects: {e}")
            return []
    
    async def delete_all_projects(self) -> int:
        """Delete all project records"""
        try:
            # First get count
            count_result = self.client.table('projects').select('id').execute()
            count = len(count_result.data) if count_result.data else 0
            
            # Delete all
            self.client.table('projects').delete().neq('id', 'impossible-id').execute()
            
            logger.info(f"🗑️ Deleted {count} project records")
            return count
            
        except Exception as e:
            logger.error(f"❌ Error deleting all projects: {e}")
            raise Exception(f"Failed to delete all projects: {str(e)}")

# Global enhanced database instance
database = DatabaseManager()