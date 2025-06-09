# app/database/connection.py - Optimized Database Operations

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
    """Optimized database manager for Supabase operations"""
    
    def __init__(self):
        self.client: Client = create_client(
            settings.supabase_url, 
            settings.supabase_service_key
        )
        logger.info("✅ Database client initialized")
    
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
    # CONSULTANT OPERATIONS
    # ==========================================
    
    async def create_consultant(self, consultant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new consultant record"""
        try:
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
                "extraction_errors": consultant_data.get("extraction_errors"),
                "processing_metadata": consultant_data.get("processing_metadata")
            }
            
            result = self.client.table('consultants').insert(data).execute()
            
            if result.data:
                consultant = result.data[0]
                logger.info(f"✅ Created consultant: {consultant['name']} ({consultant['id'][:8]}...)")
                return consultant
            else:
                raise Exception("No data returned from insert")
                
        except Exception as e:
            logger.error(f"❌ Error creating consultant: {e}")
            raise Exception(f"Failed to create consultant: {str(e)}")
    
    async def get_consultant_by_id(self, consultant_id: str) -> Optional[Dict[str, Any]]:
        """Get consultant by ID"""
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
    
    async def get_consultants(self, filters: ConsultantFilters) -> List[Dict[str, Any]]:
        """Get consultants with filters"""
        try:
            query = self.client.table('consultants').select('*')
            
            # Apply filters
            if filters.status:
                query = query.eq('processing_status', filters.status.value)
            
            if filters.prior_engagement is not None:
                query = query.eq('prior_engagement', filters.prior_engagement)
            
            if filters.location:
                query = query.ilike('location', f'%{filters.location}%')
            
            # Experience filters
            if filters.min_experience is not None:
                query = query.gte('experience_years->total', filters.min_experience)
            
            if filters.max_experience is not None:
                query = query.lte('experience_years->total', filters.max_experience)
            
            # Ordering and pagination
            query = query.order('created_at', desc=True)
            
            if filters.offset:
                query = query.range(filters.offset, filters.offset + filters.limit - 1)
            else:
                query = query.limit(filters.limit)
            
            result = query.execute()
            return result.data or []
            
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
        """Get consultants available for project matching"""
        try:
            result = self.client.table('consultants')\
                .select('*')\
                .eq('processing_status', 'completed')\
                .order('prior_engagement', desc=True)\
                .order('updated_at', desc=True)\
                .execute()
            
            consultants = result.data or []
            logger.info(f"📋 Found {len(consultants)} consultants available for matching")
            return consultants
            
        except Exception as e:
            logger.error(f"❌ Error fetching available consultants: {e}")
            return []
    
    async def update_consultant(self, consultant_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update consultant record"""
        try:
            # Add updated timestamp
            updates['updated_at'] = datetime.utcnow().isoformat()
            
            result = self.client.table('consultants')\
                .update(updates)\
                .eq('id', consultant_id)\
                .execute()
            
            if result.data:
                consultant = result.data[0]
                logger.info(f"✅ Updated consultant: {consultant.get('name', 'Unknown')} ({consultant['id'][:8]}...)")
                return consultant
            else:
                raise Exception("No data returned from update")
                
        except Exception as e:
            logger.error(f"❌ Error updating consultant {consultant_id}: {e}")
            raise Exception(f"Failed to update consultant: {str(e)}")
    
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
    # PROJECT OPERATIONS
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
    
    # ==========================================
    # ANALYTICS & STATS
    # ==========================================
    
    async def get_consultant_stats(self) -> Dict[str, Any]:
        """Get consultant statistics"""
        try:
            # Get all consultants
            all_consultants = await self.get_all_consultants()
            
            stats = {
                "total": len(all_consultants),
                "by_status": {
                    "pending": len([c for c in all_consultants if c['processing_status'] == 'pending']),
                    "processing": len([c for c in all_consultants if c['processing_status'] == 'processing']),
                    "completed": len([c for c in all_consultants if c['processing_status'] == 'completed']),
                    "failed": len([c for c in all_consultants if c['processing_status'] == 'failed'])
                },
                "with_prior_engagement": len([c for c in all_consultants if c.get('prior_engagement', False)]),
                "with_files": len([c for c in all_consultants if c.get('cv_file_path')]),
                "average_experience": 0,
                "top_skills": {},
                "top_locations": {}
            }
            
            # Calculate average experience
            completed_consultants = [c for c in all_consultants if c['processing_status'] == 'completed']
            if completed_consultants:
                total_exp = 0
                exp_count = 0
                
                for consultant in completed_consultants:
                    exp_data = consultant.get('experience_years')
                    if isinstance(exp_data, dict) and exp_data.get('total'):
                        total_exp += exp_data['total']
                        exp_count += 1
                
                if exp_count > 0:
                    stats["average_experience"] = round(total_exp / exp_count, 1)
            
            # Get top skills
            skill_counts = {}
            for consultant in completed_consultants:
                skills_data = consultant.get('skills')
                if isinstance(skills_data, dict):
                    all_skills = skills_data.get('technical', []) + skills_data.get('domain', [])
                    for skill in all_skills:
                        if skill:
                            skill_counts[skill] = skill_counts.get(skill, 0) + 1
            
            # Top 10 skills
            stats["top_skills"] = dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Get top locations
            location_counts = {}
            for consultant in all_consultants:
                location = consultant.get('location')
                if location:
                    location_counts[location] = location_counts.get(location, 0) + 1
            
            # Top 10 locations
            stats["top_locations"] = dict(sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting consultant stats: {e}")
            return {}
    
    async def get_processing_queue_status(self) -> Dict[str, Any]:
        """Get processing queue status"""
        try:
            consultants = await self.get_all_consultants()
            
            # Recent activity (last 24 hours)
            from datetime import datetime, timedelta
            yesterday = datetime.utcnow() - timedelta(days=1)
            
            recent_activity = []
            for consultant in consultants:
                try:
                    updated_at = datetime.fromisoformat(consultant['updated_at'].replace('Z', '+00:00'))
                    if updated_at >= yesterday:
                        recent_activity.append({
                            "id": consultant['id'],
                            "name": consultant.get('name', 'Unknown'),
                            "status": consultant['processing_status'],
                            "updated_at": consultant['updated_at']
                        })
                except:
                    continue
            
            # Sort by update time
            recent_activity.sort(key=lambda x: x['updated_at'], reverse=True)
            
            stats = {
                "total": len(consultants),
                "pending": len([c for c in consultants if c['processing_status'] == 'pending']),
                "processing": len([c for c in consultants if c['processing_status'] == 'processing']),
                "completed": len([c for c in consultants if c['processing_status'] == 'completed']),
                "failed": len([c for c in consultants if c['processing_status'] == 'failed'])
            }
            
            return {
                "stats": stats,
                "recent_activity": recent_activity[:10],  # Last 10 activities
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting processing queue status: {e}")
            return {}
    
    # ==========================================
    # MAINTENANCE OPERATIONS
    # ==========================================
    
    async def cleanup_failed_records(self) -> Dict[str, Any]:
        """Clean up failed and corrupted consultant records"""
        try:
            logger.info("🧹 Starting cleanup of failed consultant records...")
            
            consultants = await self.get_all_consultants()
            failed_consultants = [
                c for c in consultants 
                if (
                    c['processing_status'] == 'failed' or 
                    (c['processing_status'] == 'pending' and not c.get('cv_file_path')) or
                    (c['processing_status'] == 'processing' and not c.get('cv_file_path'))
                )
            ]
            
            logger.info(f"📋 Found {len(failed_consultants)} failed/corrupted consultant records")
            
            cleaned = 0
            errors = []
            
            for consultant in failed_consultants:
                try:
                    await self.delete_consultant(consultant['id'])
                    cleaned += 1
                    logger.info(f"🗑️ Deleted consultant: {consultant['name']} ({consultant['id'][:8]}...)")
                    
                except Exception as error:
                    error_msg = f"Database deletion failed: {consultant['id']} - {str(error)}"
                    logger.error(f"❌ {error_msg}")
                    errors.append(error_msg)
            
            result = {
                "success": True,
                "message": f"Cleanup completed: {cleaned} consultant records deleted",
                "details": {
                    "consultants_deleted": cleaned,
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

# Global database instance
database = DatabaseManager()