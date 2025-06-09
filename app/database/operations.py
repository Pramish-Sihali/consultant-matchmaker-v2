# app/database/operations.py - Advanced Database Operations

import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from app.database.connection import database
from app.models.schemas import ConsultantFilters, ProcessingStatus

logger = logging.getLogger(__name__)

class DatabaseOperations:
    """Advanced database operations and utilities"""
    
    def __init__(self):
        self.db = database
    
    async def get_consultant_statistics(self) -> Dict[str, Any]:
        """Get comprehensive consultant statistics"""
        try:
            consultants = await self.db.get_all_consultants()
            
            stats = {
                "total_consultants": len(consultants),
                "status_breakdown": self._calculate_status_breakdown(consultants),
                "experience_distribution": self._calculate_experience_distribution(consultants),
                "skills_analysis": self._analyze_skills(consultants),
                "qualifications_analysis": self._analyze_qualifications(consultants),
                "location_distribution": self._analyze_locations(consultants),
                "processing_performance": self._analyze_processing_performance(consultants),
                "recent_activity": self._get_recent_activity(consultants)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting consultant statistics: {e}")
            return {}
    
    def _calculate_status_breakdown(self, consultants: List[Dict]) -> Dict[str, Any]:
        """Calculate status breakdown with percentages"""
        status_counts = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        }
        
        for consultant in consultants:
            status = consultant.get('processing_status', 'unknown')
            if status in status_counts:
                status_counts[status] += 1
        
        total = len(consultants)
        if total > 0:
            status_percentages = {
                status: round((count / total) * 100, 1)
                for status, count in status_counts.items()
            }
        else:
            status_percentages = {status: 0 for status in status_counts}
        
        return {
            "counts": status_counts,
            "percentages": status_percentages,
            "success_rate": status_percentages.get("completed", 0)
        }
    
    def _calculate_experience_distribution(self, consultants: List[Dict]) -> Dict[str, Any]:
        """Analyze experience distribution"""
        experience_ranges = {
            "0-1 years": 0,
            "1-3 years": 0,
            "3-5 years": 0,
            "5-10 years": 0,
            "10+ years": 0
        }
        
        total_experience = 0
        valid_experience_count = 0
        
        for consultant in consultants:
            if consultant.get('processing_status') == 'completed':
                exp_data = consultant.get('experience_years')
                if isinstance(exp_data, dict):
                    exp_total = exp_data.get('total', 0)
                    if isinstance(exp_total, (int, float)) and exp_total >= 0:
                        total_experience += exp_total
                        valid_experience_count += 1
                        
                        # Categorize experience
                        if exp_total < 1:
                            experience_ranges["0-1 years"] += 1
                        elif exp_total < 3:
                            experience_ranges["1-3 years"] += 1
                        elif exp_total < 5:
                            experience_ranges["3-5 years"] += 1
                        elif exp_total < 10:
                            experience_ranges["5-10 years"] += 1
                        else:
                            experience_ranges["10+ years"] += 1
        
        avg_experience = round(total_experience / valid_experience_count, 1) if valid_experience_count > 0 else 0
        
        return {
            "ranges": experience_ranges,
            "average_experience": avg_experience,
            "total_valid_records": valid_experience_count
        }
    
    def _analyze_skills(self, consultants: List[Dict]) -> Dict[str, Any]:
        """Analyze skills across all consultants"""
        technical_skills = {}
        domain_skills = {}
        
        for consultant in consultants:
            if consultant.get('processing_status') == 'completed':
                skills_data = consultant.get('skills')
                if isinstance(skills_data, dict):
                    # Count technical skills
                    for skill in skills_data.get('technical', []):
                        if skill:
                            technical_skills[skill] = technical_skills.get(skill, 0) + 1
                    
                    # Count domain skills
                    for skill in skills_data.get('domain', []):
                        if skill:
                            domain_skills[skill] = domain_skills.get(skill, 0) + 1
        
        # Get top skills
        top_technical = dict(sorted(technical_skills.items(), key=lambda x: x[1], reverse=True)[:10])
        top_domain = dict(sorted(domain_skills.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return {
            "top_technical_skills": top_technical,
            "top_domain_skills": top_domain,
            "total_unique_technical": len(technical_skills),
            "total_unique_domain": len(domain_skills)
        }
    
    def _analyze_qualifications(self, consultants: List[Dict]) -> Dict[str, Any]:
        """Analyze qualifications across all consultants"""
        degrees = {}
        certifications = {}
        licenses = {}
        
        for consultant in consultants:
            if consultant.get('processing_status') == 'completed':
                quals_data = consultant.get('qualifications')
                if isinstance(quals_data, dict):
                    # Count degrees
                    for degree in quals_data.get('degrees', []):
                        if degree:
                            degrees[degree] = degrees.get(degree, 0) + 1
                    
                    # Count certifications
                    for cert in quals_data.get('certifications', []):
                        if cert:
                            certifications[cert] = certifications.get(cert, 0) + 1
                    
                    # Count licenses
                    for license in quals_data.get('licenses', []):
                        if license:
                            licenses[license] = licenses.get(license, 0) + 1
        
        return {
            "top_degrees": dict(sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_certifications": dict(sorted(certifications.items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_licenses": dict(sorted(licenses.items(), key=lambda x: x[1], reverse=True)[:5]),
            "total_unique_degrees": len(degrees),
            "total_unique_certifications": len(certifications),
            "total_unique_licenses": len(licenses)
        }
    
    def _analyze_locations(self, consultants: List[Dict]) -> Dict[str, int]:
        """Analyze location distribution"""
        locations = {}
        
        for consultant in consultants:
            location = consultant.get('location')
            if location:
                locations[location] = locations.get(location, 0) + 1
        
        return dict(sorted(locations.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _analyze_processing_performance(self, consultants: List[Dict]) -> Dict[str, Any]:
        """Analyze processing performance metrics"""
        processing_times = []
        success_count = 0
        failed_count = 0
        
        for consultant in consultants:
            metadata = consultant.get('processing_metadata')
            if isinstance(metadata, dict):
                total_time = metadata.get('total_time')
                if isinstance(total_time, (int, float)):
                    processing_times.append(total_time)
            
            status = consultant.get('processing_status')
            if status == 'completed':
                success_count += 1
            elif status == 'failed':
                failed_count += 1
        
        if processing_times:
            avg_time = round(sum(processing_times) / len(processing_times), 2)
            min_time = round(min(processing_times), 2)
            max_time = round(max(processing_times), 2)
        else:
            avg_time = min_time = max_time = 0
        
        total_processed = success_count + failed_count
        success_rate = round((success_count / total_processed) * 100, 1) if total_processed > 0 else 0
        
        return {
            "average_processing_time": avg_time,
            "min_processing_time": min_time,
            "max_processing_time": max_time,
            "success_rate": success_rate,
            "total_processed": total_processed
        }
    
    def _get_recent_activity(self, consultants: List[Dict], hours: int = 24) -> List[Dict]:
        """Get recent activity within specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_activity = []
        
        for consultant in consultants:
            try:
                updated_at_str = consultant.get('updated_at')
                if updated_at_str:
                    updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                    if updated_at >= cutoff_time:
                        recent_activity.append({
                            "id": consultant['id'],
                            "name": consultant.get('name', 'Unknown'),
                            "status": consultant['processing_status'],
                            "updated_at": updated_at_str
                        })
            except (ValueError, TypeError):
                continue
        
        # Sort by update time, most recent first
        recent_activity.sort(key=lambda x: x['updated_at'], reverse=True)
        return recent_activity[:20]  # Return last 20 activities
    
    async def get_consultants_by_criteria(
        self, 
        min_experience: Optional[float] = None,
        required_skills: Optional[List[str]] = None,
        location_filter: Optional[str] = None,
        prior_engagement: Optional[bool] = None
    ) -> List[Dict]:
        """Get consultants matching specific criteria"""
        try:
            # Get all completed consultants
            consultants = await self.db.get_consultants(ConsultantFilters(
                status=ProcessingStatus.COMPLETED,
                prior_engagement=prior_engagement
            ))
            
            filtered_consultants = []
            
            for consultant in consultants:
                # Check experience requirement
                if min_experience is not None:
                    exp_data = consultant.get('experience_years')
                    if isinstance(exp_data, dict):
                        exp_total = exp_data.get('total', 0)
                        if exp_total < min_experience:
                            continue
                    else:
                        continue
                
                # Check skills requirement
                if required_skills:
                    consultant_skills = []
                    skills_data = consultant.get('skills')
                    if isinstance(skills_data, dict):
                        consultant_skills.extend(skills_data.get('technical', []))
                        consultant_skills.extend(skills_data.get('domain', []))
                    
                    consultant_skills_lower = [skill.lower() for skill in consultant_skills]
                    
                    # Check if consultant has any of the required skills
                    has_required_skill = any(
                        any(req_skill.lower() in cons_skill or cons_skill in req_skill.lower() 
                            for cons_skill in consultant_skills_lower)
                        for req_skill in required_skills
                    )
                    
                    if not has_required_skill:
                        continue
                
                # Check location filter
                if location_filter:
                    consultant_location = consultant.get('location', '').lower()
                    if location_filter.lower() not in consultant_location:
                        continue
                
                filtered_consultants.append(consultant)
            
            logger.info(f"📋 Found {len(filtered_consultants)} consultants matching criteria")
            return filtered_consultants
            
        except Exception as e:
            logger.error(f"❌ Error filtering consultants: {e}")
            return []
    
    async def bulk_update_consultant_status(
        self, 
        consultant_ids: List[str], 
        new_status: str
    ) -> Dict[str, Any]:
        """Bulk update consultant status"""
        try:
            updated_count = 0
            failed_updates = []
            
            for consultant_id in consultant_ids:
                try:
                    await self.db.update_consultant(consultant_id, {
                        "processing_status": new_status,
                        "updated_at": datetime.utcnow().isoformat()
                    })
                    updated_count += 1
                except Exception as e:
                    failed_updates.append({
                        "consultant_id": consultant_id,
                        "error": str(e)
                    })
            
            return {
                "success": True,
                "updated_count": updated_count,
                "failed_count": len(failed_updates),
                "failed_updates": failed_updates
            }
            
        except Exception as e:
            logger.error(f"❌ Bulk update failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "updated_count": 0
            }
    
    async def cleanup_old_records(self, days_old: int = 30) -> Dict[str, Any]:
        """Clean up old records and files"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            consultants = await self.db.get_all_consultants()
            old_consultants = []
            
            for consultant in consultants:
                try:
                    created_at_str = consultant.get('created_at')
                    if created_at_str:
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        if created_at < cutoff_date:
                            old_consultants.append(consultant)
                except (ValueError, TypeError):
                    continue
            
            # Delete old consultants
            deleted_count = 0
            errors = []
            
            for consultant in old_consultants:
                try:
                    # Delete file if exists
                    if consultant.get('cv_file_path'):
                        from app.services.file_storage import file_storage
                        await file_storage.delete_file(consultant['cv_file_path'])
                    
                    # Delete database record
                    await self.db.delete_consultant(consultant['id'])
                    deleted_count += 1
                    
                except Exception as e:
                    errors.append({
                        "consultant_id": consultant['id'],
                        "error": str(e)
                    })
            
            return {
                "success": True,
                "total_old_records": len(old_consultants),
                "deleted_count": deleted_count,
                "errors": errors,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Cleanup operation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_processing_queue_info(self) -> Dict[str, Any]:
        """Get detailed processing queue information"""
        try:
            consultants = await self.db.get_all_consultants()
            
            queue_info = {
                "pending": [],
                "processing": [],
                "recently_completed": [],
                "failed": []
            }
            
            for consultant in consultants:
                status = consultant.get('processing_status')
                consultant_info = {
                    "id": consultant['id'],
                    "name": consultant.get('name', 'Unknown'),
                    "created_at": consultant.get('created_at'),
                    "updated_at": consultant.get('updated_at'),
                    "has_file": bool(consultant.get('cv_file_path'))
                }
                
                if status == 'pending':
                    queue_info["pending"].append(consultant_info)
                elif status == 'processing':
                    queue_info["processing"].append(consultant_info)
                elif status == 'completed':
                    # Only include recently completed (last 24 hours)
                    try:
                        updated_at = datetime.fromisoformat(consultant.get('updated_at', '').replace('Z', '+00:00'))
                        if updated_at >= datetime.utcnow() - timedelta(hours=24):
                            queue_info["recently_completed"].append(consultant_info)
                    except:
                        pass
                elif status == 'failed':
                    consultant_info["error"] = consultant.get('extraction_errors', {}).get('final_error', 'Unknown error')
                    queue_info["failed"].append(consultant_info)
            
            # Sort by creation time
            for status_list in queue_info.values():
                status_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            return queue_info
            
        except Exception as e:
            logger.error(f"❌ Error getting queue info: {e}")
            return {}

# Global database operations instance
db_operations = DatabaseOperations()