# app/services/file_storage.py - Optimized File Storage Service

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from app.config import settings

logger = logging.getLogger(__name__)

class FileStorageService:
    """Optimized file storage service for Supabase Storage"""
    
    def __init__(self):
        # Import here to avoid circular imports
        from app.database.connection import database
        self.client = database.client
        self.bucket_name = settings.supabase_bucket_name
        
    async def test_connection(self) -> bool:
        """Test storage connection and permissions"""
        try:
            # Test by listing files
            files = await self.list_files()
            logger.info(f"✅ Storage connection successful: {len(files)} files found")
            return True
        except Exception as e:
            logger.error(f"❌ Storage connection failed: {e}")
            return False
    
    async def upload_file(
        self, 
        file_content: bytes, 
        consultant_id: str, 
        filename: str, 
        content_type: str
    ) -> str:
        """Upload CV file to Supabase storage"""
        try:
            # Generate unique file path
            timestamp = int(datetime.now().timestamp())
            file_extension = self._get_file_extension(filename, content_type)
            file_path = f"{consultant_id}/{timestamp}-{filename}"
            
            logger.info(f"📤 Uploading file: {file_path} ({len(file_content)} bytes)")
            
            # Upload to Supabase storage
            result = self.client.storage.from_(self.bucket_name).upload(
                file_path,
                file_content,
                {
                    "content-type": content_type,
                    "cache-control": "3600",
                    "upsert": "false"
                }
            )
            
            if result:
                logger.info(f"✅ File uploaded successfully: {file_path}")
                return file_path
            else:
                raise Exception("Upload failed - no result returned")
                
        except Exception as e:
            logger.error(f"❌ File upload failed: {e}")
            raise Exception(f"File upload failed: {str(e)}")
    
    async def download_file(self, file_path: str) -> bytes:
        """Download CV file from Supabase storage"""
        try:
            if not file_path:
                raise Exception("File path is required for download")
            
            logger.info(f"📥 Downloading file: {file_path}")
            
            # Download from Supabase storage
            result = self.client.storage.from_(self.bucket_name).download(file_path)
            
            if result:
                logger.info(f"✅ File downloaded successfully: {file_path} ({len(result)} bytes)")
                return result
            else:
                raise Exception("Download failed - no result returned")
                
        except Exception as e:
            logger.error(f"❌ File download failed: {e}")
            raise Exception(f"File download failed: {str(e)}")
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete CV file from Supabase storage"""
        try:
            if not file_path:
                raise Exception("File path is required for deletion")
            
            logger.info(f"🗑️ Deleting file: {file_path}")
            
            # Delete from Supabase storage
            result = self.client.storage.from_(self.bucket_name).remove([file_path])
            
            logger.info(f"✅ File deleted successfully: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ File deletion failed: {e}")
            raise Exception(f"File deletion failed: {str(e)}")
    
    async def list_files(self, path: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """List files in storage bucket"""
        try:
            logger.info(f"📁 Listing files in path: {path or 'root'}")
            
            # List files from Supabase storage
            result = self.client.storage.from_(self.bucket_name).list(
                path,
                {
                    "limit": limit,
                    "sortBy": {"column": "created_at", "order": "desc"}
                }
            )
            
            files = result or []
            logger.info(f"✅ Listed {len(files)} files")
            return files
            
        except Exception as e:
            logger.error(f"❌ Error listing files: {e}")
            return []
    
    async def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file information without downloading"""
        try:
            files = await self.list_files()
            
            # Find file by path
            for file_info in files:
                if file_path.endswith(file_info.get("name", "")):
                    return {
                        "name": file_info.get("name"),
                        "size": file_info.get("metadata", {}).get("size"),
                        "content_type": file_info.get("metadata", {}).get("mimetype"),
                        "created_at": file_info.get("created_at"),
                        "updated_at": file_info.get("updated_at")
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting file info: {e}")
            return None
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in storage"""
        try:
            file_info = await self.get_file_info(file_path)
            return file_info is not None
        except Exception:
            return False
    
    async def get_file_url(self, file_path: str, expires_in: int = 3600) -> Optional[str]:
        """Get signed URL for file access"""
        try:
            if not file_path:
                return None
            
            # Create signed URL for private file access
            result = self.client.storage.from_(self.bucket_name).create_signed_url(
                file_path, expires_in
            )
            
            if result:
                return result.get("signedURL")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error creating signed URL: {e}")
            return None
    
    async def validate_file(
        self, 
        file_content: bytes, 
        filename: str, 
        content_type: str
    ) -> Dict[str, Any]:
        """Validate file before upload"""
        errors = []
        
        # Check file size
        if len(file_content) == 0:
            errors.append("File is empty")
        elif len(file_content) > settings.max_file_size:
            errors.append(
                f"File size ({self._format_file_size(len(file_content))}) "
                f"exceeds maximum allowed ({self._format_file_size(settings.max_file_size)})"
            )
        
        # Check file type
        if content_type not in settings.allowed_file_types:
            errors.append(f"File type '{content_type}' not allowed. Supported types: PDF, DOC, DOCX")
        
        # Check filename
        if not filename or len(filename) > 255:
            errors.append("Invalid filename")
        
        # Check file extension matches content type
        expected_extension = self._get_file_extension(filename, content_type)
        if not filename.lower().endswith(expected_extension):
            errors.append("File extension doesn't match content type")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "file_info": {
                "size": len(file_content),
                "type": content_type,
                "extension": expected_extension
            }
        }
    
    async def cleanup_orphaned_files(self) -> Dict[str, Any]:
        """Clean up files that don't have corresponding database records"""
        try:
            from app.database.connection import database
            
            logger.info("🧹 Starting cleanup of orphaned files...")
            
            # Get all files in storage
            all_files = await self.list_files()
            
            # Get all consultant file paths from database
            consultants = await database.get_all_consultants()
            db_file_paths = {c.get('cv_file_path') for c in consultants if c.get('cv_file_path')}
            
            # Find orphaned files
            orphaned_files = []
            for file_info in all_files:
                file_name = file_info.get("name", "")
                # Check if any database file path ends with this file name
                is_referenced = any(
                    db_path and db_path.endswith(file_name) 
                    for db_path in db_file_paths
                )
                
                if not is_referenced:
                    orphaned_files.append(file_name)
            
            # Delete orphaned files
            deleted_count = 0
            errors = []
            
            for file_name in orphaned_files:
                try:
                    await self.delete_file(file_name)
                    deleted_count += 1
                except Exception as e:
                    error_msg = f"Could not delete orphaned file {file_name}: {e}"
                    logger.warning(f"⚠️ {error_msg}")
                    errors.append(error_msg)
            
            result = {
                "total_files": len(all_files),
                "orphaned_files": len(orphaned_files),
                "deleted_files": deleted_count,
                "errors": errors
            }
            
            logger.info(f"✅ Cleanup completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_files(self, days_old: int = None) -> Dict[str, Any]:
        """Clean up files older than specified days"""
        try:
            if days_old is None:
                days_old = settings.max_file_age_days
            
            logger.info(f"🧹 Starting cleanup of files older than {days_old} days...")
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Get all files
            all_files = await self.list_files()
            
            old_files = []
            for file_info in all_files:
                try:
                    created_at_str = file_info.get("created_at")
                    if created_at_str:
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        if created_at < cutoff_date:
                            old_files.append(file_info.get("name"))
                except Exception:
                    continue
            
            # Delete old files
            deleted_count = 0
            errors = []
            
            for file_name in old_files:
                try:
                    await self.delete_file(file_name)
                    deleted_count += 1
                except Exception as e:
                    error_msg = f"Could not delete old file {file_name}: {e}"
                    logger.warning(f"⚠️ {error_msg}")
                    errors.append(error_msg)
            
            result = {
                "total_files": len(all_files),
                "old_files": len(old_files),
                "deleted_files": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "errors": errors
            }
            
            logger.info(f"✅ Old file cleanup completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Old file cleanup failed: {e}")
            return {"error": str(e)}
    
    def _get_file_extension(self, filename: str, content_type: str) -> str:
        """Get appropriate file extension based on content type"""
        if content_type == 'application/pdf':
            return '.pdf'
        elif content_type == 'application/msword':
            return '.doc'
        elif content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return '.docx'
        else:
            # Fallback to filename extension
            if '.' in filename:
                return '.' + filename.split('.')[-1].lower()
            return '.pdf'  # Default fallback
    
    def _format_file_size(self, bytes_size: int) -> str:
        """Format file size in human readable format"""
        if bytes_size == 0:
            return '0 Bytes'
        
        k = 1024
        sizes = ['Bytes', 'KB', 'MB', 'GB']
        i = min(len(sizes) - 1, int((bytes_size.bit_length() - 1) // 10))
        
        return f"{bytes_size / (k ** i):.1f} {sizes[i]}"
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        try:
            files = await self.list_files()
            
            total_size = 0
            file_types = {}
            
            for file_info in files:
                # Calculate total size
                size = file_info.get("metadata", {}).get("size", 0)
                if isinstance(size, (int, float)):
                    total_size += size
                
                # Count file types
                name = file_info.get("name", "")
                if '.' in name:
                    ext = '.' + name.split('.')[-1].lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            return {
                "total_files": len(files),
                "total_size_bytes": total_size,
                "total_size_formatted": self._format_file_size(total_size),
                "file_types": file_types,
                "average_file_size": total_size // len(files) if files else 0,
                "bucket_name": self.bucket_name
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting storage stats: {e}")
            return {"error": str(e)}
    
    async def batch_upload_files(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upload multiple files in batch"""
        try:
            results = []
            successful = 0
            failed = 0
            
            for file_data in files:
                try:
                    file_path = await self.upload_file(
                        file_data["content"],
                        file_data["consultant_id"],
                        file_data["filename"],
                        file_data["content_type"]
                    )
                    
                    results.append({
                        "filename": file_data["filename"],
                        "success": True,
                        "file_path": file_path
                    })
                    successful += 1
                    
                except Exception as e:
                    results.append({
                        "filename": file_data["filename"],
                        "success": False,
                        "error": str(e)
                    })
                    failed += 1
            
            return {
                "total": len(files),
                "successful": successful,
                "failed": failed,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"❌ Batch upload failed: {e}")
            return {"error": str(e)}

# Global file storage instance
file_storage = FileStorageService()