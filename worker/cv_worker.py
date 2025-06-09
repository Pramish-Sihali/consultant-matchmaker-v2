# worker/cv_worker.py - Optimized Background CV Processing Worker

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Set, Optional
from app.config import settings
from app.database.connection import database
from app.services.file_storage import file_storage
from app.services.cv_processor import cv_processor
from app.services.ai_client import ai_client
from app.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

class CVProcessingWorker:
    """Optimized background worker for processing CV files with Qwen 2.5"""
    
    def __init__(self):
        self.is_running = False
        self.active_jobs: Set[str] = set()
        self.retry_count: Dict[str, int] = {}
        self.max_concurrent_jobs = settings.max_concurrent_jobs
        self.polling_interval = settings.polling_interval
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
        self.processed_today = 0
        self.start_time = time.time()
        
    async def start(self):
        """Start the CV processing worker"""
        
        if self.is_running:
            logger.warning("Worker already running")
            return
        
        self.is_running = True
        logger.info("🚀 CV Processing Worker v2.0 started")
        logger.info(f"📊 Config: {self.max_concurrent_jobs} concurrent jobs, {self.polling_interval}s polling")
        logger.info(f"🤖 AI Model: {settings.get_ai_config()['model']} via {settings.ai_provider}")
        
        # Test system connections
        await self._test_system_connections()
        
        # Start the main processing loop
        await self._processing_loop()
    
    async def stop(self):
        """Stop the worker gracefully"""
        self.is_running = False
        
        # Wait for active jobs to complete (with timeout)
        timeout = 30
        start_time = time.time()
        
        while self.active_jobs and (time.time() - start_time) < timeout:
            logger.info(f"⏳ Waiting for {len(self.active_jobs)} active jobs to complete...")
            await asyncio.sleep(2)
        
        if self.active_jobs:
            logger.warning(f"⚠️ Timeout reached, {len(self.active_jobs)} jobs may be incomplete")
        
        uptime = time.time() - self.start_time
        logger.info(f"🛑 CV Processing Worker stopped (uptime: {uptime/3600:.1f}h, processed: {self.processed_today})")
    
    async def _test_system_connections(self):
        """Test all system connections on startup"""
        
        try:
            logger.info("🔍 Testing system connections...")
            
            # Test database
            db_success = await database.test_connection()
            if not db_success:
                raise Exception("Database connection failed")
            
            consultants = await database.get_all_consultants()
            logger.info(f"✅ Database connected. Found {len(consultants)} consultants total.")
            
            # Test file storage
            storage_success = await file_storage.test_connection()
            if not storage_success:
                raise Exception("File storage connection failed")
            
            storage_stats = await file_storage.get_storage_stats()
            logger.info(f"✅ Storage connected. {storage_stats.get('total_files', 0)} files in bucket.")
            
            # Test AI model
            ai_test = await ai_client.test_connection()
            if ai_test["success"]:
                logger.info(f"✅ AI connected: {ai_test['model']} ({ai_test.get('response_time', 0):.2f}s)")
            else:
                logger.warning(f"⚠️ AI connection failed: {ai_test.get('error', 'Unknown error')}")
                logger.warning("Worker will continue but CV processing will fail until AI is available")
            
            # Check pending work
            pending = [c for c in consultants if c['processing_status'] == 'pending']
            failed = [c for c in consultants if c['processing_status'] == 'failed']
            
            logger.info(f"📋 Found {len(pending)} pending consultants to process")
            if failed:
                logger.info(f"⚠️ Found {len(failed)} failed consultants (use admin endpoints to retry)")
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            raise
    
    async def _processing_loop(self):
        """Main processing loop with improved error handling"""
        
        logger.info("🔄 Starting processing loop...")
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                await self._check_for_pending_cvs()
                consecutive_errors = 0  # Reset on success
                await asyncio.sleep(self.polling_interval)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"❌ Error in processing loop (#{consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"💥 Too many consecutive errors ({consecutive_errors}), stopping worker")
                    self.is_running = False
                    break
                
                # Exponential backoff on errors
                error_delay = min(60, self.polling_interval * (2 ** consecutive_errors))
                logger.info(f"⏳ Waiting {error_delay}s before retry...")
                await asyncio.sleep(error_delay)
    
    async def _check_for_pending_cvs(self):
        """Check for pending CVs and process them"""
        
        # Don't check if we're at max capacity
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            return
        
        try:
            # Get pending CVs from database
            pending_consultants = await database.get_consultants_by_status("pending")
            
            if not pending_consultants:
                return  # No pending CVs
            
            logger.info(f"📋 Found {len(pending_consultants)} pending CVs")
            
            # Process CVs up to our concurrent limit
            available_slots = self.max_concurrent_jobs - len(self.active_jobs)
            consultants_to_process = pending_consultants[:available_slots]
            
            for consultant in consultants_to_process:
                if consultant.get('cv_file_path'):
                    logger.info(f"🎯 Queuing consultant: {consultant['name']} ({consultant['id'][:8]}...)")
                    # Start processing in background
                    asyncio.create_task(self._process_cv_async(consultant))
                else:
                    logger.warning(f"⚠️ Consultant {consultant['id'][:8]}... has no file path, marking as failed")
                    await database.update_consultant(consultant['id'], {
                        "processing_status": "failed",
                        "extraction_errors": {
                            "error": "No CV file found",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    })
        
        except Exception as e:
            logger.error(f"❌ Error checking for pending CVs: {e}")
    
    async def _process_cv_async(self, consultant: Dict):
        """Process a single CV asynchronously with comprehensive error handling"""
        
        job_id = consultant['id']
        start_time = time.time()
        
        # Add to active jobs
        self.active_jobs.add(job_id)
        
        try:
            logger.info(f"\n🔄 Starting processing: {consultant['name']}")
            logger.info(f"📋 ID: {job_id[:8]}... | File: {consultant['cv_file_path']}")
            
            # Update status to "processing"
            await database.update_consultant(consultant['id'], {
                "processing_status": "processing"
            })
            
            # Validate file path
            if not consultant.get('cv_file_path'):
                raise Exception('No CV file path found')
            
            # Download file from storage
            logger.info(f"📥 Downloading file...")
            download_start = time.time()
            
            file_content = await file_storage.download_file(consultant['cv_file_path'])
            
            if not file_content or len(file_content) == 0:
                raise Exception('Downloaded file is empty or invalid')
            
            download_time = time.time() - download_start
            logger.info(f"✅ Downloaded: {len(file_content)} bytes in {download_time:.2f}s")
            
            # Determine file type
            file_path = consultant['cv_file_path']
            filename = file_path.split('/')[-1]
            
            mime_type = self._get_mime_type_from_filename(filename)
            
            # Create progress callback
            def progress_callback(step: str, progress: int, message: str):
                logger.info(f"📊 {consultant['name'][:20]}...: {step} - {progress}% - {message}")
            
            # Process CV with AI
            logger.info(f"🤖 Processing with {settings.get_ai_config()['model']}...")
            processing_start = time.time()
            
            extracted_data = await cv_processor.process_cv(
                file_content, filename, mime_type, progress_callback
            )
            
            processing_time = time.time() - processing_start
            
            logger.info(f"✅ AI processing completed in {processing_time:.2f}s")
            logger.info(f"📋 Extracted: {extracted_data.get('name', 'Unknown')}")
            
            # Prepare update data
            update_data = {
                "name": extracted_data.get('name', consultant['name']),
                "email": extracted_data.get('email'),
                "location": extracted_data.get('location'),
                "experience_years": extracted_data.get('experience_years'),
                "qualifications": extracted_data.get('qualifications'),
                "skills": extracted_data.get('skills'),
                "processing_status": "completed",
                "extraction_errors": None,
                "processing_metadata": extracted_data.get('_processing_metadata')
            }
            
            # Update consultant with extracted data
            await database.update_consultant(consultant['id'], update_data)
            
            total_time = time.time() - start_time
            self.processed_today += 1
            
            # Log success summary
            exp_total = extracted_data.get('experience_years', {}).get('total', 0)
            skills_count = len(extracted_data.get('skills', {}).get('technical', [])) + \
                          len(extracted_data.get('skills', {}).get('domain', []))
            quals_count = len(extracted_data.get('qualifications', {}).get('degrees', [])) + \
                          len(extracted_data.get('qualifications', {}).get('certifications', [])) + \
                          len(extracted_data.get('qualifications', {}).get('licenses', []))
            
            logger.info(f"🎉 SUCCESS: {extracted_data.get('name', consultant['name'])}")
            logger.info(f"⏱️ Total time: {total_time:.2f}s | Experience: {exp_total}y | Skills: {skills_count} | Quals: {quals_count}")
            logger.info(f"📊 Today's count: {self.processed_today}\n")
            
            # Reset retry count on success
            if job_id in self.retry_count:
                del self.retry_count[job_id]
        
        except Exception as e:
            await self._handle_processing_error(consultant, e, start_time)
        
        finally:
            # Remove from active jobs
            if job_id in self.active_jobs:
                self.active_jobs.remove(job_id)
    
    async def _handle_processing_error(self, consultant: Dict, error: Exception, start_time: float):
        """Handle processing errors with retry logic"""
        
        job_id = consultant['id']
        total_time = time.time() - start_time
        
        logger.error(f"\n❌ PROCESSING FAILED: {consultant['name']}")
        logger.error(f"🔍 Error: {str(error)}")
        logger.error(f"⏱️ Failed after: {total_time:.2f}s")
        
        # Handle retries
        current_retries = self.retry_count.get(job_id, 0)
        
        if current_retries < self.max_retries:
            # Increment retry count and set back to pending
            self.retry_count[job_id] = current_retries + 1
            
            await database.update_consultant(consultant['id'], {
                "processing_status": "pending",
                "extraction_errors": {
                    "attempt": current_retries + 1,
                    "error": str(error),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "retry_delay": self.retry_delay
                }
            })
            
            logger.info(f"🔄 Retry {current_retries + 1}/{self.max_retries} scheduled for: {consultant['name']} (waiting {self.retry_delay}s)")
            
            # Wait before allowing retry
            await asyncio.sleep(self.retry_delay)
            
        else:
            # Max retries reached, mark as failed
            await database.update_consultant(consultant['id'], {
                "processing_status": "failed",
                "extraction_errors": {
                    "final_error": str(error),
                    "retries_attempted": self.max_retries,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_time": total_time
                }
            })
            
            logger.error(f"💀 PERMANENTLY FAILED after {self.max_retries} retries: {consultant['name']}\n")
            if job_id in self.retry_count:
                del self.retry_count[job_id]
    
    def _get_mime_type_from_filename(self, filename: str) -> str:
        """Get MIME type from filename"""
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.pdf'):
            return 'application/pdf'
        elif filename_lower.endswith('.docx'):
            return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        elif filename_lower.endswith('.doc'):
            return 'application/msword'
        else:
            return 'application/pdf'  # Default fallback
    
    def get_status(self) -> Dict:
        """Get comprehensive worker status"""
        
        uptime = time.time() - self.start_time
        
        return {
            "is_running": self.is_running,
            "active_jobs": len(self.active_jobs),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "polling_interval": self.polling_interval,
            "retry_queues": len(self.retry_count),
            "processed_today": self.processed_today,
            "uptime_hours": round(uptime / 3600, 1),
            "current_jobs": list(self.active_jobs),
            "pending_retries": dict(self.retry_count),
            "ai_model": settings.get_ai_config()["model"],
            "ai_provider": settings.ai_provider
        }

# Global worker instance
worker = CVProcessingWorker()

async def shutdown_handler(signame):
    """Handle shutdown signals gracefully"""
    logger.info(f"\n🛑 Received {signame}, shutting down gracefully...")
    await worker.stop()
    logger.info("✅ Worker shutdown complete")
    sys.exit(0)

async def main():
    """Main worker function"""
    
    # Setup signal handlers for graceful shutdown
    for signame in {'SIGINT', 'SIGTERM'}:
        if hasattr(signal, signame):
            asyncio.get_event_loop().add_signal_handler(
                getattr(signal, signame),
                lambda s=signame: asyncio.create_task(shutdown_handler(s))
            )
    
    try:
        logger.info("🚀 Starting CV Processing Worker v2.0...")
        await worker.start()
    except KeyboardInterrupt:
        await shutdown_handler("KeyboardInterrupt")
    except Exception as e:
        logger.error(f"💥 Worker failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Worker interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)