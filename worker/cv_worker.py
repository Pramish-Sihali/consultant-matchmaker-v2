# worker/cv_worker.py - Enhanced Worker with Phased Processing

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Set, Optional, List
from app.config import settings
from app.database.connection import database
from app.services.file_storage import file_storage
from app.services.cv_processor import cv_processor
from app.services.ai_client import ai_client
from app.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

class PhasedCVProcessingWorker:
    """Enhanced background worker with phased processing support"""
    
    def __init__(self):
        self.is_running = False
        self.active_jobs: Dict[str, str] = {}  # consultant_id -> phase
        self.retry_count: Dict[str, int] = {}
        self.max_concurrent_jobs = settings.max_concurrent_jobs
        self.polling_interval = settings.polling_interval
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
        self.processed_today = 0
        self.phase_1_completed = 0
        self.phase_2_completed = 0
        self.start_time = time.time()
        
    async def start(self):
        """Start the phased CV processing worker"""
        
        if self.is_running:
            logger.warning("Worker already running")
            return
        
        self.is_running = True
        logger.info("🚀 Phased CV Processing Worker v2.0 started")
        logger.info(f"📊 Config: {self.max_concurrent_jobs} concurrent jobs, {self.polling_interval}s polling")
        logger.info(f"🤖 AI Model: {settings.get_ai_config()['model']} via {settings.ai_provider}")
        logger.info("🔄 Phases: 1) Quick Extract (2-5s) → 2) AI Analysis (2+ min)")
        
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
        logger.info(f"🛑 Phased CV Processing Worker stopped")
        logger.info(f"📊 Session stats: {self.phase_1_completed} quick extracts, {self.phase_2_completed} AI analyses")
        logger.info(f"⏱️ Uptime: {uptime/3600:.1f}h")
    
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
                logger.warning("Worker will continue but AI analysis will fail until AI is available")
            
            # Check work distribution
            pending = [c for c in consultants if c['processing_status'] == 'pending']
            partial = [c for c in consultants if c.get('processing_phase') == 'partially_processed']
            analyzing = [c for c in consultants if c.get('processing_phase') == 'analyzing']
            failed = [c for c in consultants if c['processing_status'] == 'failed']
            
            logger.info(f"📋 Work queue:")
            logger.info(f"   {len(pending)} pending (need Phase 1 - Quick Extract)")
            logger.info(f"   {len(partial)} partially processed (need Phase 2 - AI Analysis)")
            logger.info(f"   {len(analyzing)} analyzing (retry AI Analysis)")
            if failed:
                logger.info(f"   {len(failed)} failed (available for retry)")
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            raise
    
    async def _processing_loop(self):
        """Main processing loop with phased approach"""
        
        logger.info("🔄 Starting phased processing loop...")
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                # Check for work in both phases
                await self._check_for_phase_1_work()  # Quick extractions
                await self._check_for_phase_2_work()  # AI analysis
                
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
    
    async def _check_for_phase_1_work(self):
        """Check for consultants needing Phase 1 (Quick Extract)"""
        
        # Don't check if we're at max capacity
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            return
        
        try:
            # Get consultants needing Phase 1
            pending_consultants = await database.get_consultants_by_status("pending")
            
            if not pending_consultants:
                return
            
            logger.info(f"📋 Found {len(pending_consultants)} consultants needing Phase 1 (Quick Extract)")
            
            # Process Phase 1 up to our concurrent limit
            available_slots = self.max_concurrent_jobs - len(self.active_jobs)
            consultants_to_process = pending_consultants[:available_slots]
            
            for consultant in consultants_to_process:
                if consultant.get('cv_file_path'):
                    consultant_id = consultant['id']
                    if consultant_id not in self.active_jobs:
                        logger.info(f"🎯 Queuing Phase 1: {consultant['name']} ({consultant_id[:8]}...)")
                        asyncio.create_task(self._process_phase_1_async(consultant))
                else:
                    logger.warning(f"⚠️ Consultant {consultant['id'][:8]}... has no file path, marking as failed")
                    await database.update_consultant(consultant['id'], {
                        "processing_status": "failed",
                        "processing_phase": "failed",
                        "extraction_errors": {
                            "error": "No CV file found",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    })
        
        except Exception as e:
            logger.error(f"❌ Error checking for Phase 1 work: {e}")
    
    async def _check_for_phase_2_work(self):
        """Check for consultants needing Phase 2 (AI Analysis)"""
        
        # Don't check if we're at max capacity
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            return
        
        try:
            # Get consultants needing Phase 2
            # This includes: partially_processed and analyzing (retry)
            all_consultants = await database.get_all_consultants()
            
            phase_2_candidates = [
                c for c in all_consultants 
                if c.get('processing_phase') in ['partially_processed', 'analyzing'] 
                and c.get('extracted_text')  # Must have extracted text
                and c['id'] not in self.active_jobs  # Not already being processed
            ]
            
            if not phase_2_candidates:
                return
            
            logger.info(f"🤖 Found {len(phase_2_candidates)} consultants needing Phase 2 (AI Analysis)")
            
            # Process Phase 2 up to our concurrent limit
            available_slots = self.max_concurrent_jobs - len(self.active_jobs)
            consultants_to_process = phase_2_candidates[:available_slots]
            
            for consultant in consultants_to_process:
                consultant_id = consultant['id']
                logger.info(f"🎯 Queuing Phase 2: {consultant['name']} ({consultant_id[:8]}...)")
                asyncio.create_task(self._process_phase_2_async(consultant))
        
        except Exception as e:
            logger.error(f"❌ Error checking for Phase 2 work: {e}")
    
    async def _process_phase_1_async(self, consultant: Dict):
        """Process Phase 1 (Quick Extract) asynchronously"""
        
        consultant_id = consultant['id']
        start_time = time.time()
        
        # Add to active jobs
        self.active_jobs[consultant_id] = 'phase_1'
        
        try:
            logger.info(f"\n🚀 Phase 1 starting: {consultant['name']}")
            logger.info(f"📋 ID: {consultant_id[:8]}... | File: {consultant['cv_file_path']}")
            
            # Update status to extracting
            await database.update_consultant(consultant_id, {
                "processing_status": "processing",
                "processing_phase": "extracting"
            })
            
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
            
            # Process Phase 1 (Quick Extract)
            logger.info(f"⚡ Starting Phase 1 (Quick Extract)...")
            processing_start = time.time()
            
            result = await cv_processor.process_cv_phased(
                file_content=file_content,
                filename=filename,
                mime_type=mime_type,
                consultant_id=consultant_id,
                phase='quick_extract',
                progress_callback=progress_callback
            )
            
            processing_time = time.time() - processing_start
            
            logger.info(f"✅ Phase 1 completed in {processing_time:.2f}s")
            
            # Update consultant with Phase 1 results
            update_data = result['consultant_updates']
            await database.update_consultant(consultant_id, update_data)
            
            total_time = time.time() - start_time
            self.phase_1_completed += 1
            
            # Log success summary
            basic_info = result['basic_info']
            logger.info(f"🎉 PHASE 1 SUCCESS: {basic_info['name']}")
            logger.info(f"⏱️ Total time: {total_time:.2f}s | Email: {basic_info['email']} | Exp: {basic_info['basic_experience']['total_years']}y")
            logger.info(f"📊 Today's Phase 1: {self.phase_1_completed}")
            logger.info(f"🔄 Ready for Phase 2 (AI Analysis)\n")
            
            # Reset retry count on success
            if consultant_id in self.retry_count:
                del self.retry_count[consultant_id]
        
        except Exception as e:
            await self._handle_processing_error(consultant, e, start_time, phase='phase_1')
        
        finally:
            # Remove from active jobs
            if consultant_id in self.active_jobs:
                del self.active_jobs[consultant_id]
    
    async def _process_phase_2_async(self, consultant: Dict):
        """Process Phase 2 (AI Analysis) asynchronously"""
        
        consultant_id = consultant['id']
        start_time = time.time()
        
        # Add to active jobs
        self.active_jobs[consultant_id] = 'phase_2'
        
        try:
            logger.info(f"\n🤖 Phase 2 starting: {consultant['name']}")
            logger.info(f"📋 ID: {consultant_id[:8]}... | Has extracted text: {bool(consultant.get('extracted_text'))}")
            
            # Update status to analyzing
            await database.update_consultant(consultant_id, {
                "processing_status": "processing",
                "processing_phase": "analyzing"
            })
            
            # Create progress callback
            def progress_callback(step: str, progress: int, message: str):
                logger.info(f"📊 {consultant['name'][:20]}...: {step} - {progress}% - {message}")
            
            # Process Phase 2 (AI Analysis)
            logger.info(f"🤖 Starting Phase 2 (AI Analysis) with {settings.get_ai_config()['model']}...")
            processing_start = time.time()
            
            result = await cv_processor.process_cv_phased(
                file_content=None,  # Not needed for Phase 2
                filename=consultant.get('name', 'unknown.pdf'),
                mime_type=None,  # Not needed for Phase 2
                consultant_id=consultant_id,
                phase='ai_analysis',
                extracted_text=consultant['extracted_text'],
                progress_callback=progress_callback
            )
            
            processing_time = time.time() - processing_start
            
            logger.info(f"✅ Phase 2 completed in {processing_time:.2f}s")
            
            # Update consultant with Phase 2 results
            update_data = result['consultant_updates']
            await database.update_consultant(consultant_id, update_data)
            
            total_time = time.time() - start_time
            self.phase_2_completed += 1
            self.processed_today += 1
            
            # Log success summary
            full_analysis = result['full_analysis']
            exp_total = full_analysis.get('experience_years', {}).get('total', 0)
            skills_count = len(full_analysis.get('skills', {}).get('technical', [])) + \
                          len(full_analysis.get('skills', {}).get('domain', []))
            quals_count = len(full_analysis.get('qualifications', {}).get('degrees', [])) + \
                          len(full_analysis.get('qualifications', {}).get('certifications', [])) + \
                          len(full_analysis.get('qualifications', {}).get('licenses', []))
            
            logger.info(f"🎉 PHASE 2 SUCCESS: {full_analysis.get('name', consultant['name'])}")
            logger.info(f"⏱️ Total time: {total_time:.2f}s | Experience: {exp_total}y | Skills: {skills_count} | Quals: {quals_count}")
            logger.info(f"📊 Today's Phase 2: {self.phase_2_completed} | Total completed: {self.processed_today}")
            logger.info(f"✅ FULLY PROCESSED - Available for matching!\n")
            
            # Reset retry count on success
            if consultant_id in self.retry_count:
                del self.retry_count[consultant_id]
        
        except Exception as e:
            await self._handle_processing_error(consultant, e, start_time, phase='phase_2')
        
        finally:
            # Remove from active jobs
            if consultant_id in self.active_jobs:
                del self.active_jobs[consultant_id]
    
    async def _handle_processing_error(self, consultant: Dict, error: Exception, start_time: float, phase: str):
        """Handle processing errors with retry logic"""
        
        consultant_id = consultant['id']
        total_time = time.time() - start_time
        
        logger.error(f"\n❌ {phase.upper()} FAILED: {consultant['name']}")
        logger.error(f"🔍 Error: {str(error)}")
        logger.error(f"⏱️ Failed after: {total_time:.2f}s")
        
        # Handle retries
        current_retries = self.retry_count.get(consultant_id, 0)
        
        if current_retries < self.max_retries:
            # Increment retry count and set appropriate status
            self.retry_count[consultant_id] = current_retries + 1
            
            if phase == 'phase_1':
                retry_status = "pending"
                retry_phase = "pending"
            else:  # phase_2
                retry_status = "processing"
                retry_phase = "partially_processed"  # Retry Phase 2
            
            await database.update_consultant(consultant_id, {
                "processing_status": retry_status,
                "processing_phase": retry_phase,
                "extraction_errors": {
                    "phase": phase,
                    "attempt": current_retries + 1,
                    "error": str(error),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "retry_delay": self.retry_delay
                }
            })
            
            logger.info(f"🔄 Retry {current_retries + 1}/{self.max_retries} scheduled for: {consultant['name']} ({phase})")
            
            # Wait before allowing retry
            await asyncio.sleep(self.retry_delay)
            
        else:
            # Max retries reached, mark as failed
            await database.update_consultant(consultant_id, {
                "processing_status": "failed",
                "processing_phase": "failed",
                "extraction_errors": {
                    "final_error": str(error),
                    "failed_phase": phase,
                    "retries_attempted": self.max_retries,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_time": total_time
                }
            })
            
            logger.error(f"💀 PERMANENTLY FAILED after {self.max_retries} retries: {consultant['name']} ({phase})\n")
            if consultant_id in self.retry_count:
                del self.retry_count[consultant_id]
    
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
            "active_job_phases": dict(self.active_jobs),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "polling_interval": self.polling_interval,
            "retry_queues": len(self.retry_count),
            "processed_today": self.processed_today,
            "phase_1_completed": self.phase_1_completed,
            "phase_2_completed": self.phase_2_completed,
            "uptime_hours": round(uptime / 3600, 1),
            "pending_retries": dict(self.retry_count),
            "ai_model": settings.get_ai_config()["model"],
            "ai_provider": settings.ai_provider,
            "processing_mode": "phased"
        }

# Global worker instance
worker = PhasedCVProcessingWorker()

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
        logger.info("🚀 Starting Phased CV Processing Worker v2.0...")
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