# app/main.py - Optimized FastAPI Application

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List , Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

# Internal imports
from app.config import settings
from app.models.schemas import *
from app.services.ai_client import ai_client
from app.services.cv_processor import cv_processor
from app.services.project_matcher import project_matcher
from app.services.file_storage import file_storage
from app.database.connection import database
from app.utils.logger import setup_logging, get_logger
from app.utils.validators import validate_file_upload

# Setup logging
setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Consultant Matchmaker v2.0")
    logger.info(f"🌐 Environment: {settings.environment}")
    logger.info(f"🤖 AI Provider: {settings.ai_provider} ({settings.get_ai_config()['model']})")
    
    try:
        # Test connections
        await test_all_connections()
        logger.info("✅ All startup checks completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Consultant Matchmaker v2.0")

async def test_all_connections():
    """Test all system connections"""
    
    # Test database
    db_status = await database.test_connection()
    if not db_status:
        raise Exception("Database connection failed")
    logger.info("✅ Database connected")
    
    # Test AI model
    ai_status = await ai_client.test_connection()
    if not ai_status["success"]:
        logger.warning(f"⚠️ AI connection issue: {ai_status['error']}")
    else:
        logger.info(f"✅ AI connected: {ai_status['model']} ({ai_status.get('response_time', 0):.2f}s)")
    
    # Test file storage
    storage_status = await file_storage.test_connection()
    if not storage_status:
        raise Exception("File storage connection failed")
    logger.info("✅ File storage connected")

# Create FastAPI application
app = FastAPI(
    title="Consultant Matchmaker API v2.0",
    description="AI-powered consultant matching system with Qwen 2.5",
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": process_time,
            "client_ip": request.client.host if request.client else "unknown"
        }
    )
    
    return response

# ==========================================
# DOCUMENTATION ENDPOINTS
# ==========================================

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - Interactive Docs",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """ReDoc documentation"""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - Documentation",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

# ==========================================
# HEALTH & STATUS ENDPOINTS
# ==========================================

@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint"""
    return APIResponse(
        success=True,
        message="Consultant Matchmaker API v2.0",
        data={
            "version": "2.0.0",
            "status": "running",
            "ai_provider": settings.ai_provider,
            "environment": settings.environment,
            "docs": "/docs",
            "health": "/health"
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    
    start_time = time.time()
    
    try:
        # Test all connections
        db_connected = await database.test_connection()
        ai_result = await ai_client.test_connection()
        ai_connected = ai_result["success"]
        storage_connected = await file_storage.test_connection()
        
        # Determine overall status
        if db_connected and ai_connected and storage_connected:
            status = "healthy"
        elif db_connected:
            status = "degraded"
        else:
            status = "unhealthy"
        
        response_time = time.time() - start_time
        
        return HealthResponse(
            status=status,
            timestamp=datetime.utcnow(),
            version="2.0.0",
            environment=settings.environment,
            ai_provider=settings.ai_provider,
            ai_model=settings.get_ai_config()["model"],
            database_connected=db_connected,
            ai_connected=ai_connected,
            cache_connected=storage_connected,  # Using storage as cache proxy
            response_time=response_time
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="2.0.0",
            environment=settings.environment,
            ai_provider=settings.ai_provider,
            ai_model=settings.get_ai_config()["model"],
            database_connected=False,
            ai_connected=False,
            cache_connected=False,
            response_time=time.time() - start_time
        )

@app.get("/test-ai", response_model=APIResponse)
async def test_ai_connection():
    """Test AI model connection and performance"""
    
    try:
        result = await ai_client.test_connection()
        
        return APIResponse(
            success=result["success"],
            message="AI connection test completed",
            data=result
        )
        
    except Exception as e:
        logger.error(f"❌ AI test failed: {e}")
        return APIResponse(
            success=False,
            message="AI test failed",
            errors=[str(e)]
        )

# ==========================================
# CONSULTANT ENDPOINTS
# ==========================================

@app.post("/consultants/upload", response_model=UploadResponse)
async def upload_cv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prior_engagement: bool = Form(False)
):
    """Upload CV file and queue for processing"""
    
    try:
        logger.info(f"📤 CV upload started: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_content = await file.read()
        validation = validate_file_upload(len(file_content), file.filename, file.content_type)
        
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail="; ".join(validation["errors"]))
        
        logger.info(f"📄 File validated: {file.filename} ({len(file_content)} bytes)")
        
        # Create consultant record
        consultant_data = {
            "name": f"Processing CV {int(time.time())}",
            "prior_engagement": prior_engagement,
            "processing_status": "pending"
        }
        
        consultant = await database.create_consultant(consultant_data)
        logger.info(f"✅ Created consultant record: {consultant['id'][:8]}...")
        
        # Upload file to storage
        file_path = await file_storage.upload_file(
            file_content, consultant['id'], file.filename, file.content_type
        )
        
        # Update consultant with file path
        await database.update_consultant(consultant['id'], {
            "cv_file_path": file_path,
            "processing_status": "pending"
        })
        
        logger.info(f"🎉 CV upload completed successfully!")
        
        return UploadResponse(
            success=True,
            consultant_id=consultant['id'],
            message="CV uploaded successfully and queued for processing.",
            file_info={
                "filename": file.filename,
                "size": len(file_content),
                "content_type": file.content_type,
                "file_path": file_path
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/consultants", response_model=List[ConsultantResponse])
async def get_consultants(
    status: Optional[str] = None,
    prior_engagement: Optional[bool] = None,
    limit: int = 100
):
    """Get all consultants with optional filters"""
    
    try:
        filters = ConsultantFilters(
            status=status,
            prior_engagement=prior_engagement,
            limit=limit
        )
        
        consultants = await database.get_consultants(filters)
        
        logger.info(f"📋 Retrieved {len(consultants)} consultants")
        
        return [ConsultantResponse(**consultant) for consultant in consultants]
        
    except Exception as e:
        logger.error(f"❌ Error fetching consultants: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch consultants: {str(e)}")

@app.get("/consultants/{consultant_id}", response_model=ConsultantResponse)
async def get_consultant(consultant_id: str):
    """Get consultant by ID"""
    
    try:
        consultant = await database.get_consultant_by_id(consultant_id)
        
        if not consultant:
            raise HTTPException(status_code=404, detail="Consultant not found")
        
        return ConsultantResponse(**consultant)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching consultant {consultant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch consultant: {str(e)}")

@app.get("/consultants/{consultant_id}/status", response_model=ConsultantStatusResponse)
async def get_consultant_status(consultant_id: str):
    """Get consultant processing status"""
    
    try:
        consultant = await database.get_consultant_by_id(consultant_id)
        
        if not consultant:
            raise HTTPException(status_code=404, detail="Consultant not found")
        
        # Calculate derived fields
        has_data = bool(
            consultant.get("experience_years") or 
            consultant.get("skills") or 
            consultant.get("qualifications")
        )
        
        experience_total = 0.0
        exp_data = consultant.get("experience_years")
        if isinstance(exp_data, dict):
            experience_total = exp_data.get("total", 0.0)
        
        skills_count = 0
        skills_data = consultant.get("skills")
        if isinstance(skills_data, dict):
            technical_skills = skills_data.get("technical", [])
            domain_skills = skills_data.get("domain", [])
            skills_count = len(technical_skills) + len(domain_skills)
        
        qualifications_count = 0
        quals_data = consultant.get("qualifications")
        if isinstance(quals_data, dict):
            degrees = quals_data.get("degrees", [])
            certs = quals_data.get("certifications", [])
            licenses = quals_data.get("licenses", [])
            qualifications_count = len(degrees) + len(certs) + len(licenses)
        
        confidence_score = None
        metadata = consultant.get("processing_metadata")
        if isinstance(metadata, dict):
            confidence_score = metadata.get("confidence_score")
        
        return ConsultantStatusResponse(
            success=True,
            consultant_id=consultant_id,
            name=consultant.get("name", "Unknown"),
            processing_status=consultant["processing_status"],
            created_at=consultant["created_at"],
            updated_at=consultant["updated_at"],
            extraction_errors=consultant.get("extraction_errors"),
            has_data=has_data,
            email=consultant.get("email"),
            location=consultant.get("location"),
            experience_years=experience_total,
            skills_count=skills_count,
            qualifications_count=qualifications_count,
            confidence_score=confidence_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting consultant status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get consultant status: {str(e)}")

@app.post("/consultants/{consultant_id}/retry", response_model=APIResponse)
async def retry_consultant_processing(consultant_id: str):
    """Retry processing for a failed consultant"""
    
    try:
        consultant = await database.get_consultant_by_id(consultant_id)
        
        if not consultant:
            raise HTTPException(status_code=404, detail="Consultant not found")
        
        if consultant['processing_status'] != 'failed':
            raise HTTPException(
                status_code=400,
                detail=f"Cannot retry consultant with status: {consultant['processing_status']}"
            )
        
        # Reset to pending status
        await database.update_consultant(consultant_id, {
            "processing_status": "pending",
            "extraction_errors": {
                "manual_retry": True,
                "retry_timestamp": datetime.utcnow().isoformat()
            }
        })
        
        logger.info(f"🔄 Manual retry triggered for consultant: {consultant['name']}")
        
        return APIResponse(
            success=True,
            message=f"Consultant {consultant['name']} queued for retry processing",
            data={"consultant_id": consultant_id, "new_status": "pending"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error triggering retry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger retry: {str(e)}")

@app.get("/consultants/available/matching", response_model=List[ConsultantResponse])
async def get_available_consultants():
    """Get consultants available for project matching"""
    
    try:
        consultants = await database.get_available_consultants()
        
        logger.info(f"📋 Found {len(consultants)} consultants available for matching")
        
        return [ConsultantResponse(**consultant) for consultant in consultants]
        
    except Exception as e:
        logger.error(f"❌ Error fetching available consultants: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch available consultants: {str(e)}")

# ==========================================
# PROJECT MATCHING ENDPOINTS
# ==========================================

@app.post("/projects/match", response_model=MatchProjectResponse)
async def match_project(request: ProjectMatchRequest):
    """Match consultants to a project using AI analysis"""
    
    start_time = time.time()
    
    try:
        logger.info(f"🎯 Project matching request received")
        logger.info(f"📋 Title: {request.title}")
        logger.info(f"📄 Description length: {len(request.description)} characters")
        
        # Get available consultants
        consultants = await database.get_available_consultants()
        
        if not consultants:
            return MatchProjectResponse(
                success=True,
                matches=[],
                total_consultants=0,
                message="No consultants available for matching. Upload some CVs first."
            )
        
        logger.info(f"👥 Found {len(consultants)} available consultants")
        
        # Create project record
        project_data = {
            "title": request.title or "Untitled Project",
            "description": request.description.strip()
        }
        project = await database.create_project(project_data)
        
        # Perform matching
        match_data = {
            "description": request.description.strip(),
            "title": request.title or "Untitled Project",
            "consultants": consultants,
            "max_matches": request.max_matches,
            "min_score": request.min_score
        }
        
        result = await project_matcher.match_project(match_data)
        
        # Filter by minimum score
        filtered_matches = [
            match for match in result["matches"] 
            if match["match_score"] >= request.min_score
        ]
        
        # Limit results
        limited_matches = filtered_matches[:request.max_matches]
        
        # Transform to response format
        consultant_matches = []
        for match in limited_matches:
            consultant = next(
                (c for c in consultants if c["id"] == match["consultant_id"]), 
                None
            )
            if consultant:
                match_reasons = MatchReason(**match["match_reasons"])
                consultant_matches.append(ConsultantMatch(
                    consultant=ConsultantResponse(**consultant),
                    match_score=match["match_score"],
                    match_reasons=match_reasons
                ))
        
        total_time = time.time() - start_time
        
        logger.info(f"✅ Project matching completed in {total_time:.2f}s")
        logger.info(f"🎯 Generated {len(consultant_matches)} matches")
        
        return MatchProjectResponse(
            success=True,
            matches=consultant_matches,
            project_id=project["id"],
            processing_time=total_time,
            total_consultants=len(consultants),
            message=f"Found {len(consultant_matches)} consultant matches (filtered from {len(consultants)} total)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Project matching error: {e}")
        
        return MatchProjectResponse(
            success=False,
            matches=[],
            processing_time=total_time,
            total_consultants=0,
            message=f"Matching failed: {str(e)}",
            debug_info={
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )

# ==========================================
# ADMIN ENDPOINTS
# ==========================================

@app.post("/admin/clear-database", response_model=APIResponse)
async def clear_database(request: AdminClearRequest):
    """Clear all data from database and storage"""
    
    try:
        logger.warning(f"🧹 Database clear operation initiated")
        
        # Get all consultants
        consultants = await database.get_all_consultants()
        
        # Delete files
        deleted_files = 0
        for consultant in consultants:
            if consultant.get('cv_file_path'):
                try:
                    await file_storage.delete_file(consultant['cv_file_path'])
                    deleted_files += 1
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete file {consultant['cv_file_path']}: {e}")
        
        # Delete consultant records
        deleted_consultants = await database.delete_all_consultants()
        
        # Delete projects
        deleted_projects = await database.delete_all_projects()
        
        logger.info(f"✅ Clear operation completed")
        
        return APIResponse(
            success=True,
            message="Database cleared successfully! All data has been permanently deleted.",
            data={
                "consultants_deleted": deleted_consultants,
                "projects_deleted": deleted_projects,
                "files_deleted": deleted_files
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Critical error during database clearing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Critical error occurred during database clearing: {str(e)}"
        )

@app.get("/admin/stats", response_model=SystemStats)
async def get_system_stats():
    """Get comprehensive system statistics"""
    
    try:
        consultants = await database.get_all_consultants()
        
        consultant_stats = {
            "total": len(consultants),
            "pending": len([c for c in consultants if c['processing_status'] == 'pending']),
            "processing": len([c for c in consultants if c['processing_status'] == 'processing']),
            "completed": len([c for c in consultants if c['processing_status'] == 'completed']),
            "failed": len([c for c in consultants if c['processing_status'] == 'failed']),
        }
        
        # Get AI stats
        ai_test = await ai_client.test_connection()
        ai_stats = {
            "connected": ai_test["success"],
            "provider": settings.ai_provider,
            "model": settings.get_ai_config()["model"],
            "response_time": ai_test.get("response_time", 0)
        }
        
        # Get storage stats
        storage_stats = await file_storage.get_storage_stats()
        
        # Recent consultants
        recent_consultants = [
            {
                "id": c['id'][:8] + "...",
                "name": c.get('name', 'Unknown'),
                "status": c['processing_status'],
                "created_at": c['created_at']
            }
            for c in sorted(consultants, key=lambda x: x['created_at'], reverse=True)[:5]
        ]
        
        # System metrics
        import psutil
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "uptime": time.time() - psutil.boot_time()
        }
        
        return SystemStats(
            consultants=consultant_stats,
            storage=storage_stats,
            ai_stats=ai_stats,
            recent_consultants=recent_consultants,
            system_metrics=system_metrics
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting system stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system statistics: {str(e)}"
        )

# ==========================================
# ERROR HANDLERS
# ==========================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=APIResponse(
            success=False,
            message="Internal server error",
            errors=[str(exc) if settings.debug else "Something went wrong"]
        ).dict()
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """404 handler"""
    return JSONResponse(
        status_code=404,
        content=APIResponse(
            success=False,
            message=f"Route {request.url.path} not found"
        ).dict()
    )

# ==========================================
# STARTUP MESSAGE
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )