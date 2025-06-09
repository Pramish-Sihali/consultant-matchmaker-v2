# app/services/cv_processor.py - Optimized CV Processing Service

import asyncio
import io
import logging
import re
import time
from typing import Dict, Any, Optional, Callable, List
import PyPDF2
import docx
import mammoth
from app.services.ai_client import ai_client
from app.utils.validators import validate_extracted_data
from app.config import settings

logger = logging.getLogger(__name__)

class CVProcessor:
    """Optimized CV processing service for Qwen 2.5"""
    
    def __init__(self):
        self.supported_mime_types = settings.allowed_file_types
        self.max_file_size = settings.max_file_size
    
    async def process_cv(
        self, 
        file_content: bytes, 
        filename: str, 
        mime_type: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Main CV processing function with enhanced performance"""
        
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting CV processing: {filename}")
            
            if progress_callback:
                progress_callback("extraction", 10, "Starting text extraction...")
            
            # Step 1: Extract text from file
            extraction_start = time.time()
            extracted_text = await self._extract_text_from_file(
                file_content, filename, mime_type, progress_callback
            )
            extraction_time = time.time() - extraction_start
            
            if not extracted_text or len(extracted_text.strip()) < 50:
                raise Exception(
                    "Could not extract sufficient text from CV. "
                    "Please ensure the file is readable and contains text."
                )
            
            logger.info(f"✅ Text extracted in {extraction_time:.2f}s: {len(extracted_text)} characters")
            
            if progress_callback:
                progress_callback("extraction", 30, "Text extraction completed")
            
            # Step 2: Process with AI
            if progress_callback:
                progress_callback("ai_processing", 40, "Analyzing CV with Qwen 2.5...")
            
            ai_start = time.time()
            ai_analysis = await ai_client.analyze_cv_text(
                extracted_text, progress_callback
            )
            ai_time = time.time() - ai_start
            
            logger.info(f"🤖 AI analysis completed in {ai_time:.2f}s")
            logger.info(f"   Name: {ai_analysis.get('name', 'Unknown')}")
            logger.info(f"   Experience: {ai_analysis.get('experience_years', {}).get('total', 0)} years")
            
            if progress_callback:
                progress_callback("validation", 90, "Validating extracted data...")
            
            # Step 3: Validate and format results
            validation_start = time.time()
            validated_data = validate_extracted_data(ai_analysis)
            validation_time = time.time() - validation_start
            
            if progress_callback:
                progress_callback("complete", 100, "CV processing completed")
            
            total_time = time.time() - start_time
            
            # Add processing metadata
            validated_data["_processing_metadata"] = {
                "total_time": total_time,
                "extraction_time": extraction_time,
                "ai_time": ai_time,
                "validation_time": validation_time,
                "file_size": len(file_content),
                "text_length": len(extracted_text),
                "model": settings.get_ai_config()["model"],
                "provider": settings.ai_provider
            }
            
            logger.info(f"✅ CV processing completed in {total_time:.2f}s for: {validated_data.get('name', 'Unknown')}")
            
            return validated_data
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ CV processing error after {total_time:.2f}s: {e}")
            if progress_callback:
                progress_callback("error", 0, f"Error: {str(e)}")
            raise Exception(f"CV processing failed: {str(e)}")
    
    async def _extract_text_from_file(
        self, 
        file_content: bytes, 
        filename: str, 
        mime_type: str,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Extract text from different file formats"""
        
        try:
            if progress_callback:
                progress_callback("extraction", 15, f"Processing {mime_type} file...")
            
            if mime_type == 'application/pdf':
                text = await self._extract_from_pdf(file_content, progress_callback)
            elif mime_type in [
                'application/msword', 
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ]:
                text = await self._extract_from_word(file_content, mime_type, progress_callback)
            else:
                raise Exception(f"Unsupported file type: {mime_type}")
            
            # Clean and validate extracted text
            cleaned_text = self._clean_extracted_text(text)
            
            if len(cleaned_text) < 50:
                raise Exception("File appears to be empty or contains mostly images/scanned content")
            
            # Log sample for debugging
            sample = cleaned_text[:300].replace('\n', ' ')
            logger.info(f"📄 Text sample: {sample}...")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"❌ Text extraction failed for {filename}: {e}")
            raise Exception(f"Failed to extract text from {filename}: {str(e)}")
    
    async def _extract_from_pdf(
        self, 
        file_content: bytes, 
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Extract text from PDF with improved error handling"""
        
        try:
            if progress_callback:
                progress_callback("extraction", 20, "Reading PDF content...")
            
            file_stream = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(file_stream)
            
            if len(pdf_reader.pages) == 0:
                raise Exception("PDF has no pages")
            
            text_parts = []
            total_pages = len(pdf_reader.pages)
            successful_pages = 0
            
            for i, page in enumerate(pdf_reader.pages):
                if progress_callback:
                    progress = 20 + (i / total_pages) * 10  # 20% to 30%
                    progress_callback("extraction", progress, f"Extracting page {i+1}/{total_pages}...")
                
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
                        successful_pages += 1
                except Exception as page_error:
                    logger.warning(f"⚠️ Could not extract text from page {i+1}: {page_error}")
                    continue
            
            if not text_parts:
                raise Exception("Could not extract text from any PDF pages")
            
            full_text = '\n'.join(text_parts)
            logger.info(f"✅ PDF extraction: {len(full_text)} chars from {successful_pages}/{total_pages} pages")
            
            return full_text
            
        except Exception as e:
            raise Exception(f"PDF processing failed: {str(e)}")
    
    async def _extract_from_word(
        self, 
        file_content: bytes, 
        mime_type: str, 
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Extract text from Word documents"""
        
        try:
            if progress_callback:
                progress_callback("extraction", 20, "Reading Word document...")
            
            if mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                # DOCX file
                text = await self._extract_from_docx(file_content, progress_callback)
            else:
                # DOC file - use mammoth
                text = await self._extract_from_doc(file_content, progress_callback)
            
            if len(text.strip()) < 50:
                raise Exception("Word document appears to be empty")
            
            logger.info(f"✅ Word extraction: {len(text)} characters")
            return text
            
        except Exception as e:
            raise Exception(f"Word document processing failed: {str(e)}")
    
    async def _extract_from_docx(
        self, 
        file_content: bytes, 
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Extract text from DOCX files"""
        
        file_stream = io.BytesIO(file_content)
        doc = docx.Document(file_stream)
        
        text_parts = []
        total_paragraphs = len(doc.paragraphs)
        
        # Extract paragraphs
        for i, paragraph in enumerate(doc.paragraphs):
            if progress_callback and i % 20 == 0:  # Update every 20 paragraphs
                progress = 20 + (i / total_paragraphs) * 8
                progress_callback("extraction", progress, f"Processing paragraph {i+1}/{total_paragraphs}...")
            
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        # Extract tables
        if progress_callback:
            progress_callback("extraction", 28, "Extracting tables...")
        
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_parts.append(' | '.join(row_text))
        
        return '\n'.join(text_parts)
    
    async def _extract_from_doc(
        self, 
        file_content: bytes, 
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Extract text from DOC files using mammoth"""
        
        if progress_callback:
            progress_callback("extraction", 25, "Converting DOC file...")
        
        file_stream = io.BytesIO(file_content)
        result = mammoth.extract_raw_text(file_stream)
        
        # Log any warnings from mammoth
        if result.messages:
            for message in result.messages:
                logger.warning(f"⚠️ Mammoth warning: {message}")
        
        return result.value
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text for better AI processing"""
        
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive newlines but preserve structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up common PDF extraction artifacts
        text = re.sub(r'[^\w\s\n\-\.,;:@/()&%$#!?+="\']+', ' ', text)
        
        # Remove very short lines that are likely artifacts
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) >= 3:  # Keep lines with at least 3 characters
                cleaned_lines.append(line)
        
        # Rejoin and clean final whitespace
        text = '\n'.join(cleaned_lines)
        text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single space
        
        return text.strip()
    
    def is_supported_file_type(self, mime_type: str) -> bool:
        """Check if file type is supported"""
        return mime_type in self.supported_mime_types
    
    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions"""
        return ['.pdf', '.doc', '.docx']
    
    async def extract_text_preview(
        self, 
        file_content: bytes, 
        filename: str, 
        mime_type: str,
        max_length: int = 1000
    ) -> str:
        """Extract a preview of text without full processing"""
        
        try:
            full_text = await self._extract_text_from_file(file_content, filename, mime_type)
            preview = full_text[:max_length]
            
            if len(full_text) > max_length:
                preview += "..."
            
            return preview
            
        except Exception as e:
            logger.error(f"❌ Preview extraction failed: {e}")
            return f"Could not extract preview: {str(e)}"
    
    def estimate_processing_time(self, file_size: int, mime_type: str) -> Dict[str, Any]:
        """Estimate processing time based on file characteristics"""
        
        # Base times in seconds
        base_extraction_time = {
            'application/pdf': 2.0,
            'application/msword': 1.5,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 1.0
        }
        
        # Size factor (larger files take longer)
        size_mb = file_size / (1024 * 1024)
        size_factor = 1 + (size_mb * 0.5)  # +50% time per MB
        
        extraction_time = base_extraction_time.get(mime_type, 2.0) * size_factor
        ai_time = 10.0  # Qwen 2.5 average time
        validation_time = 0.5
        
        total_time = extraction_time + ai_time + validation_time
        
        return {
            "estimated_total_seconds": round(total_time, 1),
            "breakdown": {
                "extraction": round(extraction_time, 1),
                "ai_analysis": round(ai_time, 1),
                "validation": round(validation_time, 1)
            },
            "confidence": "medium" if size_mb < 5 else "low"
        }

# Global CV processor instance
cv_processor = CVProcessor()