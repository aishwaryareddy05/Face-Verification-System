from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
import base64
import json
import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import Optional
import os
from face_matcher import FaceMatcher
from datetime import datetime
from fastapi.responses import JSONResponse,RedirectResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Verification Service",
    description="Production-ready microservice for facial verification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Pydantic Models
class VerifyFaceRequest(BaseModel):
    user_id: str = Field(..., description="Unique student identifier", min_length=1)
    image_type: str = Field(
        ...
    )
    image_1_base64: str = Field(
        ..., 
        description="First image as base64 string (min 1000 bytes when decoded)", 
        min_length=100
    )
    image_2_base64: str = Field(
        ..., 
        description="Second image as base64 string (min 1000 bytes when decoded)", 
        min_length=100
    )
    
    @validator('image_type')
    def validate_image_type(cls, v):
        allowed_types = ['id_to_selfie', 'selfie_to_selfie']
        if v not in allowed_types:
            raise ValueError(f'image_type must be one of {allowed_types}')
        return v
    
    @validator('image_1_base64', 'image_2_base64')
    def validate_base64(cls, v):
        try:
            # Handle URL-safe base64 and padding issues
            v = v.strip()
            if '-' in v or '_' in v:
                v = v.replace('-', '+').replace('_', '/')
            
            # Add padding if needed
            pad_len = len(v) % 4
            if pad_len:
                v += '=' * (4 - pad_len)
            
            decoded = base64.b64decode(v)
            if len(decoded) < 1000:
                raise ValueError('Decoded image must be at least 1000 bytes')
            return v
        except Exception as e:
            raise ValueError(f'Invalid base64 image data: {str(e)}')

class VerifyFaceResponse(BaseModel):
    user_id: str
    match_score: float
    match: bool
    confidence_level: str
    image_type: str
    threshold: float
    status: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str

class VersionResponse(BaseModel):
    service_version: str
    api_version: str
    face_recognition_library: str

# Global variables
face_matcher = None
config = None

def load_config():
    """Load configuration from config.json"""
    default_config = {
        "thresholds": {
            "id_to_selfie": 0.65,
            "selfie_to_selfie": 0.82
        },
        "preprocessing": {
            "enable_histogram_equalization": True,
            "enable_sharpening": True,
            "target_size": [160, 160],
            "padding": 20
        },
        "face_detection": {
            "model": "hog",
            "fallback_to_cnn": True
        },
        "api": {
            "max_image_size_mb": 10,
            "timeout_seconds": 30
        }
    }
    
    try:
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
                logger.info("✅ Configuration loaded from config.json")
        else:
            logger.info("📝 Using default configuration (config.json not found)")
    except Exception as e:
        logger.warning(f"⚠️ Error loading config.json: {e}. Using defaults.")
    
    return default_config

def initialize_face_matcher():
    """Initialize the face matcher with configuration"""
    global face_matcher, config
    try:
        config = load_config()
        face_matcher = FaceMatcher(config)
        logger.info("✅ Face matcher initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize face matcher: {e}")
        return False

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting Face Verification Service...")
    
    if not initialize_face_matcher():
        logger.error("❌ Failed to initialize face matcher - service may not work properly")
    else:
        logger.info("✅ Face Verification Service started successfully")

# Helper Functions
def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to OpenCV image
    
    Args:
        base64_string: Base64 encoded image
        
    Returns:
        OpenCV image as numpy array
        
    Raises:
        HTTPException: If image cannot be decoded
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Check file size
        max_size = config.get('api', {}).get('max_image_size_mb', 10) * 1024 * 1024
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image too large. Maximum size: {max_size/1024/1024:.1f}MB"
            )
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Basic image validation
        if opencv_image.shape[0] < 50 or opencv_image.shape[1] < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image too small. Minimum size: 50x50 pixels"
            )
        
        return opencv_image
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image data: {str(e)}"
        )

# API Endpoints

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to docs."""
    return RedirectResponse(url="/docs")

@app.get("/welcome", tags=["general"])
async def welcome():
    """
    Welcome endpoint with API information
    """
    return {
        "service": "Face Verification API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "verify_face": "POST /verify-face",
            "health": "GET /health", 
            "version": "GET /version",
            "config": "GET /config"
        }
    }

@app.post("/verify-face", response_model=VerifyFaceResponse)
async def verify_face(request_data: VerifyFaceRequest):
    """
    Verify if two facial images belong to the same person
    
    This endpoint compares two images and returns a similarity score
    along with a boolean match decision based on configurable thresholds.
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Processing verification request for user: {request_data.user_id}")
        logger.info(f"📊 Image type: {request_data.image_type}")
        
        # Check if face matcher is initialized
        if face_matcher is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Face verification service not available"
            )
        
        # Decode images
        try:
            image1 = decode_base64_image(request_data.image_1_base64)
            image2 = decode_base64_image(request_data.image_2_base64)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process images: {str(e)}"
            )
        
        # Perform face verification
        result = face_matcher.verify_faces(image1, image2, request_data.image_type)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Prepare response
        response = VerifyFaceResponse(
            user_id=request_data.user_id,
            match_score=result['match_score'],
            match=result['match'],
            confidence_level=result.get('confidence_level', 'unknown'),
            image_type=request_data.image_type,
            threshold=result.get('threshold', 0.0),
            status=result['status'],
            error=result.get('error')
        )
        
        logger.info(f"✅ Verification completed for {request_data.user_id}: {result['status']} (score: {result['match_score']:.3f})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in face verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during face verification"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns the current status of the face verification service
    """
    service_status = "ok"
    
    # Check if face matcher is working
    if face_matcher is None:
        service_status = "degraded"
    
    return HealthResponse(
        status=service_status,
        service="face-verification",
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/version", response_model=VersionResponse)
async def get_version():
    """
    Get version information
    
    Returns version details about the API and underlying models
    """
    try:
        import face_recognition
        face_rec_version = face_recognition.__version__ if hasattr(face_recognition, '__version__') else "unknown"
    except:
        face_rec_version = "unknown"
    
    return VersionResponse(
        service_version="1.0.0",
        api_version="1.0.0",
        face_recognition_library=f"face_recognition-{face_rec_version}"
    )

@app.get("/config")
async def get_config():
    """
    Get current configuration (for debugging)
    
    Note: In production, you might want to restrict access to this endpoint
    """
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuration not loaded"
        )
    
    # Return config without sensitive information
    safe_config = {
        "thresholds": config.get("thresholds", {}),
        "face_detection": config.get("face_detection", {}),
        "api": {k: v for k, v in config.get("api", {}).items() if "key" not in k.lower()}
    }
    
    return safe_config

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": "Please check the API documentation at /docs"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "Please try again later"}
    )

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Load configuration for host and port
    config = load_config()
    api_config = config.get("api", {})
    
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    debug = api_config.get("debug", False)
    
    logger.info(f"🚀 Starting Face Verification Service on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )