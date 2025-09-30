"""
FastAPI REST API for Phishing Detection
Provides endpoints for URL and email phishing detection
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import List, Optional, Dict, Any
import sys
import os
from pathlib import Path
import uvicorn
from datetime import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from phishing_detector import PhishingDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Phishing Detection API",
    description="AI-powered phishing detection for URLs and emails",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = PhishingDetector()

# Request/Response Models


class URLRequest(BaseModel):
    """Request model for URL analysis"""

    url: str = Field(..., description="URL to analyze for phishing")

    @validator("url")
    def validate_url(cls, v):
        if not v.startswith(("http://", "https://")):
            v = "http://" + v
        return v

    class Config:
        schema_extra = {"example": {"url": "https://example.com/login"}}


class BatchURLRequest(BaseModel):
    """Request model for batch URL analysis"""

    urls: List[str] = Field(..., description="List of URLs to analyze")

    @validator("urls")
    def validate_urls(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 URLs per batch")
        return v

    class Config:
        schema_extra = {
            "example": {
                "urls": [
                    "https://example.com",
                    "http://suspicious-site.tk",
                    "https://paypal-verify.com",
                ]
            }
        }


class URLResponse(BaseModel):
    """Response model for URL analysis"""

    url: str
    is_phishing: bool
    phishing_probability: float
    risk_score: float
    classification: str
    warnings: List[str]
    suspicious_features: List[Dict[str, Any]]
    timestamp: str


class BatchURLResponse(BaseModel):
    """Response model for batch URL analysis"""

    results: List[URLResponse]
    total_analyzed: int
    phishing_count: int
    safe_count: int
    timestamp: str


class EmailAnalysisResponse(BaseModel):
    """Response model for email analysis"""

    is_phishing: bool
    risk_score: float
    classification: str
    warnings: List[str]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: str
    model_loaded: bool
    version: str


class StatsResponse(BaseModel):
    """Statistics response"""

    total_requests: int
    urls_analyzed: int
    emails_analyzed: int
    phishing_detected: int
    uptime_seconds: float


# Global statistics
stats = {
    "total_requests": 0,
    "urls_analyzed": 0,
    "emails_analyzed": 0,
    "phishing_detected": 0,
    "start_time": datetime.now(),
}


# API Endpoints


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Phishing Detection API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "analyze_url": "/api/v1/analyze/url",
            "analyze_batch": "/api/v1/analyze/batch",
            "analyze_email": "/api/v1/analyze/email",
            "comprehensive": "/api/v1/analyze/comprehensive",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": detector.model is not None,
        "version": "1.0.0",
    }


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["General"])
async def get_stats():
    """Get API usage statistics"""
    uptime = (datetime.now() - stats["start_time"]).total_seconds()

    return {
        "total_requests": stats["total_requests"],
        "urls_analyzed": stats["urls_analyzed"],
        "emails_analyzed": stats["emails_analyzed"],
        "phishing_detected": stats["phishing_detected"],
        "uptime_seconds": uptime,
    }


@app.post("/api/v1/analyze/url", response_model=URLResponse, tags=["Analysis"])
async def analyze_url(request: URLRequest):
    """
    Analyze a single URL for phishing indicators

    Returns detailed analysis including risk score, classification,
    warnings, and suspicious features.
    """
    try:
        stats["total_requests"] += 1
        stats["urls_analyzed"] += 1

        logger.info(f"Analyzing URL: {request.url}")

        # Analyze URL
        result = detector.predict_url(request.url)

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])

        # Update stats
        if result.get("is_phishing"):
            stats["phishing_detected"] += 1

        # Prepare response
        response = {
            "url": result["url"],
            "is_phishing": result["is_phishing"],
            "phishing_probability": result["phishing_probability"],
            "risk_score": result["risk_score"],
            "classification": result["classification"],
            "warnings": result.get("warnings", []),
            "suspicious_features": result.get("suspicious_features", []),
            "timestamp": datetime.now().isoformat(),
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing URL: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/analyze/batch", response_model=BatchURLResponse, tags=["Analysis"])
async def analyze_batch_urls(request: BatchURLRequest):
    """
    Analyze multiple URLs in batch

    Maximum 100 URLs per request.
    Returns individual results for each URL plus summary statistics.
    """
    try:
        stats["total_requests"] += 1
        stats["urls_analyzed"] += len(request.urls)

        logger.info(f"Analyzing batch of {len(request.urls)} URLs")

        # Analyze all URLs
        results = detector.predict_batch(request.urls)

        # Count phishing
        phishing_count = sum(1 for r in results if r.get("is_phishing"))
        safe_count = len(results) - phishing_count

        # Update stats
        stats["phishing_detected"] += phishing_count

        # Format results
        formatted_results = []
        for result in results:
            if not result.get("error"):
                formatted_results.append(
                    {
                        "url": result["url"],
                        "is_phishing": result["is_phishing"],
                        "phishing_probability": result["phishing_probability"],
                        "risk_score": result["risk_score"],
                        "classification": result["classification"],
                        "warnings": result.get("warnings", []),
                        "suspicious_features": result.get("suspicious_features", []),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        response = {
            "results": formatted_results,
            "total_analyzed": len(request.urls),
            "phishing_count": phishing_count,
            "safe_count": safe_count,
            "timestamp": datetime.now().isoformat(),
        }

        return response

    except Exception as e:
        logger.error(f"Error analyzing batch: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@app.post(
    "/api/v1/analyze/email", response_model=EmailAnalysisResponse, tags=["Analysis"]
)
async def analyze_email(file: UploadFile = File(...)):
    """
    Analyze an email file for phishing indicators

    Upload an email file (.eml, .msg, or raw email content).
    Returns risk score, classification, and warnings.
    """
    try:
        stats["total_requests"] += 1
        stats["emails_analyzed"] += 1

        logger.info(f"Analyzing email file: {file.filename}")

        # Read email content
        email_content = await file.read()

        # Analyze email
        result = detector.predict_email(email_content)

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])

        # Update stats
        if result.get("is_phishing"):
            stats["phishing_detected"] += 1

        response = {
            "is_phishing": result["is_phishing"],
            "risk_score": result["risk_score"],
            "classification": result["classification"],
            "warnings": result.get("warnings", []),
            "timestamp": datetime.now().isoformat(),
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing email: {e}")
        raise HTTPException(status_code=500, detail=f"Email analysis failed: {str(e)}")


@app.post("/api/v1/analyze/comprehensive", tags=["Analysis"])
async def analyze_comprehensive(request: URLRequest):
    """
    Comprehensive URL analysis with detailed report

    Returns in-depth analysis including URL components,
    risk analysis, warnings, suspicious features, and recommendations.
    """
    try:
        stats["total_requests"] += 1
        stats["urls_analyzed"] += 1

        logger.info(f"Comprehensive analysis for URL: {request.url}")

        # Perform comprehensive analysis
        report = detector.analyze_url_comprehensive(request.url)

        if report.get("prediction", {}).get("error"):
            raise HTTPException(status_code=400, detail=report["prediction"]["error"])

        # Update stats
        if report.get("prediction", {}).get("is_phishing"):
            stats["phishing_detected"] += 1

        report["timestamp"] = datetime.now().isoformat()

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(
            status_code=500, detail=f"Comprehensive analysis failed: {str(e)}"
        )


@app.get("/api/v1/model/info", tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model

    Returns model type, features, and configuration details.
    """
    try:
        info = detector.get_model_info()
        info["timestamp"] = datetime.now().isoformat()
        return info

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/api/v1/model/load", tags=["Model"])
async def load_model(
    model_path: str = Body(...),
    scaler_path: Optional[str] = Body(None),
    features_path: Optional[str] = Body(None),
):
    """
    Load a trained model from disk

    Provide paths to model, scaler, and feature names files.
    """
    try:
        success = detector.load_model(model_path, scaler_path, features_path)

        if success:
            return {
                "status": "success",
                "message": "Model loaded successfully",
                "timestamp": datetime.now().isoformat(),
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to load model")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


@app.post("/api/v1/report", tags=["Reporting"])
async def report_phishing(
    url: str = Body(...),
    reason: str = Body(...),
    reporter_email: Optional[str] = Body(None),
):
    """
    Report a phishing URL

    Allow users to report suspected phishing URLs for review.
    """
    try:
        # In production, this would save to database
        report_data = {
            "url": url,
            "reason": reason,
            "reporter_email": reporter_email,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Phishing report received: {url}")

        # TODO: Save to database or notification system

        return {
            "status": "success",
            "message": "Report submitted successfully",
            "report_id": f"REP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error submitting report: {e}")
        raise HTTPException(
            status_code=500, detail=f"Report submission failed: {str(e)}"
        )


# Error handlers


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
        },
    )


# Startup event


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Phishing Detection API...")
    logger.info("API is ready to accept requests")

    # Try to load default model if it exists
    models_dir = Path(__file__).parent.parent / "models"
    default_model = models_dir / "best_model.pkl"

    if default_model.exists():
        logger.info(f"Loading default model from {default_model}")
        try:
            detector.load_model(
                str(default_model),
                str(models_dir / "best_model_scaler.pkl"),
                str(models_dir / "best_model_features.json"),
            )
            logger.info("Default model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load default model: {e}")
            logger.info("API will use rule-based detection")
    else:
        logger.info("No default model found, using rule-based detection")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Phishing Detection API...")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
