"""
Research Paper Recommender API
FastAPI application that integrates with the ResearchRecommender
to provide paper search and recommendation capabilities.
"""
import os
import time
import traceback
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import uvicorn
import logging
import logging.handlers

# Configure logging with rotation to prevent large log files
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "app.log")

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set up rotating file handler
file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=10*1024*1024, backupCount=5
)
file_handler.setFormatter(formatter)

# Set up console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("research_api")

# Configurable settings
API_SETTINGS = {
    "default_max_results": 50,
    "default_recommendations": 5,
    "max_allowed_results": 200,
    "request_timeout": 60  # seconds
}

# Global component instances
recommender = None

# Initialize components on startup and shutdown on app termination
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing components...")
    try:
        from researchpaper_recommender import ResearchRecommender
        
        global recommender
        recommender = ResearchRecommender(
            use_faster_model=True,
            load_existing=True
        )
        
        logger.info("✅ Components initialized successfully")
        yield
    except ImportError as e:
        logger.error(f"❌ Failed to import required modules: {e}")
        yield
    except Exception as e:
        logger.error(f"❌ Error during initialization: {e}")
        logger.error(traceback.format_exc())
        yield
    
    # Shutdown
    logger.info("Shutting down components...")
    if recommender:
        recommender.save_state()
    logger.info("Components shut down")

# Initialize the API
app = FastAPI(
    title="Research Paper Recommender API",
    description="API for searching and recommending research papers from arXiv",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware for request timing and logging
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"Request to {request.url.path} processed in {process_time:.3f} seconds")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request to {request.url.path} failed after {process_time:.3f} seconds: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Internal server error: {str(e)}"}
        )

# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "https://research-recommender.example.com",
    # Add more allowed origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Serve static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dependency to verify components are loaded
def verify_components():
    if recommender is None:
        logger.error("Recommender not initialized")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable. Recommender not initialized.")
    return True

# Pydantic models
class DateRange(BaseModel):
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        if v is not None:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('Date must be in YYYY-MM-DD format')
        return v

class SearchRequest(BaseModel):
    query: str = Field(..., description="ArXiv search query")
    max_results: int = Field(API_SETTINGS["default_max_results"], 
                           description=f"Maximum number of results to return (default: {API_SETTINGS['default_max_results']})")
    date_range: Optional[DateRange] = Field(None, description="Optional date range for filtering papers")
    categories: Optional[List[str]] = Field(None, description="Optional list of arXiv categories to filter by")
    
    @validator('max_results')
    def validate_max_results(cls, v):
        if v < 1:
            return API_SETTINGS["default_max_results"]
        if v > API_SETTINGS["max_allowed_results"]:
            return API_SETTINGS["max_allowed_results"]
        return v

class RecommendRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text to find similar papers to")
    paper_id: Optional[str] = Field(None, description="Paper ID to find similar papers to")
    user_id: Optional[str] = Field(None, description="User ID for personalized recommendations")
    k: int = Field(API_SETTINGS["default_recommendations"], 
                 description=f"Number of recommendations to return")
    date_range: Optional[DateRange] = Field(None, description="Optional date range for filtering")
    include_quality: bool = Field(True, description="Include quality assessment in results")
    
    @validator('k')
    def validate_k(cls, v):
        if v < 1:
            return API_SETTINGS["default_recommendations"]
        if v > API_SETTINGS["max_allowed_results"]:
            return API_SETTINGS["max_allowed_results"]
        return v

class UserInteractionRequest(BaseModel):
    user_id: str
    paper_id: str
    interaction_type: str = Field(..., description="Type of interaction: 'view', 'like', 'dislike', 'save', 'cite'")

class Paper(BaseModel):
    id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    pdf_url: Optional[str] = None
    categories: Optional[List[str]] = None
    similarity: Optional[float] = None
    quality_score: Optional[float] = None
    combined_score: Optional[float] = None
    citation_count: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "2301.12345",
                "title": "Recent Advances in Transformer Models",
                "abstract": "This paper explores the recent advances in transformer architecture...",
                "authors": ["J. Smith", "A. Lee"],
                "published": "2023-01-15",
                "pdf_url": "https://arxiv.org/pdf/2301.12345.pdf",
                "categories": ["cs.LG", "cs.CL"],
                "similarity": 0.92,
                "quality_score": 0.85,
                "combined_score": 0.89
            }
        }

class ApiStatus(BaseModel):
    status: str
    components_initialized: bool
    version: str

# UI routes
@app.get("/", response_class=HTMLResponse)
async def get_html():
    """Serve the main UI page"""
    return FileResponse("static/index.html")

@app.get("/ui")
async def serve_ui():
    """Alias for the main UI page"""
    return FileResponse("static/index.html")

# API routes
@app.get("/api-info", response_model=ApiStatus)
async def get_api_info():
    """Get information about the API status"""
    return {
        "status": "operational" if recommender else "degraded",
        "components_initialized": recommender is not None,
        "version": app.version
    }

@app.post("/search", response_model=List[Paper])
async def search_papers(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_components)
):
    """Search for papers based on a query and optional filters"""
    logger.info(f"Received search request: {request.dict()}")
    try:
        # Prepare date parameters if provided
        date_start = None
        date_end = None
        if request.date_range:
            date_start = request.date_range.start_date
            date_end = request.date_range.end_date
        
        # Fetch papers using the ResearchRecommender
        papers_df = recommender.fetch_papers(
            query=request.query,
            categories=request.categories,
            max_results=request.max_results,
            date_start=date_start,
            date_end=date_end
        )
        
        if papers_df.empty:
            logger.warning(f"No papers found for query: {request.query}")
            return []
        
        # Convert DataFrame to list of Paper models
        papers = []
        for _, row in papers_df.iterrows():
            paper = Paper(
                id=row['id'],
                title=row['title'],
                abstract=row['abstract'],
                authors=row['authors'],
                published=str(row['published']),
                pdf_url=row.get('pdf_url'),
                categories=row.get('categories', [])
            )
            papers.append(paper)
            
        return papers
        
    except Exception as e:
        logger.error(f"Error in search_papers: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing search request: {str(e)}"
        )

@app.post("/recommend", response_model=List[Paper])
async def get_recommendations(
    request: RecommendRequest,
    _: bool = Depends(verify_components)
):
    """Get paper recommendations with quality assessment"""
    logger.info(f"Received recommendation request")
    try:
        # Prepare filter criteria if date range is provided
        filter_criteria = None
        if request.date_range:
            filter_criteria = {
                'published': {
                    'min': request.date_range.start_date,
                    'max': request.date_range.end_date
                }
            }
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            text=request.text,
            paper_id=request.paper_id,
            user_id=request.user_id,
            k=request.k,
            filter_criteria=filter_criteria,
            include_quality=request.include_quality
        )
        
        if recommendations.empty:
            logger.warning("No recommendations found")
            return []
        
        # Convert to Paper models
        papers = []
        for _, row in recommendations.iterrows():
            paper = Paper(
                id=row['id'],
                title=row['title'],
                abstract=row['abstract'],
                authors=row['authors'],
                published=str(row['published']),
                pdf_url=row.get('pdf_url'),
                categories=row.get('categories', []),
                similarity=float(row['similarity_score']) if 'similarity_score' in row else None,
                quality_score=float(row['quality_score']) if 'quality_score' in row else None,
                combined_score=float(row['combined_score']) if 'combined_score' in row else None
            )
            papers.append(paper)
            
        return papers
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing recommendation request: {str(e)}"
        )

@app.post("/user/interaction")
async def record_interaction(
    request: UserInteractionRequest,
    _: bool = Depends(verify_components)
):
    """Record user interaction with a paper"""
    try:
        recommender.record_user_interaction(
            user_id=request.user_id,
            paper_id=request.paper_id,
            interaction_type=request.interaction_type
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error recording interaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/insights/{user_id}")
async def get_user_insights(
    user_id: str,
    _: bool = Depends(verify_components)
):
    """Get insights about user preferences"""
    try:
        insights = recommender.get_user_insights(user_id)
        return insights
    except Exception as e:
        logger.error(f"Error getting user insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top-cited")
async def get_top_cited(
    query: str,
    k: int = 5,
    _: bool = Depends(verify_components)
):
    """Get top cited papers for a query"""
    try:
        papers_df = recommender.get_top_cited_papers(query, k)
        
        if papers_df.empty:
            return []
            
        # Convert to Paper models
        papers = []
        for _, row in papers_df.iterrows():
            paper = Paper(
                id=row['id'],
                title=row['title'],
                abstract=row['abstract'],
                authors=row['authors'],
                published=str(row['published']),
                pdf_url=row.get('pdf_url'),
                categories=row.get('categories', []),
                citation_count=int(row['citation_count'])
            )
            papers.append(paper)
            
        return papers
    except Exception as e:
        logger.error(f"Error getting top cited papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/paper/quality/{paper_id}")
async def assess_paper_quality(
    paper_id: str,
    _: bool = Depends(verify_components)
):
    """Assess the quality of a paper"""
    try:
        quality_score = recommender.assess_paper_quality(paper_id)
        return {"paper_id": paper_id, "quality_score": quality_score}
    except Exception as e:
        logger.error(f"Error assessing paper quality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/save-state", status_code=200)
async def trigger_save_state(
    _: bool = Depends(verify_components)
):
    """Manually trigger state saving"""
    try:
        recommender.save_state()
        return {"status": "success", "message": "State saved successfully"}
    except Exception as e:
        logger.error(f"Error saving state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application when executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
