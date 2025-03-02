

"""
Research Paper Recommender API
FastAPI application that integrates with the ResearchRecommender system
to provide paper search and recommendation capabilities.
"""
import os
import time
import json
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
import traceback
import logging
import sys

# Ensure UTF-8 encoding for logs
sys.stdout.reconfigure(encoding='utf-8')

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',)

# Set up console handler with UTF-8 encoding
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler],  # Add file_handler if needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure logging with rotation to prevent large log files
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "app.log")


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
        # Import the ResearchRecommender class directly from the module
        from research_recommender import ResearchRecommender
        
        global recommender
        recommender = ResearchRecommender()
        
        # Load existing index if available
        try:
            recommender.load_index("research_index")
            logger.info("Loaded existing research index")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
        
        logger.info("  Components initialized successfully")
        yield
    except ImportError as e:
        logger.error(f"   Failed to import required modules: {e}")
        # Still start the app, but with limited functionality
        yield
    except Exception as e:
        logger.error(f"   Error during initialization: {e}")
        logger.error(traceback.format_exc())
        yield
    
    # Shutdown
    logger.info("Shutting down components...")
    # Save the index on shutdown
    if recommender and recommender.embedding_system.index.ntotal > 0:
        try:
            recommender.save_index("research_index")
            logger.info("Saved research index")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
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
    "http://localhost:3000",  # Add your React app's default URL
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve static files for the frontend app.mount("/static", StaticFiles(directory="static"), name="static")

# Dependency to verify components are loaded
def verify_components():
    if recommender is None:
        logger.error("Components not initialized")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable. Components not initialized.")
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
    k: int = Field(API_SETTINGS["default_recommendations"], 
                 description=f"Number of recommendations to return (default: {API_SETTINGS['default_recommendations']})")
    date_range: Optional[DateRange] = Field(None, description="Optional date range for filtering recommendations")
    quality_aware: bool = Field(True, description="Whether to use quality assessment in ranking")
    
    @validator('k')
    def validate_k(cls, v):
        if v < 1:
            return API_SETTINGS["default_recommendations"]
        if v > API_SETTINGS["max_allowed_results"]:
            return API_SETTINGS["max_allowed_results"]
        return v
    
    @validator('text', 'paper_id')
    def validate_input(cls, v, values):
        # Ensure either text or paper_id is provided
        if 'text' not in values and 'paper_id' not in values:
            raise ValueError('Either text or paper_id must be provided')
        return v

class SeminalPapersRequest(BaseModel):
    topic: str = Field(..., description="Research topic to find seminal papers for")
    max_results: int = Field(10, description="Maximum number of papers to return")
    
    @validator('max_results')
    def validate_max_results(cls, v):
        if v < 1:
            return 10
        if v > API_SETTINGS["max_allowed_results"]:
            return API_SETTINGS["max_allowed_results"]
        return v

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
                "quality_score": 0.85
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
    """
    Search for papers based on a query and optional filters
    """
    logger.info(f"Received search request: {request.dict()}")
    try:
        # Prepare date parameters if provided
        date_start = None
        date_end = None
        if request.date_range:
            date_start = request.date_range.start_date
            date_end = request.date_range.end_date
        
        # Build the query with categories if provided
        final_query = request.query
        if request.categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in request.categories])
            final_query = f"({request.query}) AND ({cat_query})"
        
        # Fetch papers using the recommender
        start_time = time.time()
        papers_df = recommender.fetch_and_index(
            query=final_query, 
            max_results=request.max_results,
            date_start=date_start,
            date_end=date_end
        )
        fetch_time = time.time() - start_time
        logger.info(f"Fetched {len(papers_df)} papers in {fetch_time:.2f} seconds")
        
        if papers_df.empty:
            logger.warning(f"No papers found for query: {final_query}")
            return []
        
        # Ensure date is properly formatted as string
        papers_df['published'] = papers_df['published'].astype(str)
        
        # Convert DataFrame to list of Paper models
        papers = []
        for _, row in papers_df.iterrows():
            paper = Paper(
                id=row['id'],
                title=row['title'],
                abstract=row['abstract'],
                authors=row['authors'],
                published=row['published'],
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
    """
    Get paper recommendations based on text or a paper ID
    """
    logger.info(f"Received recommendation request")
    try:
        # Get recommendations using the recommender
        recommendations = recommender.recommend(
            text=request.text,
            paper_id=request.paper_id,
            k=request.k,
            min_date=request.date_range.start_date if request.date_range else None,
            max_date=request.date_range.end_date if request.date_range else None,
            quality_aware=request.quality_aware
        )
        
        if recommendations.empty:
            logger.warning("No recommendations found")
            return []
        
        # Ensure date is properly formatted as string
        if 'published' in recommendations.columns:
            recommendations['published'] = recommendations['published'].astype(str)
        
        # Convert to Paper models
        papers = []
        for _, row in recommendations.iterrows():
            # Determine which similarity field to use
            similarity = None
            if 'combined_score' in row:
                similarity = float(row['combined_score'])
            elif 'similarity_score' in row:
                similarity = float(row['similarity_score'])
                
            quality_score = float(row['quality_score']) if 'quality_score' in row else None
                
            paper = Paper(
                id=row['id'],
                title=row['title'],
                abstract=row['abstract'],
                authors=row['authors'],
                published=row['published'],
                pdf_url=row.get('pdf_url'),
                categories=row.get('categories', []),
                similarity=similarity,
                quality_score=quality_score
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

@app.post("/seminal-papers", response_model=List[Paper])
async def find_seminal_papers(
    request: SeminalPapersRequest,
    _: bool = Depends(verify_components)
):
    """
    Find seminal papers on a topic
    """
    logger.info(f"Received seminal papers request for topic: {request.topic}")
    try:
        # Get seminal papers using the recommender
        papers_df = recommender.find_seminal_papers(
            topic=request.topic,
            max_results=request.max_results
        )
        
        if papers_df.empty:
            logger.warning(f"No seminal papers found for topic: {request.topic}")
            return []
        
        # Ensure date is properly formatted as string
        papers_df['published'] = papers_df['published'].astype(str)
        
        # Convert DataFrame to list of Paper models
        papers = []
        for _, row in papers_df.iterrows():
            # Get quality score for each paper
            quality_score = recommender.assess_paper_quality(row['id'])
            
            paper = Paper(
                id=row['id'],
                title=row['title'],
                abstract=row['abstract'],
                authors=row['authors'],
                published=row['published'],
                pdf_url=row.get('pdf_url'),
                categories=row.get('categories', []),
                quality_score=quality_score
            )
            papers.append(paper)
            
        return papers
        
    except Exception as e:
        logger.error(f"Error in find_seminal_papers: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing seminal papers request: {str(e)}"
        )

@app.get("/paper-quality/{paper_id}", response_model=Dict[str, float])
async def get_paper_quality(
    paper_id: str,
    _: bool = Depends(verify_components)
):
    """
    Get quality assessment for a specific paper
    """
    logger.info(f"Received quality assessment request for paper: {paper_id}")
    try:
        quality_score = recommender.assess_paper_quality(paper_id)
        return {"quality_score": quality_score}
    except Exception as e:
        logger.error(f"Error in get_paper_quality: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing quality assessment request: {str(e)}"
        )

@app.get("/citation-info/{paper_id}", response_model=Dict[str, Any])
async def get_citation_info(
    paper_id: str,
    _: bool = Depends(verify_components)
):
    """
    Get citation information for a specific paper
    """
    logger.info(f"Received citation info request for paper: {paper_id}")
    try:
        citation_info = recommender.get_citation_info(paper_id)
        return citation_info
    except Exception as e:
        logger.error(f"Error in get_citation_info: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing citation info request: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring services
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "recommender": recommender is not None,
            "index_size": recommender.embedding_system.index.ntotal if recommender else 0
        }
    }

# Custom exception handler for better error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. See logs for details."}
    )
if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    # Log to stdout as well when running directly
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info",
        workers=4,  # Adjust based on available CPU cores
        timeout_keep_alive=API_SETTINGS["request_timeout"]
    )
