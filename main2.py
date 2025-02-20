# main2.py
print("1️⃣ Importing libraries...")  # Debug step 1
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse  # Fixed import here
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

print("2️⃣ Setting up logging...")  # Debug step 2
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("3️⃣ Importing research_recommender2...")  # Debug step 3
try:
    from research_reccomender2 import ArxivFetcher, EmbeddingSystem
    print("✅ Successfully imported research_recommender2")
except ImportError as e:
    print(f"❌ ImportError: {e}")
    # Fix typographical error in the import name if needed
    try:
        from research_reccomender2 import ArxivFetcher, EmbeddingSystem
        print("✅ Successfully imported with alternate spelling: research_reccomender2")
    except ImportError:
        raise

print("4️⃣ Initializing FastAPI app...")  # Debug step 4
app = FastAPI(title="Research Paper Recommender API")

# Serve static files (for the frontend UI)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add UI route to serve the frontend
@app.get("/ui")
async def serve_ui():
    return FileResponse("static/index.html")

# Add CORS middleware
print("5️⃣ Adding CORS middleware...")  # Debug step 5
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global instances
print("6️⃣ Initializing ArxivFetcher and EmbeddingSystem...")  # Debug step 6
try:
    fetcher = ArxivFetcher()
    embedder = EmbeddingSystem()
except Exception as e:
    print(f"❌ Error initializing components: {e}")
    raise

class SearchRequest(BaseModel):
    query: str
    max_results: int = 50

class RecommendRequest(BaseModel):
    text: Optional[str] = None
    paper_id: Optional[str] = None
    k: int = 5

class Paper(BaseModel):
    id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    similarity: Optional[float] = None

@app.post("/search", response_model=List[Paper])
async def search_papers(request: SearchRequest):
    print("7️⃣ Received search request")
    logger.info(f"Received search request with query: {request.query}")
    try:
        papers_df = fetcher.fetch(query=request.query, max_results=request.max_results)
        embedder.process_papers(papers_df)

        # Convert the DataFrame to records, but first convert date to string
        papers_df['published'] = papers_df['published'].astype(str)
        return papers_df.to_dict('records')
    except Exception as e:
        logger.error(f"Error in search_papers: {str(e)}")
        print(f"❌ Error in search_papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend", response_model=List[Paper])
async def get_recommendations(request: RecommendRequest):
    print("8️⃣ Received recommend request")  # Debug step 8
    logger.info(f"Received recommend request")
    try:
        recommendations = embedder.recommend(
            text=request.text,
            paper_id=request.paper_id,
            k=request.k
        )
        # Convert published date to string if it's not already
        if 'published' in recommendations.columns:
            recommendations['published'] = recommendations['published'].astype(str)
        return recommendations.to_dict('records')
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        print(f"❌ Error in get_recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api-info")
async def root():
    print("9️⃣ Handling root request")  # Debug step 9
    return {"message": "Research Paper Recommender API is running"}

# Handle direct HTML request for the root path
@app.get("/", response_class=HTMLResponse)
async def get_html():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    print("🔟 Starting Uvicorn server...")  # Debug step 10
    logger.info("Starting server...")
    # Use the string reference format for proper reload support
    uvicorn.run("main2:app", host="0.0.0.0", port=8000, reload=True)