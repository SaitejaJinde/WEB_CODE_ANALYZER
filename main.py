import os
import logging
from typing import Union, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import MCP components
from mcp.models import (
    WebsiteAnalysisRequest,
    WebsiteAnalysisResponse,
    MusicSearchRequest,
    MusicSearchResponse,
    ImageGenerationRequest,
    ImageGenerationResponse
)
from mcp.analyzers import WebsiteAnalyzer, MusicSearcher, ImageGenerator
from mcp.services import (
    WebCodeFetcher,
    GeminiAnalyzer,
    YouTubeSearchAPI,
    DeepAIGenerator
)

# Load environment variables
load_dotenv()

# Initialize services
code_fetcher = WebCodeFetcher()
content_analyzer = GeminiAnalyzer()
youtube_api = YouTubeSearchAPI()
image_api = DeepAIGenerator()

# Initialize analyzers
website_analyzer = WebsiteAnalyzer(code_fetcher, content_analyzer)
music_searcher = MusicSearcher(youtube_api)
image_generator = ImageGenerator(image_api)

# Validate environment variables
required_env_vars = [
    "GOOGLE_API_KEY",
    "YOUTUBE_API_KEY",
    "DEEPAI_API_KEY"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Web Analyzer",
    description="Analyze websites, search music, and generate images using AI.",
    version="1.0.0"
)

# Enable CORS with more secure settings
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount the static files directory
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Successfully mounted static files directory")
except Exception as e:
    logger.error(f"Failed to mount static files directory: {e}")
    raise

# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request format", "errors": exc.errors()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status
    """
    return {"status": "healthy", "version": "1.0.0"}

# API Endpoints
@app.post("/api/analyze")
async def analyze_website(request: WebsiteAnalysisRequest):
    """
    Analyze a website with a given prompt
    """
    logger.info(f"Received website analysis request for URL: {request.url}")
    try:
        result = await website_analyzer.analyze(request)
        logger.info(f"Successfully analyzed website: {request.url}")
        return result
    except Exception as e:
        logger.error(f"Website analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/music/search")
async def search_music(request: MusicSearchRequest):
    """
    Search for music based on a query
    """
    logger.info(f"Received music search request: {request.query}")
    try:
        result = await music_searcher.analyze(request)
        logger.info(f"Successfully searched for music: {len(result.results)} results")
        return result
    except Exception as e:
        logger.error(f"Music search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/image/generate")
async def generate_image(request: ImageGenerationRequest):
    """
    Generate an image based on a prompt
    """
    logger.info(f"Received image generation request: {request.prompt}")
    try:
        result = await image_generator.analyze(request)
        logger.info("Successfully generated image")
        return result
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoint for backward compatibility
@app.post("/agent-task")
async def handle_agent_task(request: Dict[str, Any]):
    """
    Legacy endpoint that routes requests to the appropriate MCP endpoint.
    """
    logger.info(f"Received legacy agent task request: {request}")
    try:
        if "url" in request and request["url"]:
            return await analyze_website(WebsiteAnalysisRequest(**request))
        elif "prompt" in request and any(keyword in request["prompt"].lower() 
            for keyword in ["find music", "play music", "search song"]):
            return await search_music(MusicSearchRequest(query=request["prompt"]))
        elif "prompt" in request and any(keyword in request["prompt"].lower() 
            for keyword in ["generate image", "create image"]):
            return await generate_image(ImageGenerationRequest(prompt=request["prompt"]))
        else:
            msg = "Please specify a valid request type (website analysis, music search, or image generation)"
            logger.warning(f"Invalid request type: {request}")
            raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        logger.error(f"Agent task failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint to serve web interface
@app.get("/")
async def read_root():
    """
    Serve the main web interface
    """
    try:
        return FileResponse('static/index.html')
    except Exception as e:
        logger.error(f"Failed to serve index.html: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load web interface")