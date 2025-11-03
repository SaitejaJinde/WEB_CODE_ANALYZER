from typing import List, Optional
from pydantic import BaseModel, HttpUrl

class WebsiteAnalysisRequest(BaseModel):
    url: HttpUrl
    prompt: str

class WebsiteAnalysisResponse(BaseModel):
    task: str = "website_analysis"
    source_url: str
    user_prompt: str
    analysis_result: str

class MusicSearchRequest(BaseModel):
    query: str

class MusicVideo(BaseModel):
    title: str
    video_url: str
    embed_url: str
    thumbnail: str
    channel: str
    description: str

class MusicSearchResponse(BaseModel):
    task: str = "music_search"
    query: str
    results: List[MusicVideo]
    error: Optional[str] = None

class ImageGenerationRequest(BaseModel):
    prompt: str

class ImageGenerationResponse(BaseModel):
    task: str = "image_generation"
    prompt: str
    image_url: str
    error: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str