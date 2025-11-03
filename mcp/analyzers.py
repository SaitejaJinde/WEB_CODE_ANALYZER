from typing import List, Dict, Any
from abc import ABC, abstractmethod
from .models import (
    WebsiteAnalysisRequest,
    WebsiteAnalysisResponse,
    MusicSearchRequest,
    MusicSearchResponse,
    MusicVideo,
    ImageGenerationRequest,
    ImageGenerationResponse
)

class BaseAnalyzer(ABC):
    @abstractmethod
    async def analyze(self, request: Any) -> Any:
        pass

class WebsiteAnalyzer(BaseAnalyzer):
    def __init__(self, code_fetcher, content_analyzer):
        self.code_fetcher = code_fetcher
        self.content_analyzer = content_analyzer

    async def analyze(self, request: WebsiteAnalysisRequest) -> WebsiteAnalysisResponse:
        try:
            print(f"Starting analysis of URL: {request.url}")
            
            # Validate URL
            if not request.url or not str(request.url).startswith(('http://', 'https://')):
                raise ValueError("Invalid URL. Must start with http:// or https://")
            
            # Fetch website content
            print("Fetching website content...")
            code = await self.code_fetcher.fetch(str(request.url))
            if code.startswith("Error:"):
                raise ValueError(code)
            print(f"Successfully fetched content: {len(code)} characters")
            
            # Validate prompt
            if not request.prompt or len(request.prompt.strip()) == 0:
                raise ValueError("Empty prompt provided")
            
            # Analyze the content
            print(f"Analyzing content with prompt: {request.prompt[:100]}...")
            analysis = await self.content_analyzer.analyze(code, request.prompt)
            if analysis.startswith("Error:"):
                raise ValueError(analysis)
            print("Analysis completed successfully")
            
            # Return the response
            return WebsiteAnalysisResponse(
                source_url=str(request.url),
                user_prompt=request.prompt,
                analysis_result=analysis
            )
        except ValueError as ve:
            print(f"Validation error: {str(ve)}")
            raise ValueError(f"Analysis failed: {str(ve)}")
        except Exception as e:
            print(f"Unexpected error during analysis: {str(e)}")
            raise ValueError(f"Analysis failed due to unexpected error: {str(e)}")

class MusicSearcher(BaseAnalyzer):
    def __init__(self, youtube_api):
        self.youtube_api = youtube_api

    async def analyze(self, request: MusicSearchRequest) -> MusicSearchResponse:
        try:
            print(f"Starting music search for query: {request.query}")
            
            # Validate query
            if not request.query or len(request.query.strip()) == 0:
                raise ValueError("Empty search query provided")
            
            # Search for videos
            print("Searching for music videos...")
            results = await self.youtube_api.search(request.query)
            
            if not results:
                print("No results found")
                return MusicSearchResponse(
                    query=request.query,
                    results=[]
                )
            
            print(f"Found {len(results)} videos")
            
            # Convert results to MusicVideo objects
            music_videos = []
            for result in results:
                try:
                    video = MusicVideo(
                        title=result['title'],
                        video_url=result['video_url'],
                        embed_url=result['embed_url'],
                        thumbnail=result['thumbnail'],
                        channel=result['channel'],
                        description=result['description']
                    )
                    music_videos.append(video)
                except KeyError as ke:
                    print(f"Skipping malformed result: {ke}")
                    continue
            
            print(f"Successfully processed {len(music_videos)} videos")
            
            # Return the response
            return MusicSearchResponse(
                query=request.query,
                results=music_videos
            )
        except ValueError as ve:
            print(f"Validation error: {str(ve)}")
            raise ValueError(f"Music search failed: {str(ve)}")
        except Exception as e:
            print(f"Unexpected error during music search: {str(e)}")
            raise ValueError(f"Music search failed due to unexpected error: {str(e)}")

class ImageGenerator(BaseAnalyzer):
    def __init__(self, image_api):
        self.image_api = image_api

    async def analyze(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        try:
            print(f"Starting image generation for prompt: {request.prompt}")
            
            # Validate prompt
            if not request.prompt or len(request.prompt.strip()) == 0:
                raise ValueError("Empty prompt provided")
                
            if len(request.prompt) > 1000:
                raise ValueError("Prompt too long. Maximum length is 1000 characters")
            
            # Generate image
            print("Generating image...")
            image_url = await self.image_api.generate(request.prompt)
            
            if not image_url or image_url.startswith("Error:"):
                raise ValueError(image_url if image_url else "No image URL returned")
                
            if not image_url.startswith(('http://', 'https://')):
                raise ValueError("Invalid image URL returned")
            
            print("Image generated successfully")
            
            # Return the response
            return ImageGenerationResponse(
                prompt=request.prompt,
                image_url=image_url
            )
        except ValueError as ve:
            print(f"Validation error: {str(ve)}")
            raise ValueError(f"Image generation failed: {str(ve)}")
        except Exception as e:
            print(f"Unexpected error during image generation: {str(e)}")
            raise ValueError(f"Image generation failed due to unexpected error: {str(e)}")