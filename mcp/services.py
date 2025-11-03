from typing import Protocol, Dict, Any
import aiohttp
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
import deepai
from youtubesearchpython import VideosSearch

class CodeFetcher(Protocol):
    async def fetch(self, url: str) -> str:
        ...

class ContentAnalyzer(Protocol):
    async def analyze(self, content: str, prompt: str) -> str:
        ...

class YouTubeAPI(Protocol):
    async def search(self, query: str) -> list:
        ...

class ImageAPI(Protocol):
    async def generate(self, prompt: str) -> str:
        ...

class WebCodeFetcher:
    async def fetch(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                async with session.get(str(url), headers=headers) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    for tag in soup(['script', 'style']):
                        tag.decompose()
                    return soup.get_text(separator=' ', strip=True)
            except Exception as e:
                return f"Error: Could not fetch website. {e}"

class GeminiAnalyzer:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found in environment variables")
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    async def analyze(self, content: str, prompt: str) -> str:
        try:
            final_prompt = f"""
            Here is the text content from a website:
            ---
            {content[:30000]}
            ---
            
            Based on this content, please answer the following question:
            {prompt}
            """
            # Using sync invoke since LangChain doesn't support async yet
            print("Sending request to Gemini...")
            response = self.llm.invoke([HumanMessage(content=final_prompt)])
            print("Received response from Gemini")
            
            if not response or not response.content:
                raise ValueError("Empty response from Gemini")
            return response.content
        except Exception as e:
            print(f"Gemini analysis error: {str(e)}")
            return f"Error: Analysis failed - {str(e)}"

class YouTubeSearchAPI:
    async def search(self, query: str) -> list:
        try:
            api_key = os.getenv("YOUTUBE_API_KEY")
            if not api_key:
                raise ValueError("YouTube API key not found")

            async with aiohttp.ClientSession() as session:
                base_url = "https://www.googleapis.com/youtube/v3/search"
                params = {
                    'part': 'snippet',
                    'q': query,
                    'key': api_key,
                    'maxResults': 5,
                    'type': 'video',
                    'videoCategoryId': '10',
                    'videoEmbeddable': 'true'
                }

                async with session.get(base_url, params=params) as response:
                    data = await response.json()

                results = []
                for item in data.get('items', []):
                    video_id = item['id']['videoId']
                    title = item['snippet']['title']
                    thumbnail = item['snippet']['thumbnails']['high']['url']
                    channel = item['snippet']['channelTitle']
                    description = item['snippet']['description']
                    
                    results.append({
                        'title': title,
                        'video_url': f'https://www.youtube.com/watch?v={video_id}',
                        'embed_url': f'https://www.youtube.com/embed/{video_id}',
                        'thumbnail': thumbnail,
                        'channel': channel,
                        'description': description
                    })

                return results
        except Exception as e:
            return []

class DeepAIGenerator:
    async def generate(self, prompt: str) -> str:
        try:
            api_key = os.getenv("DEEPAI_API_KEY")
            if not api_key:
                raise ValueError("DeepAI API key not found")

            deepai.set_api_key(api_key)
            response = deepai.call_standard_api("text2img", text=prompt)
            return response['output_url']
        except Exception as e:
            return f"Error: Image generation failed. {e}"