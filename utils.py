import requests
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from bs4 import BeautifulSoup
from youtubesearchpython import VideosSearch
import deepai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Utility 1: Website Scraper ---
def get_website_code(url: str) -> str:
    """
    Fetches the raw HTML code from a website.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
            
        return soup.get_text(separator=' ', strip=True)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching website: {e}")
        return f"Error: Could not fetch website. {e}"

# --- Utility 2: LLM Code Analyzer (Gemini) ---
def analyze_code_with_llm(code: str, prompt: str) -> str:
    """
    Sends the fetched code to the free Gemini API to be analyzed.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
        
        final_prompt = f"""
        Here is the text content from a website:
        ---
        {code[:30000]}
        ---
        
        Based on this content, please answer the following question:
        {prompt}
        """
        
        print("Sending request to Gemini API...")
        response = llm.invoke([HumanMessage(content=final_prompt)])
        print("Received response from Gemini API")
        if not response or not response.content:
            raise ValueError("Empty response from Gemini API")
        return response.content
        
    except Exception as e:
        print(f"Error analyzing with Gemini LLM: {str(e)}")
        error_msg = str(e)
        if "API key not available" in error_msg:
            return "Error: Google API key is not set. Please check your .env file."
        return f"Error: Gemini LLM analysis failed. {error_msg}"

# --- NEW Utility 3: Music Search with YouTube Data API v3 ---
def search_for_music(query: str) -> dict:
    """
    Searches YouTube for music videos using the official YouTube Data API v3.
    Returns detailed information about the found videos.
    """
    try:
        # YouTube API endpoint
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise ValueError("YouTube API key not found in environment variables")

        # Build the search URL
        base_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            'part': 'snippet',
            'q': query,
            'key': api_key,
            'maxResults': 5,
            'type': 'video',
            'videoCategoryId': '10',  # Music category
            'videoEmbeddable': 'true'
        }

        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Process results
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

        return {
            'task': 'music_search',
            'query': query,
            'results': results
        }

    except requests.exceptions.RequestException as e:
        print(f"YouTube API request failed: {e}")
        return {
            'task': 'music_search',
            'query': query,
            'error': f"Failed to connect to YouTube API: {str(e)}"
        }
    except Exception as e:
        print(f"Error searching for music: {e}")
        return {
            'task': 'music_search',
            'query': query,
            'error': f"An error occurred while searching for music: {str(e)}"
        }

# --- NEW Utility 4: Image Generation (DeepAI) ---
def generate_image(prompt: str) -> str:
    """
    Generates an image based on a detailed text description using DeepAI's text2img API.
    Provide a descriptive prompt (e.g., 'a vibrant sunset over a futuristic city').
    Returns the URL of the generated image.
    """
    try:
        import deepai
        deepai.set_api_key(os.getenv("DEEPAI_API_KEY"))
        
        # Call the DeepAI API
        response = deepai.call_standard_api("text2img", text=prompt)
        
        # Return the URL of the image
        return f"Image generated successfully! You can view it here: {response['output_url']}"
    except Exception as e:
        print(f"Error generating image: {e}")
        return f"Error: Could not generate image. {e}. Please ensure DeepAI API Key is valid."