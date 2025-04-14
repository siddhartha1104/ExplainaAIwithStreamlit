import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # load all the environment variables
import os
import requests
import json
from youtube_transcript_api import YouTubeTranscriptApi
import wikipediaapi
import re
import time
from bs4 import BeautifulSoup
import urllib.parse

# Configure Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Function to split text into chunks of approximately equal size
def split_into_chunks(text, max_chunk_size=4000):
    """Split text into chunks of approximately max_chunk_size characters."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        # Add word length plus space
        if current_size + len(word) + 1 > max_chunk_size and current_chunk:
            # If adding this word would exceed the limit, save current chunk and start a new one
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word) + 1
        else:
            # Add word to current chunk
            current_chunk.append(word)
            current_size += len(word) + 1
            
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks

# Prompts for different content types
chunk_prompt = """You are summarizing a part of a larger content. Summarize this section concisely, focusing on key facts, arguments, and information. Don't try to introduce or conclude the entire topic, just focus on this specific section:

"""

final_youtube_prompt = """You are an expert YouTube video summarizer with exceptional attention to detail.

Below are summaries of different parts of a YouTube video transcript. Your task is to create a final, coherent summary that integrates all these sections into one comprehensive summary that captures:
1. The main topic and purpose of the video
2. Key points, insights, and arguments presented
3. Important facts, statistics, and examples mentioned
4. Any conclusions or recommendations

Please format your summary as follows:
- Begin with a brief overview of the video's main topic (1-2 sentences)
- Follow with structured bullet points highlighting the most important information
- Ensure no significant details are omitted
- Maintain the original meaning and intent of the content
- Keep the entire summary within 300-400 words for readability while preserving comprehensive coverage

The section summaries are as follows:

"""

final_webpage_prompt = """You are an expert web content summarizer with exceptional attention to detail.

Below are summaries of different parts of a webpage. Your task is to create a final, coherent summary that integrates all these sections into one comprehensive summary that captures:
1. The main subject and purpose of the webpage
2. Key points, arguments, and information presented
3. Important facts, statistics, and examples mentioned
4. Any conclusions, recommendations, or calls to action

Please format your summary as follows:
- Begin with a brief overview of the webpage's main topic (1-2 sentences)
- Follow with structured bullet points highlighting the most important information
- Ensure no significant details are omitted
- Maintain the original meaning and intent of the content
- Keep the entire summary within 300-400 words for readability while preserving comprehensive coverage

The section summaries are as follows:

"""

final_wikipedia_prompt = """You are an expert Wikipedia article summarizer with exceptional attention to detail.

Below are summaries of different parts of a Wikipedia article. Your task is to create a final, coherent summary that integrates all these sections into one comprehensive summary that captures:
1. The main subject and significance
2. Key facts, definitions, and historical information
3. Important developments, relationships, and concepts
4. Notable controversies or alternative viewpoints (if any)

Please format your summary as follows:
- Begin with a brief overview of the article's main subject (1-2 sentences)
- Follow with structured bullet points highlighting the most important information
- Ensure no significant details are omitted
- Maintain the original meaning and intent of the content
- Keep the entire summary within 300-400 words for readability while preserving comprehensive coverage

The section summaries are as follows:

"""

# Extract YouTube Transcript
def extract_transcript_details(youtube_video_url):
    try:
        if "youtube.com" in youtube_video_url and "=" in youtube_video_url:
            video_id = youtube_video_url.split("=")[1]
        elif "youtu.be" in youtube_video_url:
            video_id = youtube_video_url.split("/")[-1]
        else:
            st.error("Invalid YouTube URL format")
            return None
            
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]
        return transcript, video_id
    except Exception as e:
        st.error(f"Error extracting YouTube transcript: {str(e)}")
        return None, None

# Extract content from Wikipedia
def extract_wikipedia_content(wikipedia_url):
    try:
        # Extract the title from the URL
        title_match = re.search(r'wikipedia\.org/wiki/(.+)', wikipedia_url)
        if not title_match:
            st.error("Invalid Wikipedia URL. Please provide a link in the format: https://en.wikipedia.org/wiki/Article_Title")
            return None
            
        title = title_match.group(1)
        title = title.replace('_', ' ')
        
        # Initialize Wikipedia API
        wiki_wiki = wikipediaapi.Wikipedia('WikiSummarizerApp/1.0', 'en')
        page = wiki_wiki.page(title)
        
        if not page.exists():
            st.error(f"Wikipedia page '{title}' does not exist or could not be found.")
            return None
            
        return page.text
    except Exception as e:
        st.error(f"Error extracting Wikipedia content: {str(e)}")
        return None

# Extract content from any general webpage
def extract_webpage_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "No title found"
        
        # Remove script, style elements and comments
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            element.decompose()
            
        # Extract text from paragraphs, headings, and lists
        content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        
        content = []
        for element in content_elements:
            text = element.get_text(strip=True)
            if text and len(text) > 20:  # Filter out very short texts
                content.append(text)
                
        # Join all paragraphs with newlines
        full_text = "\n\n".join(content)
        
        # Get the webpage favicon or domain icon
        domain = urllib.parse.urlparse(url).netloc
        favicon_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
        
        return full_text, title, favicon_url
    except Exception as e:
        st.error(f"Error extracting webpage content: {str(e)}")
        return None, None, None

# Generate content summary using Groq API
def generate_groq_content(content_text, prompt, model="llama3-70b-8192"):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert content summarizer that extracts comprehensive yet concise information from provided text."
            },
            {
                "role": "user",
                "content": prompt + content_text
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        elif response.status_code == 429:  # Rate limit error
            st.warning("Rate limit reached. Waiting before retrying...")
            time.sleep(5)  # Wait 5 seconds before retrying
            return generate_groq_content(content_text, prompt, model)  # Retry
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error making API call: {str(e)}"

# Process content in chunks
def process_large_content(content, content_type):
    st.write("Content is large. Processing in chunks...")
    
    # Split content into chunks
    chunks = split_into_chunks(content)
    st.write(f"Split into {len(chunks)} chunks.")
    
    # Process each chunk
    chunk_summaries = []
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(chunks):
        st.write(f"Processing chunk {i+1}/{len(chunks)}...")
        chunk_summary = generate_groq_content(chunk, chunk_prompt, "llama3-8b-8192")  # Using smaller model for chunks
        chunk_summaries.append(chunk_summary)
        progress_bar.progress((i + 1) / len(chunks))
        # Add a delay to respect rate limits
        time.sleep(1)
    
    # Combine chunk summaries
    combined_summaries = "\n\n--- SECTION SUMMARY " + " ---\n\n--- SECTION SUMMARY ".join(chunk_summaries) + " ---\n\n"
    
    # Generate final summary based on content type
    if content_type == "youtube":
        final_prompt = final_youtube_prompt
    elif content_type == "wikipedia":
        final_prompt = final_wikipedia_prompt
    else:
        final_prompt = final_webpage_prompt
        
    final_summary = generate_groq_content(combined_summaries, final_prompt, "llama3-70b-8192")
    return final_summary

# Determine URL type
def get_url_type(url):
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    elif "wikipedia.org" in url:
        return "wikipedia"
    else:
        return "webpage"

# Main Streamlit app
st.title("Universal Content Summarizer")
st.write("Enter any URL (YouTube video, Wikipedia article, or general webpage) to get a detailed summary.")

# Link input
url = st.text_input("Enter URL:")

if url:
    # Detect URL type
    url_type = get_url_type(url)
    
    if st.button("Get Summary"):
        with st.spinner(f"Processing {url_type} content..."):
            if url_type == "youtube":
                # Process YouTube URL
                content, video_id = extract_transcript_details(url)
                if content and video_id:
                    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)
            elif url_type == "wikipedia":
                # Process Wikipedia URL
                content = extract_wikipedia_content(url)
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Wikipedia-logo-v2.svg/200px-Wikipedia-logo-v2.svg.png", width=100)
            else:
                # Process general webpage
                content, page_title, favicon_url = extract_webpage_content(url)
                if content:
                    st.write(f"Extracted content from: **{page_title}**")
                    st.image(favicon_url, width=32)
                
            if content:
                # Check content length to determine processing method
                if len(content) > 5000:  # If content is large
                    summary = process_large_content(content, url_type)
                else:
                    # For smaller content, process normally
                    with st.spinner("Generating summary..."):
                        if url_type == "youtube":
                            prompt = """Summarize this YouTube video transcript concisely: """
                        elif url_type == "wikipedia":
                            prompt = """Summarize this Wikipedia article concisely: """
                        else:
                            prompt = """Summarize this webpage content concisely: """
                        summary = generate_groq_content(content, prompt)
                
                st.markdown("## Summary:")
                st.write(summary)
            else:
                st.error("Failed to extract content from the provided URL.")