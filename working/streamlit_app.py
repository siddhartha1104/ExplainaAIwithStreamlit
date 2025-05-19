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

# Import Selenium for dynamic website scraping
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configure Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Initialize session state for chat history and content storage
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'extracted_content' not in st.session_state:
    st.session_state.extracted_content = ""
if 'content_source' not in st.session_state:
    st.session_state.content_source = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'url_processed' not in st.session_state:
    st.session_state.url_processed = False
if 'url_type' not in st.session_state:
    st.session_state.url_type = None
if 'page_title' not in st.session_state:
    st.session_state.page_title = ""

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

# Updated QA prompt that includes conversation history
qa_prompt = """You are an AI assistant that answers questions based on the content provided and remembers previous conversation. 
You have been given context information extracted from a URL and the conversation history so far.
Answer the user's question based on the provided context information and taking into account the previous conversation.
If the answer cannot be determined from the provided context or conversation history, acknowledge that you don't have enough information to answer accurately rather than making up information.
Be concise, helpful, and accurate in your responses.

CONTEXT INFORMATION:
{context}

SUMMARY OF CONTEXT:
{summary}

CONVERSATION HISTORY:
{conversation_history}

Now answer the following question based on the above context and conversation history:
{question}
"""

# Function to format conversation history for the prompt
def format_conversation_history(chat_history):
    if not chat_history:
        return "No previous conversation."
    
    formatted_history = ""
    for i, message in enumerate(chat_history):
        role = "User" if message["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {message['content']}\n\n"
    
    return formatted_history

# Extract YouTube Transcript
def extract_transcript_details(youtube_video_url):
    try:
        if "youtube.com" in youtube_video_url and "=" in youtube_video_url:
            video_id = youtube_video_url.split("=")[1]
        elif "youtu.be" in youtube_video_url:
            video_id = youtube_video_url.split("/")[-1]
        else:
            st.error("Invalid YouTube URL format")
            return None, None
            
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
            return None, None
            
        title = title_match.group(1)
        title = title.replace('_', ' ')
        
        # Initialize Wikipedia API
        wiki_wiki = wikipediaapi.Wikipedia('WikiSummarizerApp/1.0', 'en')
        page = wiki_wiki.page(title)
        
        if not page.exists():
            st.error(f"Wikipedia page '{title}' does not exist or could not be found.")
            return None, None
            
        return page.text, title
    except Exception as e:
        st.error(f"Error extracting Wikipedia content: {str(e)}")
        return None, None

# Extract content from dynamic webpage using Selenium
def extract_dynamic_webpage_content(url, wait_time=5):
    try:
        with st.spinner("Loading dynamic content with Selenium..."):
            # Configure Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            
            # Set up the driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Navigate to the URL
            driver.get(url)
            
            # Wait for the dynamic content to load
            time.sleep(wait_time)
            
            # Extract the page title
            title = driver.title
            
            # Get the fully rendered page source
            page_source = driver.page_source
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            
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
            
            # Close the driver
            driver.quit()
            
            return full_text, title, favicon_url
    except Exception as e:
        st.error(f"Error extracting dynamic webpage content: {str(e)}")
        if 'driver' in locals():
            driver.quit()
        return None, None, None

# Extract content from any general webpage with BeautifulSoup
def extract_static_webpage_content(url):
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
        st.error(f"Error extracting static webpage content: {str(e)}")
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

# Answer questions based on extracted content with memory of past conversations
def answer_question(question):
    context = st.session_state.extracted_content
    summary = st.session_state.summary
    
    # Get conversation history (excluding the current question and initial system message)
    conversation_history = st.session_state.chat_history[1:] if len(st.session_state.chat_history) > 1 else []
    
    # Format the conversation history
    formatted_history = format_conversation_history(conversation_history)
    
    # Prepare the prompt with context, conversation history, and question
    formatted_prompt = qa_prompt.format(
        context=context[:5000],  # Limit context to avoid token limits
        summary=summary,
        conversation_history=formatted_history,
        question=question
    )
    
    # Generate answer using Groq API with memory-aware prompt
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant that answers questions based on content and remembers past conversation."
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error making API call: {str(e)}"

# Process content in chunks
def process_large_content(content, content_type):
    with st.status("Processing content in chunks...") as status:
        # Split content into chunks
        chunks = split_into_chunks(content)
        status.update(label=f"Split into {len(chunks)} chunks")
        
        # Process each chunk
        chunk_summaries = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            status.update(label=f"Processing chunk {i+1}/{len(chunks)}...")
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
            
        status.update(label="Generating final summary...")
        final_summary = generate_groq_content(combined_summaries, final_prompt, "llama3-70b-8192")
        status.update(label="Processing complete!", state="complete")
        
        return final_summary

# Determine URL type
def get_url_type(url):
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    elif "wikipedia.org" in url:
        return "wikipedia"
    else:
        return "webpage"

# Process URL and extract content
def process_url(url):
    url_type = get_url_type(url)
    st.session_state.url_type = url_type
    
    with st.status(f"Processing {url_type} content...") as status:
        if url_type == "youtube":
            # Process YouTube URL
            content, video_id = extract_transcript_details(url)
            if content and video_id:
                st.session_state.content_source = f"YouTube Video (ID: {video_id})"
                st.session_state.extracted_content = content
                st.session_state.page_title = "YouTube Video"
                status.update(label="YouTube transcript extracted", state="complete")
                return True
        elif url_type == "wikipedia":
            # Process Wikipedia URL
            content, title = extract_wikipedia_content(url)
            if content:
                st.session_state.content_source = f"Wikipedia Article: {title}"
                st.session_state.extracted_content = content
                st.session_state.page_title = title
                status.update(label="Wikipedia content extracted", state="complete")
                return True
        else:
            # Process general webpage - try dynamic first, then static if it fails
            try_dynamic = st.sidebar.checkbox("Use Selenium for dynamic content", value=True)
            
            if try_dynamic:
                content, page_title, favicon_url = extract_dynamic_webpage_content(url)
                if not content:  # If dynamic extraction failed, try static
                    status.update(label="Dynamic extraction failed, trying static method...")
                    content, page_title, favicon_url = extract_static_webpage_content(url)
            else:
                content, page_title, favicon_url = extract_static_webpage_content(url)
                
            if content:
                st.session_state.content_source = f"Webpage: {page_title}"
                st.session_state.extracted_content = content
                st.session_state.page_title = page_title
                status.update(label=f"Content extracted from {page_title}", state="complete")
                return True
        
        status.update(label="Failed to extract content", state="error")
        return False

# Generate summary of extracted content
def summarize_content():
    content = st.session_state.extracted_content
    url_type = st.session_state.url_type
    
    if len(content) > 5000:  # If content is large
        summary = process_large_content(content, url_type)
    else:
        # For smaller content, process normally
        with st.status("Generating summary...") as status:
            if url_type == "youtube":
                prompt = """Summarize this YouTube video transcript concisely: """
            elif url_type == "wikipedia":
                prompt = """Summarize this Wikipedia article concisely: """
            else:
                prompt = """Summarize this webpage content concisely: """
            summary = generate_groq_content(content, prompt)
            status.update(label="Summary generated", state="complete")
    
    st.session_state.summary = summary
    return summary

# Function to clear conversation history
def clear_conversation():
    # Preserve the first message (system introduction)
    if len(st.session_state.chat_history) > 0:
        initial_message = st.session_state.chat_history[0]
        st.session_state.chat_history = [initial_message]
    else:
        st.session_state.chat_history = []
    st.success("Conversation history cleared!")

# Main Streamlit app
st.title("Explaina AI with Vector Memory")
st.write("Enter any URL to extract content and chat with Explaina AI. The Explaina AI will remember your conversation history!")

# Sidebar for URL input and processing
with st.sidebar:
    st.markdown("""
    ### Developed by: 
    [**Siddhartha Pathak**](https://www.siddharthapathak.com.np)
    """, unsafe_allow_html=True)

    st.divider()
    
    st.header("Content Source")
    url = st.text_input("Enter URL (YouTube, Wikipedia, or any webpage):")
    
    # New option for scraping method
    scraping_method = st.radio(
        "Select scraping method:",
        ["Auto (Try dynamic first, then static)", "Static only (BeautifulSoup)", "Dynamic only (Selenium)"]
    )
    
    # Dynamic scraping settings
    if scraping_method != "Static only (BeautifulSoup)":
        wait_time = st.slider("Wait time for dynamic content (seconds)", 1, 20, 5)
    
    if st.button("Process URL"):
        st.session_state.url_processed = False
        st.session_state.chat_history = []
        
        # Adjust the process_url function call based on the selected method
        if process_url(url):
            summary = summarize_content()
            st.session_state.url_processed = True
            
            # Add system message to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"I've analyzed the content from {st.session_state.content_source}. Here's a summary:\n\n{summary}\n\nYou can now ask me questions about this content!"
            })
    
    if st.session_state.url_processed:
        st.success(f"Content processed: {st.session_state.page_title}")
        
        # Display content source info
        st.subheader("Content Source")
        st.write(st.session_state.content_source)
        
        if st.session_state.url_type == "youtube" and "YouTube Video (ID:" in st.session_state.content_source:
            video_id = st.session_state.content_source.split("ID: ")[1].strip(")")
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)
        elif st.session_state.url_type == "wikipedia":
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Wikipedia-logo-v2.svg/200px-Wikipedia-logo-v2.svg.png", width=100)

        # Button to view the full extracted content
        if st.button("View Full Extracted Content"):
            st.text_area("Raw Extracted Content", st.session_state.extracted_content, height=300)
            
        # Button to view the summary
        if st.button("View Summary"):
            st.text_area("Content Summary", st.session_state.summary, height=300)
        
        # Button to clear conversation history
        if st.button("Clear Conversation History"):
            clear_conversation()

# Chat interface
st.divider()
st.subheader("Chat with the Content")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if st.session_state.url_processed:
    user_question = st.chat_input("Ask a question about the content...")
    
    if user_question:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display user question
        with st.chat_message("user"):
            st.write(user_question)
        
        # Generate and display answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(user_question)
                st.write(answer)
                
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Please enter a URL first to start chatting about its content.")