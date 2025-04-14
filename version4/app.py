from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get HuggingFace API key from environment variables
# HF_API_KEY = os.getenv("HF_API_KEY")
# if not HF_API_KEY:
#     print("Warning: HF_API_KEY not found in environment variables!")

# Get Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment variables!")

# Configure Groq API URL
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Global store for maintaining session data (in a production app, use Redis or a database)
sessions = {}

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
            return None, None, "Invalid YouTube URL format"
            
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]
        return transcript, video_id, None
    except Exception as e:
        return None, None, f"Error extracting YouTube transcript: {str(e)}"

# Extract content from Wikipedia
def extract_wikipedia_content(wikipedia_url):
    try:
        # Extract the title from the URL
        title_match = re.search(r'wikipedia\.org/wiki/(.+)', wikipedia_url)
        if not title_match:
            return None, None, "Invalid Wikipedia URL. Please provide a link in the format: https://en.wikipedia.org/wiki/Article_Title"
            
        title = title_match.group(1)
        title = title.replace('_', ' ')
        
        # Initialize Wikipedia API
        wiki_wiki = wikipediaapi.Wikipedia('WikiSummarizerApp/1.0', 'en')
        page = wiki_wiki.page(title)
        
        if not page.exists():
            return None, None, f"Wikipedia page '{title}' does not exist or could not be found."
            
        return page.text, title, None
    except Exception as e:
        return None, None, f"Error extracting Wikipedia content: {str(e)}"

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
        
        return full_text, title, None
    except Exception as e:
        return None, None, f"Error extracting webpage content: {str(e)}"

# Generate content summary using Groq API
def generate_groq_content(content_text, prompt, api_key, model="llama3-70b-8192"):
    if not api_key:
        return "Error: API key is missing"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
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
            time.sleep(5)  # Wait 5 seconds before retrying
            return generate_groq_content(content_text, prompt, api_key, model)  # Retry
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error making API call: {str(e)}"

# Answer questions based on extracted content with memory of past conversations
def answer_question(session_id, question, api_key):
    if session_id not in sessions:
        return "Error: No active session found. Please process a URL first."
    
    session_data = sessions[session_id]
    context = session_data.get("extracted_content", "")
    summary = session_data.get("summary", "")
    
    # Get conversation history (excluding the initial system message)
    conversation_history = session_data.get("chat_history", [])[1:] if len(session_data.get("chat_history", [])) > 1 else []
    
    # Format the conversation history
    formatted_history = format_conversation_history(conversation_history)
    
    # Prepare the prompt with context, conversation history, and question
    formatted_prompt = qa_prompt.format(
        context=context[:5000],  # Limit context to avoid token limits
        summary=summary,
        conversation_history=formatted_history,
        question=question
    )
    
    if not api_key:
        return "Error: API key is missing"
    
    # Generate answer using Groq API with memory-aware prompt
    headers = {
        "Authorization": f"Bearer {api_key}",
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
def process_large_content(content, content_type, api_key):
    # Split content into chunks
    chunks = split_into_chunks(content)
    
    # Process each chunk
    chunk_summaries = []
    
    for chunk in chunks:
        chunk_summary = generate_groq_content(chunk, chunk_prompt, api_key, "llama3-8b-8192")  # Using smaller model for chunks
        chunk_summaries.append(chunk_summary)
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
        
    final_summary = generate_groq_content(combined_summaries, final_prompt, api_key, "llama3-70b-8192")
    
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
def process_url(url, api_key):
    url_type = get_url_type(url)
    
    if url_type == "youtube":
        # Process YouTube URL
        content, video_id, error = extract_transcript_details(url)
        if error:
            return None, error
        
        if content and video_id:
            content_source = f"YouTube Video (ID: {video_id})"
            page_title = "YouTube Video"
            
            # Summarize the content
            summary = summarize_content(content, url_type, api_key)
            
            # Create a new session
            session_id = generate_session_id()
            sessions[session_id] = {
                "url_type": url_type,
                "content_source": content_source,
                "extracted_content": content,
                "page_title": page_title,
                "summary": summary,
                "chat_history": [{
                    "role": "assistant", 
                    "content": f"I've analyzed the content from {content_source}. Here's a summary:\n\n{summary}\n\nYou can now ask me questions about this content!"
                }]
            }
            
            return {
                "session_id": session_id,
                "url_type": url_type,
                "content_source": content_source,
                "page_title": page_title,
                "summary": summary
            }, None
            
    elif url_type == "wikipedia":
        # Process Wikipedia URL
        content, title, error = extract_wikipedia_content(url)
        if error:
            return None, error
            
        if content:
            content_source = f"Wikipedia Article: {title}"
            page_title = title
            
            # Summarize the content
            summary = summarize_content(content, url_type, api_key)
            
            # Create a new session
            session_id = generate_session_id()
            sessions[session_id] = {
                "url_type": url_type,
                "content_source": content_source,
                "extracted_content": content,
                "page_title": page_title,
                "summary": summary,
                "chat_history": [{
                    "role": "assistant", 
                    "content": f"I've analyzed the content from {content_source}. Here's a summary:\n\n{summary}\n\nYou can now ask me questions about this content!"
                }]
            }
            
            return {
                "session_id": session_id,
                "url_type": url_type,
                "content_source": content_source,
                "page_title": page_title,
                "summary": summary
            }, None
            
    else:
        # Process general webpage
        content, page_title, error = extract_webpage_content(url)
        if error:
            return None, error
            
        if content:
            content_source = f"Webpage: {page_title}"
            
            # Summarize the content
            summary = summarize_content(content, url_type, api_key)
            
            # Create a new session
            session_id = generate_session_id()
            sessions[session_id] = {
                "url_type": url_type,
                "content_source": content_source,
                "extracted_content": content,
                "page_title": page_title,
                "summary": summary,
                "chat_history": [{
                    "role": "assistant", 
                    "content": f"I've analyzed the content from {content_source}. Here's a summary:\n\n{summary}\n\nYou can now ask me questions about this content!"
                }]
            }
            
            return {
                "session_id": session_id,
                "url_type": url_type,
                "content_source": content_source,
                "page_title": page_title,
                "summary": summary
            }, None
    
    return None, "Failed to extract content from the URL"

# Generate summary of extracted content
def summarize_content(content, url_type, api_key):
    if len(content) > 5000:  # If content is large
        summary = process_large_content(content, url_type, api_key)
    else:
        # For smaller content, process normally
        if url_type == "youtube":
            prompt = """Summarize this YouTube video transcript concisely: """
        elif url_type == "wikipedia":
            prompt = """Summarize this Wikipedia article concisely: """
        else:
            prompt = """Summarize this webpage content concisely: """
        summary = generate_groq_content(content, prompt, api_key)
    
    return summary

# Generate a unique session ID
def generate_session_id():
    import uuid
    return str(uuid.uuid4())

@app.route('/')
def index():
    return render_template('index.html')

# Route for API status check --> url to check api is running or not(check api status -> to debug while facing any problem)
@app.route('/api/status', methods=['GET']) 
def status():
    return jsonify({"status": "UP", "message": "Content Chatbot API is running"}), 200

# Route to process a URL --> the url which is given for the 1st time for the processing of the doc,YT, etc.. 
@app.route('/api/process-url', methods=['POST'])
def api_process_url():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    # Use API key from environment variable instead of request
    api_key = GROQ_API_KEY
    
    if not api_key:
        return jsonify({"error": "Groq API key not configured on the server"}), 500
    
    result, error = process_url(url, api_key)
    
    if error:
        return jsonify({"error": error}), 400
    
    return jsonify(result), 200

# Route to ask a question --> url where we ask questoins 
@app.route('/api/ask', methods=['POST'])
def api_ask_question():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    session_id = data.get('session_id')
    question = data.get('question')
    
    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    # Use API key from environment variable
    api_key = GROQ_API_KEY
    
    if not api_key:
        return jsonify({"error": "Groq API key not configured on the server"}), 500
    
    if session_id not in sessions:
        return jsonify({"error": "Invalid session ID or session expired"}), 404
    
    # Add user question to chat history
    sessions[session_id]["chat_history"].append({"role": "user", "content": question})
    
    # Generate answer
    answer = answer_question(session_id, question, api_key)
    
    # Add assistant response to chat history
    sessions[session_id]["chat_history"].append({"role": "assistant", "content": answer})
    
    return jsonify({
        "session_id": session_id,
        "answer": answer,
        "chat_history": sessions[session_id]["chat_history"]
    }), 200

# Route to get session information
@app.route('/api/session/<session_id>', methods=['GET'])
def api_get_session(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Invalid session ID or session expired"}), 404
    
    session_data = sessions[session_id]
    
    return jsonify({
        "session_id": session_id,
        "url_type": session_data.get("url_type", ""),
        "content_source": session_data.get("content_source", ""),
        "page_title": session_data.get("page_title", ""),
        "summary": session_data.get("summary", ""),
        "chat_history": session_data.get("chat_history", [])
    }), 200

# Route to clear conversation history
@app.route('/api/clear-conversation', methods=['POST'])
def api_clear_conversation():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400
    
    if session_id not in sessions:
        return jsonify({"error": "Invalid session ID or session expired"}), 404
    
    # Preserve the first message (system introduction)
    if len(sessions[session_id]["chat_history"]) > 0:
        initial_message = sessions[session_id]["chat_history"][0]
        sessions[session_id]["chat_history"] = [initial_message]
    else:
        sessions[session_id]["chat_history"] = []
    
    return jsonify({
        "session_id": session_id,
        "message": "Conversation history cleared",
        "chat_history": sessions[session_id]["chat_history"]
    }), 200

# Route to get extracted content
@app.route('/api/content/<session_id>', methods=['GET'])
def api_get_content(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Invalid session ID or session expired"}), 404
    
    return jsonify({
        "session_id": session_id,
        "extracted_content": sessions[session_id].get("extracted_content", "")
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)