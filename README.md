# 🌐 ExplainaAI

**Summarize and Understand Any URL Instantly**

ExplainaAI is a smart Streamlit-based chatbot that extracts and summarizes content from any YouTube video, website, or Wikipedia URL. It then allows you to ask questions based on the summarized content. This app leverages the power of **Groq's `llama3-70b-8192`** model for high-quality summarization and Q&A.

---

## 🚀 Features

- 🔗 Summarizes content from:
  - YouTube Videos
  - Web Articles
  - Wikipedia Pages
- 💬 Interactive Q&A based on the summary
- 🧠 Handles large content using chunking to prevent token overflow
- ⚡ Fast, lightweight, and easy to use
- 🔐 Powered by Groq's `llama3-70b-8192` language model

---

## 🛠️ Tech Stack

- [Python 3.11](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq API](https://console.groq.com/)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)
- [YoutubeLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/youtube)
- [UnstructuredURLLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/html)

---

## 📦 Installation

### ✅ Prerequisites

- Install Python 3.11 on your system

### 📂 Steps to Run This App

```bash
# Step 1: Install Python 3.11 (if not already installed)

# Step 2: Create a virtual environment using conda or venv
conda create -p venv python=3.11 -y

# Step 3: Activate the environment
conda activate ./venv

# Step 4: Install required Python packages
pip install -r requirements.txt

# Step 5: Run the Streamlit app
streamlit run app6.py
```
