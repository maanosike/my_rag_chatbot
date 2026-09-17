# RAG Chatbot
A local RAG chatbot built with LangChain, ChromaDB, and Ollama (TinyLlama).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
1. Add PDFs or text files to `data/`
2. Run `python3 ingest.py` to process documents
3. Run `python3 chatbot.py` to start chatting
