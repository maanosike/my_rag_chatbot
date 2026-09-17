import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

def load_documents():
    loaders = [
        DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} document(s).")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s).")
    return chunks

def embed_and_store(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Stored {len(chunks)} chunk(s) in ChromaDB at '{CHROMA_DIR}'.")
    return vectorstore

if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("No documents found in 'data/'. Add PDFs or text files and try again.")
    else:
        chunks = split_documents(docs)
        embed_and_store(chunks)
        print("Ingestion complete.")
