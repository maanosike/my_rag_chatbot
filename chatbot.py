from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Paths
CHROMA_DIR = "chroma_db"

# Prompt template
prompt_template = """Use the following context to answer the question at the end.
If you don't know the answer, just say you don't know — do not make anything up.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore

def build_chain(vectorstore):
    llm = Ollama(model="tinyllama")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return chain

def chat():
    print("Loading vectorstore...")
    vectorstore = load_vectorstore()
    chain = build_chain(vectorstore)
    print("Chatbot ready. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break
        if not query:
            continue
        result = chain.invoke({"query": query})
        print(f"\nBot: {result['result']}")
        print("\nSources:")
        for doc in result["source_documents"]:
            print(f"  - {doc.metadata.get('source', 'unknown')}")
        print()

if __name__ == "__main__":
    chat()
