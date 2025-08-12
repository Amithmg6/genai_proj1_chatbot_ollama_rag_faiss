
print("Welcome to Chat with Machine Learning Encyclopedia.\n")

print("Genrative AI project with RAG and LLM.")

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


print("# --- Configuring the model ---#")
# Replace with the actual path to your technical book (e.g., "C:/Users/YourUser/Documents/my_tech_book.pdf")
BOOK_PATH = "Encyclopedia of Machine Learning.pdf" 
# Choose one of the small LLMs you pulled with Ollama (e.g., "gemma:2b", "phi3", "qwen:3b", "mistral")
OLLAMA_MODEL_NAME = "gemma:2b" 
# Embedding model (all-MiniLM-L6-v2 is a good default, or "nomic-embed-text" if using Ollama for embeddings)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" ## "all-mpnet-base-v2"
# Path to save/load your FAISS vector store to avoid re-embedding
FAISS_INDEX_PATH = "faiss_index_tech_book"


print("--- 1. Data Preparation ---")

def load_and_chunk_documents(book_path: str):
    """Loads a PDF document and splits it into chunks."""
    try:
        loader = PyPDFLoader(book_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from '{os.path.basename(book_path)}'.")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        print("Please ensure the book path is correct and the PDF is not corrupted.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split book into {len(chunks)} chunks.")
    return chunks


print("--- 2. Create Embeddings and Vector Store ---")
def create_vector_store(chunks, embedding_model_name: str, faiss_path: str):
    """Creates or loads a FAISS vector store."""
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    if os.path.exists(faiss_path):
        print(f"Loading existing vector store from '{faiss_path}'...")
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded.")
    else:
        print("Creating new vector store (this might take a moment)...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(faiss_path)
        print("Vector store created and saved.")
    return vectorstore


print("--- 3. Implement RAG ---")

def setup_rag_chain(llm_model_name: str, retriever):
    """Sets up the RAG chain with the specified LLM and retriever."""
    try:
        llm = Ollama(model=llm_model_name)
    except Exception as e:
        print(f"Error initializing Ollama with model '{llm_model_name}': {e}")
        print("Please ensure the Ollama server is running and the model is pulled.")
        return None

    rag_prompt_template = """Use the following context to answer the question at the end.
If you don't know the answer, state that you don't know, and do not make up an answer.

Context:
{context}

Question: {question}

Answer:"""
    RAG_PROMPT = PromptTemplate(
        template=rag_prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True, # Set to True to get the retrieved chunks
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )
    return qa_chain

print("--- Chatbot Loading ---\n")

def main():
    print("Initializing Chatbot...")

    # Load and chunk documents
    chunks = load_and_chunk_documents(BOOK_PATH)
    if not chunks:
        print("Exiting due to document loading error.")
        return

    # Create/Load vector store
    vectorstore = create_vector_store(chunks, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # Retrieve top 4 relevant chunks

    # Set up RAG chain
    qa_chain = setup_rag_chain(OLLAMA_MODEL_NAME, retriever)
    if not qa_chain:
        print("Exiting due to RAG chain setup error.")
        return

    print(f"\nChatbot initialized with '{OLLAMA_MODEL_NAME}'! Ask questions about your technical book.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break

        print("Chatbot thinking...")
        try:
            result = qa_chain.invoke({"query": user_query})
            chatbot_response = result["result"]
            source_documents = result.get("source_documents", [])

            print("\n--- Chatbot Response ---")
            print(chatbot_response)

            if source_documents:
                print("\n--- Sources Used (Top Retrieved Chunks) ---")
                for i, doc in enumerate(source_documents):
                    # Attempt to get page number if available in metadata
                    page_info = f"Page: {doc.metadata.get('page') + 1}, " if 'page' in doc.metadata else ""
                    source_filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    print(f"Chunk {i+1} ({page_info}Source: {source_filename}):")
                    print(f"  {doc.page_content[:200]}...") # Display first 200 chars of the chunk
            print("------------------------")

        except Exception as e:
            print(f"An error occurred during response generation: {e}")
            print("Please check your Ollama server and model, and ensure your prompt is well-formed.")

if __name__ == "__main__":
    main()