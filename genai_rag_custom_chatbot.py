import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import glob # To list PDF files
import PyPDF2 # Required for viewing PDF content

print("Welcome to Chat with Machine Learning Encyclopedia.\n")
print("Generative AI project with RAG and LLM.")

# --- Configuration Section ---
print("\n# --- Configuring the LLM and Embeddings ---#")

def view_pdf_content(pdf_path: str, pages_to_view: int = 1):
    """
    Opens a PDF and prints content from the first few pages.
    Args:
        pdf_path (str): The path to the PDF file.
        pages_to_view (int): The number of pages to display from the start of the PDF.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"\n--- Viewing content from '{os.path.basename(pdf_path)}' ---")
            print(f"Total pages: {num_pages}")
            for i in range(min(pages_to_view, num_pages)):
                page = reader.pages[i]
                text = page.extract_text()
                print(f"\n--- Page {i+1} ---")
                # Print first 500 characters or less if page content is shorter
                print(text[:500] + ('...' if len(text) > 500 else ''))
            print("--- End of Preview ---")
    except PyPDF2.errors.PdfReadError:
        print(f"Error: Could not read '{pdf_path}'. It might be encrypted or corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred while viewing '{pdf_path}': {e}")


def get_available_pdfs():
    """Returns a list of PDF files in the current directory."""
    return glob.glob("*.pdf")

def choose_pdf_file():
    """
    Prompts the user to select up to 5 PDF files.
    Allows viewing PDF content before and after selection.
    """
    pdf_files = get_available_pdfs()
    if not pdf_files:
        print("No PDF files found in the current directory. Please place your PDF(s) here.")
        return []

    selected_pdfs = []
    max_selection = 5

    while len(selected_pdfs) < max_selection:
        print(f"\nAvailable PDF files (currently selected: {len(selected_pdfs)}/{max_selection}):")
        for i, pdf_file in enumerate(pdf_files):
            status = "(SELECTED)" if pdf_file in selected_pdfs else ""
            print(f"{i+1}. {pdf_file} {status}")

        print("\nEnter a number to select/deselect a PDF, 'v' to view, 'd' to finish selection, 'q' to quit.")
        user_input = input("Your choice: ").lower().strip()

        if user_input == 'q':
            print("Exiting PDF selection.")
            return []
        elif user_input == 'd':
            if not selected_pdfs:
                print("No PDFs selected. Please select at least one, or type 'q' to quit.")
                continue
            break # Done selecting
        elif user_input == 'v':
            try:
                view_choice = int(input("Enter the number of the PDF to view: "))
                if 1 <= view_choice <= len(pdf_files):
                    view_pdf_content(pdf_files[view_choice-1])
                else:
                    print("Invalid number. Please enter a valid number from the list.")
            except ValueError:
                print("Invalid input. Please enter a number for viewing.")
            continue # Continue loop after viewing

        try:
            choice_index = int(user_input) - 1
            if 0 <= choice_index < len(pdf_files):
                chosen_pdf = pdf_files[choice_index]
                if chosen_pdf in selected_pdfs:
                    selected_pdfs.remove(chosen_pdf)
                    print(f"Deselected '{chosen_pdf}'.")
                elif len(selected_pdfs) < max_selection:
                    selected_pdfs.append(chosen_pdf)
                    print(f"Selected '{chosen_pdf}'.")
                    view_pdf_content(chosen_pdf, pages_to_view=1) # View after selection
                else:
                    print(f"You can select a maximum of {max_selection} PDF files. Deselect one first.")
            else:
                print("Invalid choice. Please enter a number within the range.")
        except ValueError:
            print("Invalid input. Please enter a number, 'v', 'd', or 'q'.")

    print("\n--- Final Selected PDFs ---")
    if selected_pdfs:
        for i, pdf in enumerate(selected_pdfs):
            print(f"{i+1}. {pdf}")
            view_pdf_content(pdf, pages_to_view=1) # View content of selected PDFs
        return selected_pdfs
    else:
        print("No PDFs were selected.")
        return []

# Function to let user choose an embedding model
def choose_embedding_model():
    """Prompts the user to select an embedding model."""
    embedding_models = {
        "1": "all-MiniLM-L6-v2",
        "2": "all-mpnet-base-v2",
        "3": "nomic-embed-text" # Popular choice for Ollama embeddings
    }
    print("\nChoose an Embedding Model:")
    for key, value in embedding_models.items():
        print(f"{key}. {value}")

    while True:
        choice = input("Enter the number of your desired embedding model: ")
        if choice in embedding_models:
            return embedding_models[choice]
        else:
            print("Invalid choice. Please select a valid number.")

# Set global configuration based on user input for main execution
# This will now be a list of paths
BOOK_PATHS = choose_pdf_file() # Now it returns a list of selected PDF paths
if not BOOK_PATHS:
    exit("No PDFs selected. Exiting.")

OLLAMA_MODEL_NAME = "gemma:2b" # Still hardcoded to gemma:2b, but can be made user-selectable
EMBEDDING_MODEL_NAME = choose_embedding_model()

# For FAISS index path, create a generic one since multiple PDFs are used
FAISS_INDEX_PATH = "faiss_index_multi_pdf"

print(f"\nSelected PDF(s): {', '.join(BOOK_PATHS)}")
print(f"Selected Embedding Model: {EMBEDDING_MODEL_NAME}")
print(f"Ollama LLM Model: {OLLAMA_MODEL_NAME}")
print(f"FAISS Index Path: {FAISS_INDEX_PATH}")


# --- 1. Data Preparation ---
print("\n--- 1. Data Preparation ---")

def load_and_chunk_documents(book_path: str):
    """
    Loads a PDF document and splits it into manageable chunks.
    These chunks are used for creating embeddings and retrieving context.
    """
    try:
        loader = PyPDFLoader(book_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from '{os.path.basename(book_path)}'.")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        print("Please ensure the book path is correct and the PDF is not corrupted.")
        return []

    # Defines how to split the text: chunk_size is max characters per chunk,
    # chunk_overlap ensures continuity between chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split book into {len(chunks)} chunks.")
    return chunks


# --- 2. Create Embeddings and Vector Store ---
print("\n--- 2. Create Embeddings and Vector Store ---")

def create_vector_store(chunks, embedding_model_name: str, faiss_path: str):
    """
    Creates or loads a FAISS vector store.
    FAISS stores vector embeddings of text chunks for efficient similarity search.
    If an index already exists, it loads it; otherwise, it creates and saves a new one.
    """
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    if os.path.exists(faiss_path):
        print(f"Loading existing vector store from '{faiss_path}'...")
        # allow_dangerous_deserialization=True is needed for loading local FAISS indexes
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded.")
    else:
        print("Creating new vector store (this might take a moment)...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(faiss_path)
        print("Vector store created and saved.")
    return vectorstore


# --- 3. Implement RAG Chain ---
print("\n--- 3. Implement RAG Chain ---")

def setup_rag_chain(llm_model_name: str, retriever):
    """
    Sets up the Retrieval Augmented Generation (RAG) chain.
    This chain connects the LLM with the vector store (retriever)
    to answer questions using context from the documents.
    """
    try:
        # Initialize the Ollama LLM (ensure Ollama server is running)
        llm = Ollama(model=llm_model_name)
    except Exception as e:
        print(f"Error initializing Ollama with model '{llm_model_name}': {e}")
        print("Please ensure the Ollama server is running and the model is pulled (e.g., 'ollama pull gemma:2b').")
        return None

    # Define the prompt template for the RAG chain
    # {context} will be filled by retrieved document chunks
    # {question} will be the user's query
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

    # Create the RetrievalQA chain
    # chain_type="stuff" means all retrieved documents are "stuffed" into the prompt.
    # return_source_documents=True allows seeing which parts of the PDF were used.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )
    return qa_chain

# --- Main Chatbot Logic ---
print("\n--- Chatbot Loading ---\n")

def main():
    """Main function to run the chatbot."""
    print("Initializing Chatbot...")

    # Step 1: Load and chunk documents from potentially multiple PDFs
    all_chunks = []
    for book_path in BOOK_PATHS:
        print(f"\nProcessing: {os.path.basename(book_path)}")
        chunks_from_pdf = load_and_chunk_documents(book_path)
        if chunks_from_pdf: # Only extend if chunks were successfully loaded
            all_chunks.extend(chunks_from_pdf)
    
    if not all_chunks:
        print("Exiting due to no content loaded from selected PDFs.")
        return

    # Step 2: Create/Load vector store with chosen embedding model using all chunks
    vectorstore = create_vector_store(all_chunks, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH)
    # Configure the retriever to fetch the top 4 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Step 3: Set up the RAG chain with the Ollama LLM
    qa_chain = setup_rag_chain(OLLAMA_MODEL_NAME, retriever)
    if not qa_chain:
        print("Exiting due to RAG chain setup error.")
        return

    print(f"\nChatbot initialized with '{OLLAMA_MODEL_NAME}' and '{EMBEDDING_MODEL_NAME}' embeddings, using {len(BOOK_PATHS)} PDF(s)!")
    print("Type 'exit' to quit.")

    # --- Chat Loop ---
    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break

        print("Chatbot thinking...")
        try:
            # Invoke the RAG chain with the user's query
            result = qa_chain.invoke({"query": user_query})
            chatbot_response = result["result"]
            source_documents = result.get("source_documents", [])

            print("\n--- Chatbot Response ---")
            print(chatbot_response)

            if source_documents:
                print("\n--- Sources Used (Top Retrieved Chunks) ---")
                for i, doc in enumerate(source_documents):
                    # Display page number if available in document metadata
                    page_info = f"Page: {doc.metadata.get('page') + 1}, " if 'page' in doc.metadata else ""
                    source_filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    print(f"Chunk {i+1} ({page_info}Source: {source_filename}):")
                    # Display first 200 characters of the content for brevity
                    print(f"  {doc.page_content[:200]}...")
            print("------------------------")

        except Exception as e:
            print(f"An error occurred during response generation: {e}")
            print("Please check your Ollama server and model, and ensure your prompt is well-formed.")

if __name__ == "__main__":
    main()