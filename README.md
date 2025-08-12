Chat with Machine Learning Encyclopedia
Project Overview
This project is a Generative AI application designed to provide a conversational interface for a large technical document, such as an encyclopedia or textbook. By leveraging the Retrieval-Augmented Generation (RAG) framework, the system answers user queries with information directly extracted from a provided PDF, thereby ensuring that responses are grounded in factual, relevant content. This approach prevents the Large Language Model (LLM) from hallucinating and provides verifiable answers. The application is built for local execution, offering a secure and private way to interact with proprietary or sensitive documents.

Key Features
PDF Document Ingestion: The system can load and process any PDF document, such as a technical book or encyclopedia.

Intelligent Content Retrieval: It employs an embedding model to convert document content into a numerical vector format, enabling highly efficient and semantic search.

Retrieval-Augmented Generation (RAG): Instead of relying solely on the LLM's pre-trained knowledge, the system retrieves the most relevant information from the document and uses it as context for the LLM to formulate a precise answer.

Local LLM Integration: The application is powered by a locally hosted LLM via Ollama, which ensures privacy, reduces latency, and eliminates the need for an external API key.

Source Citation: Each answer is accompanied by a reference to the specific chunks of the document used to generate the response, including the page number and filename.

Efficient Vector Store: The generated vector store is saved locally, allowing for near-instantaneous startup on subsequent runs without the need to reprocess the entire document.

How It Works: Technical Details
The project's pipeline is orchestrated using the LangChain framework and involves three main components:

Data Preparation: The PyPDFLoader handles the initial ingestion of the PDF. The document is then passed to a RecursiveCharacterTextSplitter, which divides the text into smaller, overlapping chunks. This ensures that a user's query can be mapped to a specific, manageable portion of the document.

Embeddings and Vector Search: Each text chunk is converted into a high-dimensional vector using a pre-trained SentenceTransformerEmbeddings model. These vectors are then stored in a FAISS (Facebook AI Similarity Search) index. FAISS is a library for efficient similarity search, allowing the application to quickly find the most relevant document chunks for a given user query.

Retrieval-Augmented Generation (RAG) Chain:

When a user submits a question, the system first performs a similarity search on the FAISS vector store to retrieve the top k (e.g., 4) most relevant document chunks.

These retrieved chunks are then combined with the user's question and a custom PromptTemplate to form a single, well-structured prompt.

This enhanced prompt is sent to the Ollama LLM (e.g., gemma:2b), which uses the provided context to generate a precise, factual, and non-hallucinated response.

The final output includes the LLM's answer and the specific source documents that informed it.
