import os
import time
from pathlib import Path
from typing import Optional
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Environment setup
IS_DEV = os.getenv("ENVIRONMENT", "production").lower() == "development"

# Constants
VECTOR_STORE_DIR = "vector_store"
DOCS_DIR = "docs"
CHUNK_SIZE = 300  # Reduced for faster processing
CHUNK_OVERLAP = 30  # Reduced for faster processing
BATCH_SIZE = 10  # Increased batch size to reduce number of API calls
MAX_RETRIES = 3

# Custom styling
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #FF8C00;
        color: #4A4A4A;
    }
    
    /* Title styling */
    .title-container {
        background-color: #FF8C00;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .title-text {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Input box styling */
    .stTextInput input {
        background-color: #FFF5E6;
        border: 2px solid #8B4513;
        border-radius: 10px;
        padding: 10px;
        font-size: 1.1rem;
            color: black;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #8B4513;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #6B3410;
        transform: translateY(-2px);
    }
    
    /* Answer container styling */
    .answer-container {
        background-color: #FFF5E6;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #8B4513;
        margin: 1rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #8B4513;
        color: white;
        border-radius: 10px;
    }
    
    /* Progress bar color */
    .stProgress > div > div {
        background-color: #8B4513;
    }
</style>
""", unsafe_allow_html=True)

def test_google_connection():
    """Test connection to Google API."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            if IS_DEV:
                st.error("❌ GOOGLE_API_KEY not found in environment variables")
            return False
        
        if IS_DEV:
            st.info("🔄 Testing connection to Google API...")
        
        # Configure the Google API
        genai.configure(api_key=api_key)
        
        # Try to list models as a connection test
        models = genai.list_models()
        if any(model.name == "models/embedding-001" for model in models):
            if IS_DEV:
                st.success("✅ Successfully connected to Google API!")
            return True
        else:
            if IS_DEV:
                st.error("❌ Could not find required embedding model")
            return False
            
    except Exception as e:
        if IS_DEV:
            st.error(f"❌ Failed to connect to Google API: {str(e)}")
        return False

def setup_directories():
    """Create necessary directories if they don't exist."""
    try:
        Path(VECTOR_STORE_DIR).mkdir(exist_ok=True)
        Path(DOCS_DIR).mkdir(exist_ok=True)
        if IS_DEV:
            st.text(f"Created/verified directories: {VECTOR_STORE_DIR}, {DOCS_DIR}")
    except Exception as e:
        if IS_DEV:
            st.error(f"Error creating directories: {str(e)}")

def get_embeddings():
    """Initialize and return the Gemini embeddings model."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

def get_llm():
    """Initialize and return the Gemini LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.7,
        convert_system_message_to_human=True
    )

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type((Exception))
)
def generate_embeddings_with_retry(embeddings, batch):
    """Generate embeddings with retry logic."""
    try:
        if isinstance(batch, list) and len(batch) == 0:
            return None
        return FAISS.from_documents(
            documents=batch,
            embedding=embeddings
        )
    except Exception as e:
        st.text(f"Retrying due to error: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type((Exception))
)
def add_documents_with_retry(vectorstore, batch):
    """Add documents to vectorstore with retry logic."""
    try:
        if isinstance(batch, list) and len(batch) == 0:
            return
        vectorstore.add_documents(batch)
    except Exception as e:
        st.text(f"Retrying due to error: {str(e)}")
        raise

def process_documents():
    """Load and process documents from the docs directory using manual reading."""
    vectorstore = None
    file_path = os.path.join(DOCS_DIR, "sign_docs.txt")  # Explicitly target sign_docs.txt
    
    try:
        st.info("🔄 Starting document processing...") if IS_DEV else None
        processing_start_time = time.time()

        # --- Start Manual Loading --- 
        if IS_DEV:
            st.text(f"Attempting to manually load document from: {os.path.abspath(file_path)}")
        load_start_time = time.time()
        
        if not os.path.exists(file_path):
            if IS_DEV:
                st.error(f"❌ File not found: {file_path}")
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            load_end_time = time.time()
            if IS_DEV:
                st.text(f"✅ File content read successfully in {load_end_time - load_start_time:.2f} seconds.")
        except Exception as read_error:
            if IS_DEV:
                st.error(f"❌ Error reading file content: {str(read_error)}")
            return None
            
        # Create a single LangChain Document object
        documents = [
            Document(page_content=file_content, metadata={"source": file_path})
        ]
        
        if IS_DEV:
            total_docs = len(documents)
            st.success(f"✅ Successfully loaded {total_docs} document object.")
            st.text("Splitting document into chunks...")
        split_start_time = time.time()
        
        # --- Start Splitting --- 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        split_end_time = time.time()
        if IS_DEV:
            st.text(f"✅ Splitting completed in {split_end_time - split_start_time:.2f} seconds.")
        
        if IS_DEV:
            total_chunks = len(splits)
            st.success(f"✅ Created {total_chunks} chunks.")
            st.text("Generating embeddings and adding to vector store in batches...")
        embed_start_time = time.time()
        
        # --- Start Embedding/Storing --- 
        embeddings = get_embeddings()
        
        if IS_DEV:
            progress_bar = st.progress(0)
        error_count = 0
        
        for i in range(0, len(splits), BATCH_SIZE):
            try:
                batch = splits[i:i + BATCH_SIZE]
                if IS_DEV:
                    st.text(f"Processing batch {i // BATCH_SIZE + 1}/{ (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE } (chunks {i+1}-{min(i+BATCH_SIZE, len(splits))})...")
                batch_start_time = time.time()
                if i == 0:
                    vectorstore = generate_embeddings_with_retry(embeddings, batch)
                else:
                    if vectorstore:
                       add_documents_with_retry(vectorstore, batch)
                    else:
                        if IS_DEV:
                            st.error("Vector store not initialized. Skipping batch.")
                        error_count +=1
                        continue
                
                if IS_DEV:
                    batch_end_time = time.time()
                    st.text(f"Batch {i // BATCH_SIZE + 1} processed in {batch_end_time - batch_start_time:.2f} seconds.")
                    progress = min(1.0, (i + BATCH_SIZE) / len(splits))
                    progress_bar.progress(progress)
                    time.sleep(0.5)
                
            except Exception as e:
                error_count += 1
                if IS_DEV:
                    st.error(f"Error processing batch {i//BATCH_SIZE + 1}: {str(e)}")
                if error_count >= MAX_RETRIES:
                    if IS_DEV:
                        st.error("Too many errors occurred during batch processing. Aborting.")
                    return None
                if IS_DEV:
                    st.text("Continuing to next batch...")
                continue
        
        embed_end_time = time.time()

        if vectorstore:
            persist_start_time = time.time()
            if IS_DEV:
                st.text("Persisting vector store...")
            vectorstore.persist()
            persist_end_time = time.time()
            if IS_DEV:
                st.text(f"✅ Vector store persisted in {persist_end_time - persist_start_time:.2f} seconds.")
                progress_bar.progress(1.0)
                st.success(f"✅ Document processing complete! Total time: {persist_end_time - processing_start_time:.2f} seconds.")
            return vectorstore
        else:
            if IS_DEV:
                st.error("❌ Vector store could not be created or populated.")
            return None
        
    except Exception as e:
        if IS_DEV:
            st.error(f"❌ An unexpected error occurred during document processing: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
        return None

def load_documents():
    """Load documents from the docs directory."""
    try:
        file_path = os.path.join(DOCS_DIR, "sign_docs.txt")
        abs_path = os.path.abspath(file_path)
        
        # Detailed logging only in dev mode
        if IS_DEV:
            st.text("=== Document Loading Debug Info ===")
            st.text(f"Attempting to load file: {abs_path}")
            st.text(f"File exists check: {os.path.exists(file_path)}")
        
        if not os.path.exists(file_path):
            if IS_DEV:
                st.error(f"File not found: {file_path}")
            return None
            
        try:
            if IS_DEV:
                st.text("Attempting to read file contents...")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if IS_DEV:
                st.text(f"Successfully read file. Content length: {len(content)} characters")
            
            if len(content.strip()) == 0:
                if IS_DEV:
                    st.error("File is empty!")
                return None
                
        except Exception as read_error:
            if IS_DEV:
                st.error(f"Error reading file: {str(read_error)}")
                st.text(f"Error type: {type(read_error)}")
            return None
            
        if IS_DEV:
            st.text("Creating document object...")
        
        # Create document with metadata
        document = Document(
            page_content=content,
            metadata={"source": file_path}
        )
        
        if IS_DEV:
            st.text("Initializing text splitter...")
        
        # Split the document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        
        if IS_DEV:
            st.text("Splitting document...")
        splits = text_splitter.split_documents([document])
        if IS_DEV:
            st.text(f"Document split into {len(splits)} chunks")
        
        return splits
        
    except Exception as e:
        if IS_DEV:
            st.error("=== Document Loading Error ===")
            st.error(f"Error type: {type(e)}")
            st.error(f"Error message: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
        return None

def get_vectorstore() -> Optional[FAISS]:
    """Get or create the vector store."""
    try:
        if IS_DEV:
            st.text("=== Vector Store Initialization ===")
            st.text("Initializing embeddings...")
        embeddings = get_embeddings()
        
        if IS_DEV:
            st.text("Loading documents...")
        documents = load_documents()
        if not documents:
            if IS_DEV:
                st.error("Document loading failed - no documents returned")
            else:
                st.error("Unable to initialize the knowledge base. Please try again later.")
            return None
        
        if IS_DEV:
            st.text(f"Creating vector store with {len(documents)} documents...")
        try:
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=embeddings
            )
            if IS_DEV:
                st.text("Vector store created successfully")
            return vectorstore
        except Exception as ve:
            if IS_DEV:
                st.error("=== Vector Store Creation Error ===")
                st.error(f"Error type: {type(ve)}")
                st.error(f"Error message: {str(ve)}")
                import traceback
                st.text(traceback.format_exc())
            else:
                st.error("Unable to initialize the knowledge base. Please try again later.")
            return None
            
    except Exception as e:
        if IS_DEV:
            st.error("=== General Vector Store Error ===")
            st.error(f"Error type: {type(e)}")
            st.error(f"Error message: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
        else:
            st.error("Unable to initialize the knowledge base. Please try again later.")
        return None

def create_qa_chain(vectorstore: FAISS):
    """Create the RAG QA chain."""
    llm = get_llm()
    
    # Create prompt template
    prompt_template = """You are a helpful assistant for Sign's Orange Dynasty community. Use the following pieces of context to answer the question. 
    If the exact answer isn't found in the context, try to:
    1. Provide relevant information from the context that might be helpful
    2. Suggest where the user might find more information (e.g., Sign's social media, community channels)
    3. Explain what is known from the context that's related to their question

    Remember to maintain a friendly, supportive tone aligned with Sign's community values.

    Context:
    {context}

    Question: {question}

    Answer: Let me help you with that. """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),  # Increased from 4 to 6 for more context
        chain_type_kwargs={
            "prompt": PROMPT,
            "verbose": IS_DEV
        },
        return_source_documents=True
    )
    
    return chain

def main():
    # Debug information about environment and files - only in dev mode
    if IS_DEV:
        st.text(f"Current working directory: {os.getcwd()}")
        st.text(f"Contents of current directory: {os.listdir('.')}")
        if os.path.exists(DOCS_DIR):
            st.text(f"Contents of {DOCS_DIR} directory: {os.listdir(DOCS_DIR)}")
        else:
            st.text(f"'{DOCS_DIR}' directory does not exist!")
    
    # Custom title with styling
    st.markdown('<div class="title-container"><h1 class="title-text">Sign OrangePrint Q&A</h1></div>', unsafe_allow_html=True)
    
    # Test Google API connection first - this must be the first operation
    connection_status = test_google_connection()
    
    if not connection_status:
        if IS_DEV:
            st.error("❌ Cannot proceed without Google API connection. Please check your internet connection and API key configuration.")
        else:
            st.error("Sorry, the service is temporarily unavailable. Please try again later.")
        st.stop()
    
    st.markdown("""
    <div style="background-color: #FFF5E6; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
    Welcome to the Sign OrangePrint Q&A! Ask any questions about the OrangePrint documents, and I'll help you find the answers.
    </div>
    """, unsafe_allow_html=True)
    
    # Get vector store
    vectorstore = get_vectorstore()
    if not vectorstore:
        return
    
    # Create QA chain
    qa_chain = create_qa_chain(vectorstore)
    
    # Create question input
    question = st.text_input("What would you like to know about Sign?")
    
    if st.button("Get Answer") and question:
        with st.spinner("Searching through the OrangePrint..."):
            try:
                # Get answer
                result = qa_chain.invoke({"query": question})
                
                # Display answer with custom styling
                st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                st.markdown("### 🔍 Answer:")
                st.write(result["result"])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display sources only in development mode
                if IS_DEV:
                    with st.expander("📚 View Source Documents"):
                        for doc in result["source_documents"]:
                            st.markdown(f"**Source:** {doc.metadata['source']}")
                            st.markdown(f"**Content:** {doc.page_content}")
                            st.markdown("---")
                
            except Exception as e:
                if IS_DEV:
                    st.error(f"An error occurred: {str(e)}")
                else:
                    st.error("Sorry, something went wrong. Please try again later.")

if __name__ == "__main__":
    setup_directories()  # Call this first
    main() 