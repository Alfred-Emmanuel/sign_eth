# Document Q&A Bot

A beginner-friendly web application that allows users to ask questions about their documents using Google's Gemini AI and the RAG (Retrieval-Augmented Generation) pattern.

## Features

- Load and process text documents from a specified directory
- Create and maintain a vector store using ChromaDB
- Ask questions about your documents and get AI-generated answers
- View source documents for each answer
- Simple and intuitive Streamlit interface
- Automatic document reloading capability

## Prerequisites

- Python 3.9 or higher
- A Google API key for Gemini AI

## Setup Instructions

1. **Clone or download this repository**

2. **Create and activate a virtual environment**

   On Windows:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   On macOS/Linux:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Google API key**

   - Create a `.env` file in the root directory
   - Add your Google API key to the file:
     ```
     GOOGLE_API_KEY='your-api-key-here'
     ```
   - You can get a Google API key from: https://makersuite.google.com/app/apikey

5. **Prepare your documents**
   - Create a `docs` folder in the root directory (if it doesn't exist)
   - Place your `.txt` files in the `docs` folder
   - The application will automatically process these files on first run

## Running the Application

1. **Start the Streamlit app**

   ```bash
   streamlit run app.py
   ```

2. **Access the application**

   - Open your web browser
   - Navigate to the URL shown in the terminal (typically http://localhost:8501)

3. **Using the application**
   - The application will automatically process documents in the `docs` folder on first run
   - Use the "Reload Documents" button to reprocess documents if you add new ones
   - Enter your question in the text input box
   - Click "Get Answer" to receive an AI-generated response
   - Expand "View Source Documents" to see the relevant document chunks used for the answer

## Directory Structure

```
.
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── docs/              # Place your .txt files here
└── vector_store/      # ChromaDB vector store (created automatically)
```

## Notes

- The first run might take longer as it processes your documents
- The application uses ChromaDB for vector storage, which is created in the `vector_store` directory
- You can add new documents at any time and use the "Reload Documents" button to process them
- The application currently supports `.txt` files only
