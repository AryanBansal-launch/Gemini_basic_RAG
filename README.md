# RAG Gemini Chatbot

A Streamlit-powered Retrieval-Augmented Generation (RAG) chatbot that answers questions using content from your uploaded PDF files. It leverages Google Gemini for embeddings and answer generation, and ChromaDB for efficient document retrieval.

## Features
- **Dynamic PDF Upload:** Upload multiple PDFs at any time; the bot instantly incorporates their content.
- **Source Tracking:** Answers include the source PDF filenames.
- **Modern UI:** Chat interface powered by Streamlit.
- **Persistent Knowledge Base:** Uses ChromaDB for fast, persistent retrieval.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd RAG_GEMINI
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
- Copy the sample `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```
- You will need a Google Gemini API key. [Get one here.](https://aistudio.google.com/app/apikey)

### 5. Run the App
```bash
streamlit run main.py
```
- The app will open in your browser. Upload PDFs and start chatting!

## Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `CHROMA_PATH`: Path to store ChromaDB data (e.g., `chroma/chroma.sqlite3`)
- `CHROMA_COLLECTION`: Name for your ChromaDB collection (e.g., `rag_collection`)
- `PDF_PATH`: (Optional) Path to a PDF for CLI ingestion mode
- `GEMINI_GENERATION_MODEL`: (Optional) Gemini model for answer generation (default: `gemini-pro`)

See `.env.example` for a template.

## Notes
- Uploaded PDFs are processed and stored in the ChromaDB collection for fast retrieval.
- You can upload new PDFs at any time; the bot will use all available sources.
- Source filenames are shown with answers for transparency.

## CLI Usage
You can also ingest a PDF via CLI (for batch mode):
```bash
python main.py  # Uses PDF_PATH from .env
```

---

## License
MIT 