import os
import re
from typing import List
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
import google.generativeai as genai
import streamlit as st

st.write(':green[Streamlit app loaded successfully!]')

try:
    # Load environment variables from .env
    load_dotenv()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    CHROMA_PATH = os.getenv("CHROMA_PATH")
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION")
    PDF_PATH = os.getenv("PDF_PATH")  # Only used for CLI mode

    if not all([GEMINI_API_KEY, CHROMA_PATH, CHROMA_COLLECTION]):
        raise ValueError("One or more environment variables are missing. Please check your .env file.")

    # Step 1: Load PDF(s)
    def load_pdf_from_filelike(filelike):
        reader = PdfReader(filelike)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def split_text(text: str):
        split_text = re.split(r'\n\s*\n', text)
        return [i for i in split_text if i.strip() != ""]

    from chromadb import Documents, EmbeddingFunction, Embeddings

    def print_available_models():
        genai.configure(api_key=GEMINI_API_KEY)
        print("Listing available Gemini models for your API key:")
        for model in genai.list_models():
            print(f"Name: {model.name}, Supported: {model.supported_generation_methods}")

    class GeminiEmbeddingFunction(EmbeddingFunction):
        def __init__(self):
            pass
        def __call__(self, input: Documents) -> Embeddings:
            genai.configure(api_key=GEMINI_API_KEY)
            model = "models/text-embedding-004"
            title = "Custom query"
            return genai.embed_content(model=model,
                                       content=input,
                                       task_type="retrieval_document",
                                       title=title)["embedding"]

    def create_or_load_chroma_collection(path, name):
        chroma_client = chromadb.PersistentClient(path=path)
        try:
            db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        except Exception:
            db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        return db

    def add_chunks_to_chroma(db, chunks, file_name, start_id=0):
        # Get current count to avoid ID collisions
        current_count = len(db.get()["ids"]) if db.count() > 0 else 0
        for i, chunk in enumerate(chunks):
            db.add(documents=[chunk], ids=[str(current_count + i)], metadatas=[{"source": file_name}])

    def get_relevant_passage(query, db, n_results):
        result = db.query(query_texts=[query], n_results=n_results)
        return result['documents'][0], result.get('metadatas', [[]])[0]

    def make_rag_prompt(query, relevant_passage):
        escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
        prompt = (f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
strike a friendly and conversational tone. \
If the passage is irrelevant to the answer, you may ignore it.\nQUESTION: '{query}'\nPASSAGE: '{escaped}'\n\nANSWER:\n""")
        return prompt

    def generate_gemini_answer(prompt):
        genai.configure(api_key=GEMINI_API_KEY)
        model_name = os.getenv("GEMINI_GENERATION_MODEL", "gemini-pro")
        model = genai.GenerativeModel(model_name)
        answer = model.generate_content(prompt)
        return answer.text

    def answer_query(db, query):
        relevant_texts, metadatas = get_relevant_passage(query, db, n_results=3)
        prompt = make_rag_prompt(query, relevant_passage=" ".join(relevant_texts))
        answer = generate_gemini_answer(prompt)
        # Optionally, show sources
        sources = set(md.get("source", "") for md in metadatas if md)
        if sources:
            answer += f"\n\n:sparkles: **Sources:** {', '.join(sources)}"
        return answer

    def interactive_chat():
        st.set_page_config(page_title="RAG Gemini Chatbot", page_icon="🤖")
        st.title("RAG Gemini Chatbot")
        st.write("Ask questions about your uploaded PDFs!")
        db = create_or_load_chroma_collection(path=CHROMA_PATH, name=CHROMA_COLLECTION)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "files_added" not in st.session_state:
            st.session_state.files_added = set()
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.files_added:
                    text = load_pdf_from_filelike(file)
                    chunks = split_text(text)
                    add_chunks_to_chroma(db, chunks, file.name)
                    st.session_state.files_added.add(file.name)
                    st.success(f"Added {file.name} to knowledge base!")
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                st.markdown(a)
        user_input = st.chat_input("Type your question and press Enter...")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            answer = answer_query(db, user_input)
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat_history.append((user_input, answer))

    if __name__ == "__main__":
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--list-models":
            print_available_models()
            exit(0)
        # CLI mode: fallback to PDF_PATH for batch ingestion
        if PDF_PATH:
            pdf_text = load_pdf_from_filelike(open(PDF_PATH, "rb"))
            chunked_text = split_text(text=pdf_text)
            db = create_or_load_chroma_collection(path=CHROMA_PATH, name=CHROMA_COLLECTION)
            add_chunks_to_chroma(db, chunked_text, os.path.basename(PDF_PATH))
        interactive_chat()
except Exception as e:
    st.error(f"An error occurred: {e}")
    import traceback
    st.text(traceback.format_exc())