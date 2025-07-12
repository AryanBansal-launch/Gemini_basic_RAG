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
    PDF_PATH = os.getenv("PDF_PATH")
    CHROMA_PATH = os.getenv("CHROMA_PATH")
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION")

    if not all([GEMINI_API_KEY, PDF_PATH, CHROMA_PATH, CHROMA_COLLECTION]):
        raise ValueError("One or more environment variables are missing. Please check your .env file.")

    # Step 1: Load PDF

    def load_pdf(file_path):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    pdf_text = load_pdf(file_path=PDF_PATH)

    # Step 2: Split text into chunks

    def split_text(text: str):
        split_text = re.split(r'\n\s*\n', text)
        return [i for i in split_text if i.strip() != ""]

    chunked_text = split_text(text=pdf_text)

    # Step 3: Embedding function
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

    # Step 4: Create Chroma DB

    def create_chroma_db(documents: List[str], path: str, name: str):
        chroma_client = chromadb.PersistentClient(path=path)
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        for i, d in enumerate(documents):
            db.add(documents=[d], ids=[str(i)])
        return db

    # Step 5: Load Chroma Collection

    def load_chroma_collection(path, name):
        chroma_client = chromadb.PersistentClient(path=path)
        db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        return db

    # Step 6: Get relevant passage

    def get_relevant_passage(query, db, n_results):
        result = db.query(query_texts=[query], n_results=n_results)
        return result['documents'][0]

    # Step 7: Make RAG prompt

    def make_rag_prompt(query, relevant_passage):
        escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
        prompt = (f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
strike a friendly and conversational tone. \
If the passage is irrelevant to the answer, you may ignore it.\nQUESTION: '{query}'\nPASSAGE: '{escaped}'\n\nANSWER:\n""")
        return prompt

    # Step 8: Generate answer

    def generate_gemini_answer(prompt):
        genai.configure(api_key=GEMINI_API_KEY)
        # Use model name from env or fallback to 'gemini-pro'
        model_name = os.getenv("GEMINI_GENERATION_MODEL", "gemini-pro")
        model = genai.GenerativeModel(model_name)
        answer = model.generate_content(prompt)
        return answer.text

    def answer_query(db, query):
        relevant_text = get_relevant_passage(query, db, n_results=3)
        prompt = make_rag_prompt(query, relevant_passage=" ".join(relevant_text))
        answer = generate_gemini_answer(prompt)
        return answer

    def interactive_chat():
        st.set_page_config(page_title="RAG Gemini Chatbot", page_icon="🤖")
        st.title("RAG Gemini Chatbot")
        st.write("Ask questions about your PDF!")
        db = load_chroma_collection(path=CHROMA_PATH, name=CHROMA_COLLECTION)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
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
        interactive_chat()
except Exception as e:
    st.error(f"An error occurred: {e}")
    import traceback
    st.text(traceback.format_exc())