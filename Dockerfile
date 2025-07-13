# Dockerfile for RAG Gemini Chatbot
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies (if needed for pypdf, chromadb, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default environment (can be overridden)
ENV PYTHONUNBUFFERED=1

# Entrypoint to run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0"] 