FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application files
COPY . .

# Create .streamlit directory and config
RUN mkdir -p .streamlit
COPY .streamlit/config.toml .streamlit/

# Set environment variables for large uploads and stability
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024
ENV STREAMLIT_SERVER_MAX_MESSAGE_SIZE=1024
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_SERVER_RUN_ON_SAVE=false

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Run Streamlit with optimized settings for large TIFF files
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.maxUploadSize=1024", \
    "--server.maxMessageSize=1024", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false", \
    "--server.fileWatcherType=none", \
    "--server.runOnSave=false"]
