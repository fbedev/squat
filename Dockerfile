# ---- Base ----
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Runtime deps for PyAV (ffmpeg). Headless OpenCV doesn't need libGL.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- App ----
COPY . .
EXPOSE 8080

# Use backslashes for line continuation in CMD
CMD ["streamlit", "run", "main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8080", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false", \
     "--client.showErrorDetails=false", \
     "--client.toolbarMode=minimal"]
