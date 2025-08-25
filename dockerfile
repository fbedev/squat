FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Runtime deps: ffmpeg for PyAV, (no libGL since we use headless OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
CMD ["streamlit", "run", "main.py",
     "--server.address", "0.0.0.0",
     "--server.port", "8080",
     "--server.fileWatcherType", "none",
     "--browser.gatherUsageStats", "false",
     "--client.showErrorDetails", "false",
     "--client.toolbarMode", "minimal"]
