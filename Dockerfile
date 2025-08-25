# ---- Base ----
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
EXPOSE 8080

# Shell form so $PORT expands; default to 8080 if not provided
CMD bash -lc 'streamlit run main.py \
  --server.address=0.0.0.0 \
  --server.port=${PORT:-8080} \
  --server.fileWatcherType=none \
  --browser.gatherUsageStats=false \
  --client.showErrorDetails=false \
  --client.toolbarMode=minimal'

