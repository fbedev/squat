FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# PyAV runtime (ffmpeg). Headless OpenCV doesn't need libGL.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN chmod +x /app/start.sh
EXPOSE 8080

# Run via script that unsets the rogue env vars
CMD ["/app/start.sh"]
