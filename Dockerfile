FROM python:3.12-slim

WORKDIR /app

# Install system deps for OpenCV (needed by some transforms)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (weights are mounted as a volume)
COPY depth/ depth/
COPY models/ models/
COPY server.py .

EXPOSE 8081

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8081"]
