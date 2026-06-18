FROM python:3.11-slim

WORKDIR /app

# System dependencies for PyMuPDF and spaCy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_md

# Copy application code
COPY app.py .

EXPOSE 8080

CMD ["python", "app.py"]