FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data needed for preprocessing
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"

# Copy application code
COPY flask_app/ /app/
COPY .env /app/.env

EXPOSE 5001

CMD ["python", "app.py"]
