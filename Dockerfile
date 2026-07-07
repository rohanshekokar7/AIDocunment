FROM python:3.10-slim

# Install system dependencies, including poppler-utils for pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create uploads directory
RUN mkdir -p uploads

# Copy application code
COPY ./app ./app
COPY ./tests ./tests
COPY ./static ./static

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
