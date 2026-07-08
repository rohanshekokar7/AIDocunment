FROM python:3.10-slim

# Install system dependencies, including poppler-utils for pdf2image and libGL for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# Define environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    FLAGS_use_mkldnn=0 \
    FLAGS_enable_pir_api=0

WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create uploads directory
RUN mkdir -p uploads

# Copy application code
COPY --chown=user ./app ./app
COPY --chown=user ./tests ./tests
COPY --chown=user ./static ./static

# Hugging Face routes traffic to port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
