# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy environment file
COPY .env .env

# Copy GCP service account key
COPY GCP_Account_Key.json ./GCP_Account_Key.json

# Set GCP credentials environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/GCP_Account_Key.json

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "-m", "app.main"]