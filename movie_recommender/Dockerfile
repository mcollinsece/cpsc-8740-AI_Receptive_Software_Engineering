# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=recommender_service.py \
    FLASK_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p models

# Copy application code and model files
COPY recommender_service.py .
COPY models/ models/
COPY recommender_model.pth .
COPY model_mappings.json .

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "recommender_service:app"] 