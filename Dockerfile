# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip

# Install PyTorch CPU-only version to reduce size
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies with verbose output
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Copy project files
COPY . .

# Create data directory
RUN mkdir -p data

# Clean up pip cache
RUN pip cache purge

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"] 