FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create data and results directories
RUN mkdir -p /app/data /app/results

# Copy application code
COPY . .

# Set permissions
RUN chmod -R 755 /app

# Create a non-root user
RUN useradd -m tsagent
USER tsagent

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-c", "print('Container ready')"]