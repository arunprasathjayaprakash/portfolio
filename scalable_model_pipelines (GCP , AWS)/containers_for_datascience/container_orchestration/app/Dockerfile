# Use a stable base image with Python 3
FROM python:3.11-slim

# Set the working directory
WORKDIR /scripts

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libatlas-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application scripts to the container
COPY /scripts ./scripts

# Expose the application port
EXPOSE 8080

# Set the entrypoint to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "scripts/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
