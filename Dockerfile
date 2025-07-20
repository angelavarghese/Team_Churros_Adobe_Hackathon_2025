# Use a lightweight Python image compatible with AMD64 based on Debian Bookworm (Debian 12)
# Bookworm is the current stable release, ensuring active package repositories.
# The --platform flag here specifies the target architecture for the base image.
FROM --platform=linux/amd64 python:3.9-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by pdfplumber (e.g., for Pillow)
# These are common dependencies for image processing and font handling.
# 'poppler-utils' for PDF manipulation (like pdftotext)
# 'libpoppler-dev' for development headers needed by some Python libraries interacting with Poppler
# 'build-essential' for compiling C/C++ extensions during pip installs (e.g., if any Python package
#   has native dependencies, though slim images try to minimize this)
# 'python3-dev' for Python development headers, often needed when pip installing packages with C extensions
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libpoppler-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install Python dependencies specified in requirements.txt
# '--no-cache-dir' helps keep the image size smaller by not storing pip's cache.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application script into the working directory
COPY main.py .

# Command to run the application when the container starts
# This script is expected to automatically process PDFs in the /app/input directory (if applicable to your script)
CMD ["python", "main.py"]