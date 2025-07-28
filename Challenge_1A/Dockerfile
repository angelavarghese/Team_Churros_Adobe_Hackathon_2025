FROM python:3.11-slim

# Install system dependencies for OCR and PDF processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-kan \
    tesseract-ocr-san \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    tesseract-ocr-tam \
    tesseract-ocr-mar \
    tesseract-ocr-urd \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create required directories
RUN mkdir -p /app/input /app/output

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command for hackathon compliance
CMD ["python", "robust_pdf_extractor.py"]
