# Enhanced Multilingual PDF Heading Extractor with OCR
# Supports 10+ languages: English, Hindi, Kannada, Sanskrit, Japanese, Korean, Tamil, Marathi, Urdu, etc.
# Constraints: ≤200MB, ≤10s execution, offline, amd64, OCR fallback

FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF and Tesseract OCR
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Tesseract language packs for multilingual OCR
# Only install required languages to stay within size constraints
RUN apt-get update && apt-get install -y \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-kan \
    tesseract-ocr-san \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    tesseract-ocr-tam \
    tesseract-ocr-mar \
    tesseract-ocr-urd \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/output /app/model /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

# Expose port for web interface
EXPOSE 5000

# Default command - can be overridden
CMD ["python", "predict_and_export.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import joblib; joblib.load('model/model.joblib')" || exit 1
