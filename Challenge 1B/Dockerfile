# Use an official Python runtime as a parent image.
FROM python:3.12-slim-bullseye

# Set environment variables for better Docker build behavior and model caching.
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/huggingface_cache
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container.
WORKDIR /app

# Install system-level dependencies.
# - libglib2.0-0: General library for PyMuPDF.
# - tesseract-ocr: The OCR engine for pytesseract.
# - poppler-utils: PDF rendering utilities for pdf2image.
# - libpoppler-glib-dev: Sometimes needed for pdf2image compilation if wheels aren't found.
# - libleptonica-dev: Dependency for tesseract-ocr.
# - imagemagick: Used by pdf2image on some systems.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    libleptonica-dev \
    libpoppler-glib-dev \
    imagemagick \
    && rm -rf /var/lib/apt/lists/* # Clean up apt lists to reduce image size

# Copy the requirements file into the container.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# CRUCIAL STEP for offline functionality:
# Download the Sentence-Transformer model into the container's cache during the build process.
# This ensures the model is present inside the image when the container runs,
# eliminating the need for internet access at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Copy your teammate's application code and model files into the container.
# This assumes your 'adobe_repo' folder (with the underscore) is at the root of your project on the host.
COPY adobe_repo /app/adobe_repo/ 

# Copy the rest of your application code (pdf_processor.py) into the container.
COPY . .

# Define the default command to run when the container starts.
CMD ["python", "pdf_processor.py", "default_collection_name"]