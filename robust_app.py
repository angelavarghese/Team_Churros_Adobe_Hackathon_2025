#!/usr/bin/env python3
"""
Enhanced Flask Web Interface for Robust Multilingual PDF Heading Extractor
Supports both text-based and scanned PDFs with OCR fallback
"""

import os
import json
import time
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pandas as pd

# Import the robust PDF extractor
from robust_pdf_extractor import process_pdf, OCR_AVAILABLE

# Flask app configuration
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    """Home page with upload interface."""
    return render_template("robust_index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle PDF upload and processing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the PDF
            start_time = time.time()
            result = process_pdf(file_path)
            
            if result:
                # Add processing time
                result['processing_info']['web_processing_time'] = time.time() - start_time
                
                # Clean up uploaded file
                os.remove(file_path)
                
                return jsonify(result)
            else:
                # Clean up uploaded file
                os.remove(file_path)
                return jsonify({"error": "Failed to process PDF"}), 500
                
        except Exception as e:
            # Clean up uploaded file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    
    return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400

@app.route("/status")
def status():
    """Return system status and capabilities."""
    return jsonify({
        "status": "operational",
        "ocr_available": OCR_AVAILABLE,
        "supported_languages": "eng+hin+kan+san+jpn+kor+tam+mar+urd",
        "max_file_size": "50MB",
        "supported_formats": ["PDF"],
        "features": [
            "Text-based PDF processing",
            "Scanned PDF OCR processing",
            "Multilingual support (10+ languages)",
            "Heading detection (H1, H2, H3, Title)",
            "JSON output with UTF-8 encoding"
        ]
    })

@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": time.time()})

if __name__ == "__main__":
    print("🌍 Robust Multilingual PDF Heading Extractor - Web Interface")
    print("=" * 60)
    print(f"✅ OCR Available: {OCR_AVAILABLE}")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🌐 Starting web server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 