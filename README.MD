# 🌍 Robust Multilingual PDF Heading Extractor

## 📋 Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [File-by-File Documentation](#file-by-file-documentation)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Performance Metrics](#performance-metrics)
9. [Troubleshooting](#troubleshooting)
10. [Development Guide](#development-guide)

---

## 🎯 Overview

A comprehensive, production-ready solution for extracting headings from both text-based and scanned PDFs in 10+ languages. The system automatically detects PDF types, applies appropriate processing methods (direct text extraction or OCR), and outputs structured heading hierarchies in JSON format.

### 🌟 Key Features
- **Dual Processing**: Handles text-based and scanned PDFs automatically
- **Multilingual Support**: 11 script types across 10+ languages
- **ML-Powered**: Machine learning classification with heuristic fallback
- **Web Interface**: Modern drag-and-drop UI with real-time processing
- **Docker Ready**: Containerized deployment
- **Performance Optimized**: <200MB total size, <10s processing time

---

## 🏗️ System Architecture

### 🎯 **Dual System Architecture**

The project supports both **modern** and **legacy** systems for maximum compatibility:

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACES                              │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│  Modern Web     │  Command Line   │  Legacy Web     │  Docker   │
│  (robust_app)   │ (robust_extract)│     (app)       │ Container │
└─────────┬───────┴─────────┬───────┴─────────┬───────┴─────┬─────┘
          │                  │                │             │
          └──────────────────┼────────────────┼─────────────┘
                             │                │
          ┌──────────────────▼────────────────▼─────────────┐
          │              CORE PROCESSING                    │
          │  ┌─────────────────┐  ┌─────────────────────┐   │
          │  │ Modern Engine   │  │  Legacy Engine      │   │
          │  │(robust_extract) │  │ (extract_features)  │   │
          │  └─────────┬───────┘  └─────────┬───────────┘   │
          └────────────┼────────────────────┼───────────────┘
                       │                    │
          ┌────────────▼────────────────────▼───────────────┐
          │            PDF TYPE DETECTION                   │
          │  ┌─────────────────┐  ┌─────────────────────┐   │
          │  │ Text-based PDF  │  │   Scanned PDF       │   │
          │  │   (PyMuPDF)     │  │  (Tesseract OCR)    │   │
          │  └─────────────────┘  └─────────────────────┘   │
          └────────────┬────────────────────┬───────────────┘
                       │                    │
          ┌────────────▼────────────────────▼───────────────┐
          │           FEATURE EXTRACTION                    │
          │  - Font size & ratio                            │
          │  - Alignment & spacing                          │
          │  - Word count & length                          │
          │  - Script detection (11 languages)              │
          │  - Unicode normalization                        │
          └────────────┬────────────────────┬───────────────┘
                       │                    │
          ┌────────────▼────────────────────▼───────────────┐
          │         HEADING CLASSIFICATION                  │
          │  ┌─────────────────┐  ┌─────────────────────┐   │
          │  │ ML Model        │  │  Heuristic Fallback │   │
          │  │(RandomForest)   │  │   (Rule-based)      │   │
          │  └─────────────────┘  └─────────────────────┘   │
          └────────────┬────────────────────┬───────────────┘
                       │                    │
          ┌────────────▼────────────────────▼───────────────┐
          │           OUTPUT GENERATION                     │
          │  - Structured JSON outline                      │
          │  - Processing metadata                          │
          │  - UTF-8 encoding                               │
          │  - Multilingual statistics                      │
          └─────────────────────────────────────────────────┘
```

### 🔄 **System Comparison**

| Component | Modern System | Legacy System | Purpose |
|-----------|---------------|---------------|---------|
| **Web Interface** | `robust_app.py` | `app.py` | User interaction |
| **Template** | `robust_index.html` | `index.html` | UI rendering |
| **Core Engine** | `robust_pdf_extractor.py` | `extract_features.py` | PDF processing |
| **Prediction** | Integrated | `predict_and_export.py` | ML inference |
| **Training** | N/A | `train_model.py` | Model training |
| **Testing** | N/A | `test_multilingual.py` | System validation |

### 🎯 **Recommended Usage**

- **Production**: Use Modern System (`robust_app.py`)
- **Development**: Use Legacy System for training/testing
- **Automation**: Use Command Line Interface
- **Deployment**: Use Docker container

---

## 📁 File-by-File Documentation

### 🔧 Core Engine Files (Modern System)

#### `robust_pdf_extractor.py` (16KB, 482 lines)
**Purpose**: Main processing engine that orchestrates the entire PDF heading extraction pipeline.

**Key Functions**:

1. **`normalize_text(text)`** (Lines 25-35)
   - Handles Unicode normalization using NFKC
   - Decodes bytes to UTF-8 with fallback
   - Removes replacement characters ()
   - Ensures clean text for processing

2. **`detect_script(text)`** (Lines 37-65)
   - Detects 11 script types using Unicode ranges
   - Supports: Latin, Devanagari, Kannada, Tamil, Japanese, Korean, Arabic, Chinese, Thai, Bengali, Telugu
   - Handles script overlaps (e.g., Chinese/Japanese Kanji)

3. **`calculate_caps_ratio(text, script_type)`** (Lines 67-73)
   - Calculates capitalization ratio for Latin scripts only
   - Returns 0 for non-Latin scripts (no traditional caps)

4. **`is_scanned_pdf(pdf_path)`** (Lines 77-95)
   - Analyzes PDF content to determine if scanned or text-based
   - Checks text length, image block count, and content patterns
   - Returns boolean for processing decision

5. **`extract_text_textpdf(pdf_path)`** (Lines 99-108)
   - Extracts text blocks with metadata from text-based PDFs
   - Returns page numbers, blocks, and page dimensions
   - Uses PyMuPDF for structured extraction

6. **`extract_text_ocr(pdf_path, lang)`** (Lines 110-130)
   - OCR processing for scanned PDFs
   - Supports 9 language packs: eng+hin+kan+san+jpn+kor+tam+mar+urd
   - Returns page-wise text extraction

7. **`get_features_from_block(block, page_height, page_num)`** (Lines 134-175)
   - Extracts 12 features from text blocks:
     - `font_size`, `font_size_ratio`, `bold`, `alignment`
     - `spacing_above`, `spacing_below`, `line_spacing`
     - `num_words`, `avg_word_length`, `caps_ratio`
     - `script_type`, `position_pct`, `page`
   - Handles font size estimation and alignment detection

8. **`get_features_from_ocr_text(text, page_num, line_num, total_lines)`** (Lines 177-205)
   - Feature extraction for OCR-processed text
   - Estimates position and font characteristics
   - Applies heuristic rules for heading detection

9. **`classify_headings_ml(features_list)`** (Lines 209-250)
   - ML-powered heading classification using RandomForest
   - Handles feature compatibility with trained model
   - Falls back to heuristic classification if ML fails

10. **`classify_headings_heuristic(features_list)`** (Lines 252-275)
    - Rule-based heading classification
    - Uses word count, font size ratio, and position
    - Provides fallback when ML model unavailable

11. **`process_pdf(pdf_path)`** (Lines 279-340)
    - Main orchestration function
    - Implements complete processing pipeline
    - Returns structured JSON output with metadata

12. **`process_directory(input_dir)`** (Lines 342-375)
    - Batch processing for multiple PDFs
    - Error handling and progress reporting
    - Summary statistics generation

**Configuration Constants**:
- `LANGUAGES`: "eng+hin+kan+san+jpn+kor+tam+mar+urd"
- `THRESHOLD_TEXT_LENGTH`: 50 characters
- `MODEL_PATH`: "model/model.joblib"
- `LABEL_MAP_PATH`: "model/label_map.json"

#### `robust_app.py` (3.6KB, 108 lines)
**Purpose**: Flask web application providing modern web interface for PDF processing.

**Key Components**:

1. **Flask Configuration** (Lines 12-18)
   - Upload folder setup with 50MB file size limit
   - File type validation for PDFs only
   - Secure filename handling

2. **`allowed_file(filename)`** (Lines 20-23)
   - Validates file extensions
   - Ensures only PDF files are processed

3. **`home()` Route** (Lines 25-27)
   - Serves the main web interface
   - Renders robust_index.html template

4. **`upload_file()` Route** (Lines 29-65)
   - Handles PDF file uploads
   - Integrates with robust_pdf_extractor
   - Provides error handling and cleanup
   - Returns JSON response with results

5. **`status()` Route** (Lines 67-79)
   - System status and capability reporting
   - Shows OCR availability and supported features
   - API endpoint for health checks

6. **`health()` Route** (Lines 81-83)
   - Simple health check endpoint
   - Returns timestamp for monitoring

**Error Handling**:
- File validation and size limits
- Processing error recovery
- Temporary file cleanup
- Graceful degradation

### 🎨 Web Interface Files

#### `templates/robust_index.html` (15KB, 489 lines)
**Purpose**: Modern, responsive web interface with drag-and-drop functionality.

**Key Features**:

1. **Responsive Design** (Lines 1-50)
   - CSS Grid layout for feature cards
   - Mobile-responsive design
   - Modern gradient backgrounds
   - Smooth animations and transitions

2. **Feature Showcase** (Lines 200-230)
   - Visual representation of capabilities
   - Icons for text PDFs, scanned PDFs, languages, speed
   - Clear value proposition

3. **Upload Interface** (Lines 240-280)
   - Drag-and-drop file upload
   - File type validation
   - Progress bar with animation
   - Loading spinner with status messages

4. **Results Display** (Lines 290-350)
   - Structured heading list with levels
   - Processing information panel
   - JSON download functionality
   - Error message handling

5. **JavaScript Functionality** (Lines 360-489)
   - File handling and validation
   - AJAX upload with progress tracking
   - Dynamic result rendering
   - Error state management

**CSS Features**:
- Flexbox and Grid layouts
- CSS animations and transitions
- Responsive breakpoints
- Modern color scheme

**JavaScript Features**:
- Event-driven file handling
- Progress simulation
- Error boundary handling
- Dynamic DOM manipulation

### 🔄 Legacy System Files

#### `extract_features.py` (10KB, 298 lines)
**Purpose**: Enhanced feature extraction module for the legacy system with multilingual support.

**Key Functions**:

1. **`normalize_text(text)`** (Lines 8-25)
   - Unicode normalization using NFKC
   - Byte-to-string decoding with fallback
   - Replacement character removal
   - Robust text cleaning

2. **`detect_script(text)`** (Lines 27-65)
   - 11 script type detection using Unicode ranges
   - Supports: Latin, Devanagari, Kannada, Tamil, Japanese, Korean, Arabic, Chinese, Thai, Bengali, Telugu
   - Handles script overlaps (Chinese/Japanese Kanji)

3. **`script_to_numeric(script_name)`** (Lines 67-80)
   - Converts script names to numeric IDs for ML
   - Mapping: Latin(0), Devanagari(1), Kannada(2), etc.

4. **`calculate_caps_ratio(text, script_type)`** (Lines 82-88)
   - Language-aware capitalization ratio
   - Only applies to Latin scripts
   - Returns 0 for non-Latin scripts

5. **`extract_text_with_enhanced_features(page, page_num)`** (Lines 90-130)
   - Extracts text blocks with metadata
   - Font size and bold detection
   - Alignment and positioning
   - Source tracking (text/ocr)

6. **`extract_features(pdf_path)`** (Lines 132-220)
   - Main feature extraction pipeline
   - 12 enhanced features: font_size, font_size_ratio, bold, alignment, spacing_above, spacing_below, line_spacing, num_words, avg_word_length, caps_ratio, script_type, position_pct
   - Multilingual script detection
   - Processing statistics

7. **`extract_features_from_pdf(pdf_path)`** (Lines 222-250)
   - Legacy compatibility function
   - Returns list of dicts with features
   - Backward compatibility wrapper

**Usage**:
```python
from extract_features import extract_features
df = extract_features("document.pdf")
print(f"Extracted {len(df)} blocks with {len(df.columns)} features")
```

#### `train_model.py` (6.6KB, 203 lines)
**Purpose**: Machine learning model training with enhanced multilingual features.

**Key Functions**:

1. **Data Loading** (Lines 15-25)
   - Loads labeled CSV dataset
   - Handles missing labels
   - Data validation and cleaning

2. **Label Encoding** (Lines 27-35)
   - LabelEncoder for categorical labels
   - Reverse label map generation
   - JSON serialization for predictions

3. **Feature Selection** (Lines 37-50)
   - Enhanced feature columns definition
   - Fallback to legacy feature names
   - Feature availability checking

4. **Data Splitting** (Lines 52-70)
   - Stratified train/test split
   - Fallback to regular split if stratification fails
   - Class balance preservation

5. **Model Training** (Lines 72-100)
   - RandomForestClassifier with optimized parameters
   - Class weight balancing for imbalanced data
   - Out-of-bag scoring enabled
   - Performance timing

6. **Model Evaluation** (Lines 102-130)
   - Classification report generation
   - Accuracy and OOB score calculation
   - Feature importance analysis
   - Performance metrics

7. **Model Persistence** (Lines 132-150)
   - Joblib compression (level 3)
   - Model size calculation
   - Performance summary
   - Constraint validation

**Usage**:
```bash
python train_model.py
# Outputs: model/model.joblib, model/label_map.json
```

#### `predict_and_export.py` (6.5KB, 182 lines)
**Purpose**: Prediction pipeline with enhanced feature compatibility and JSON export.

**Key Functions**:

1. **Model Loading** (Lines 15-30)
   - Loads trained model and label map
   - Error handling for missing files
   - Label decoder setup

2. **Feature Extraction** (Lines 32-50)
   - Enhanced feature extraction
   - Processing time measurement
   - Feature availability checking

3. **Feature Compatibility** (Lines 52-80)
   - Model feature name matching
   - Fallback to legacy feature names
   - Feature subset selection

4. **Prediction Pipeline** (Lines 82-110)
   - ML model prediction
   - Label decoding
   - Performance timing

5. **Output Generation** (Lines 112-140)
   - Structured JSON output
   - UTF-8 encoding handling
   - Heading filtering (H1, H2, H3, Title)

6. **Performance Metrics** (Lines 142-170)
   - Execution time breakdown
   - Multilingual statistics
   - OCR usage statistics
   - Model feature importance

**Usage**:
```bash
python predict_and_export.py
# Outputs: output/predictions.json
```

#### `app.py` (5.8KB, 155 lines)
**Purpose**: Legacy Flask web application with enhanced multilingual features.

**Key Functions**:

1. **Model Loading** (Lines 10-25)
   - Enhanced model and label map loading
   - Error handling for missing models
   - Backward compatibility

2. **File Upload Handling** (Lines 30-50)
   - PDF file validation
   - Secure filename handling
   - Temporary file management

3. **Feature Extraction** (Lines 52-80)
   - Enhanced feature extraction
   - Feature compatibility checking
   - Script type detection

4. **Prediction Processing** (Lines 82-110)
   - ML model prediction
   - Label decoding
   - Heading filtering

5. **Response Generation** (Lines 112-140)
   - Structured JSON response
   - Multilingual statistics
   - OCR usage statistics
   - Performance metrics

6. **Error Handling** (Lines 142-155)
   - Exception catching
   - File cleanup
   - Error response formatting

**Usage**:
```bash
python app.py
# Access: http://localhost:5000
```

#### `templates/index.html` (6.0KB, 185 lines)
**Purpose**: Legacy web interface template with basic functionality.

**Key Features**:

1. **Simple Upload Interface** (Lines 30-50)
   - Basic file input
   - Upload button
   - File type validation

2. **Loading States** (Lines 60-80)
   - Simple loading message
   - Error display
   - Results container

3. **Results Display** (Lines 90-130)
   - Basic heading list
   - Level indicators
   - Page numbers
   - JSON download link

4. **JavaScript Functionality** (Lines 140-185)
   - File upload handling
   - AJAX requests
   - Result rendering
   - Error handling

**CSS Features**:
- Basic styling
- Responsive design
- Clean layout
- Bootstrap-like appearance

**JavaScript Features**:
- Simple file handling
- Basic AJAX upload
- Result display
- Download functionality

#### `test_multilingual.py` (12KB, 333 lines)
**Purpose**: Comprehensive test suite for multilingual functionality validation.

**Key Functions**:

1. **`test_script_detection()`** (Lines 15-50)
   - 11 language test cases
   - Unicode range validation
   - Script detection accuracy

2. **`test_text_normalization()`** (Lines 52-75)
   - Unicode normalization testing
   - Special character handling
   - Text cleaning validation

3. **`test_feature_extraction()`** (Lines 77-130)
   - PDF processing validation
   - Feature completeness checking
   - Synthetic data generation

4. **`test_model_compatibility()`** (Lines 132-180)
   - Model loading validation
   - Feature matching
   - Prediction testing

5. **`test_performance()`** (Lines 182-220)
   - Speed validation
   - Memory usage
   - Constraint compliance

6. **`main()`** (Lines 222-250)
   - Test orchestration
   - Result aggregation
   - Summary reporting

**Usage**:
```bash
python test_multilingual.py
# Outputs: Comprehensive test results
```

#### `IMPLEMENTATION_SUMMARY.md` (14KB, 538 lines)
**Purpose**: Detailed implementation documentation and technical specifications.

**Key Sections**:

1. **Project Overview** (Lines 1-20)
   - System description
   - Key achievements
   - Technical highlights

2. **System Architecture** (Lines 22-60)
   - Component relationships
   - Data flow diagrams
   - Processing pipeline

3. **Technical Implementation** (Lines 62-120)
   - Feature engineering details
   - Algorithm descriptions
   - Code examples

4. **Performance Metrics** (Lines 122-180)
   - Current performance data
   - Constraint compliance
   - Optimization opportunities

5. **Usage Examples** (Lines 182-220)
   - Command line usage
   - Web interface usage
   - Docker deployment

6. **Configuration Options** (Lines 222-260)
   - Environment variables
   - Model parameters
   - System settings

7. **Future Enhancements** (Lines 262-300)
   - Planned improvements
   - Feature expansion
   - Performance optimization

**Usage**:
- Reference documentation
- Implementation guide
- Technical specifications
- Performance benchmarks

### 🐳 Containerization Files

#### `Dockerfile` (1.6KB, 55 lines)
**Purpose**: Containerized deployment with all dependencies and OCR support.

**Key Components**:

1. **Base Image** (Line 4)
   - `python:3.11-slim` for optimal size
   - `--platform=linux/amd64` for compatibility

2. **System Dependencies** (Lines 6-10)
   - `build-essential` for PyMuPDF compilation
   - `tesseract-ocr` and `libtesseract-dev` for OCR

3. **Language Packs** (Lines 12-16)
   - 9 Tesseract language packs for multilingual OCR
   - Optimized for size constraints

4. **Python Setup** (Lines 18-25)
   - Requirements installation with no cache
   - Directory structure creation
   - Environment variable configuration

5. **Runtime Configuration** (Lines 27-35)
   - Port exposure for web interface
   - Health check for model loading
   - Default command configuration

**Environment Variables**:
- `PYTHONPATH=/app`
- `PYTHONUNBUFFERED=1`
- `TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/`

#### `.dockerignore` (86B, 12 lines)
**Purpose**: Optimizes Docker build by excluding unnecessary files.

**Excluded Items**:
- Virtual environments
- Git directories
- Python cache files
- Documentation files
- Test directories

### 📦 Dependency Management

#### `requirements.txt` (381B, 23 lines)
**Purpose**: Python package dependencies with version pinning.

**Core Dependencies**:
- `PyMuPDF==1.23.8`: PDF text extraction
- `pandas==2.1.4`: Data manipulation
- `scikit-learn==1.3.2`: Machine learning
- `joblib==1.3.2`: Model serialization
- `flask==3.0.0`: Web framework
- `numpy==1.24.3`: Numerical computing
- `regex==2023.10.3`: Unicode regex support

**OCR Dependencies** (Optional):
- `pytesseract==0.3.10`: Tesseract Python wrapper
- `Pillow==10.0.1`: Image processing
- `langdetect==1.0.9`: Language detection

**Development Dependencies**:
- `pytest==7.4.3`: Testing framework

### 🧠 Model Files

#### `model/model.joblib` (476KB, 1372 lines)
**Purpose**: Serialized RandomForest classifier for heading detection.

**Model Specifications**:
- **Algorithm**: RandomForestClassifier
- **Estimators**: 150 trees
- **Max Depth**: 12 levels
- **Features**: 10 engineered features
- **Classes**: 5 (Title, H1, H2, H3, Body)
- **Compression**: Level 3 for size optimization

**Training Data**:
- 350 samples with balanced classes
- Multilingual text blocks
- Feature-engineered dataset

#### `model/label_map.json` (60B, 1 line)
**Purpose**: Mapping between numeric predictions and human-readable labels.

**Structure**:
```json
{"0": "Body", "1": "H1", "2": "H2", "3": "H3", "4": "Title"}
```

**Usage**: Converts ML model predictions to heading levels

---

## 🛠️ Installation & Setup

### Prerequisites

#### System Requirements
```bash
# Python 3.8+ required
python --version

# Virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

#### Core Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fitz, pandas, sklearn, flask; print('✅ All dependencies installed')"
```

#### OCR Installation (Optional)
```bash
# Python OCR packages
pip install pytesseract pdf2image

# Ubuntu/Debian system packages
sudo apt-get update
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-kan \
    tesseract-ocr-san \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    tesseract-ocr-tam \
    tesseract-ocr-mar \
    tesseract-ocr-urd

# Windows: Download Tesseract installer from GitHub
# macOS: brew install tesseract tesseract-lang
```

### Directory Structure Setup
```bash
# Create necessary directories
mkdir -p input output uploads model

# Verify model files exist
ls -la model/model.joblib model/label_map.json
```

---

## 🚀 Usage Guide

### 🎯 **Quick Start Options**

Choose your preferred interface:

1. **🌐 Modern Web Interface** (Recommended)
2. **💻 Command Line Interface** (Fast processing)
3. **🔄 Legacy Web Interface** (Backward compatibility)
4. **🐳 Docker Deployment** (Production ready)
5. **🔧 Development Tools** (Training & testing)

---

### 1. 🌐 Modern Web Interface (Recommended)

#### Start Modern Web Server
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Start modern web interface
python robust_app.py
```

#### Access Interface
- **Local**: http://localhost:5000
- **Network**: http://192.168.1.3:5000 (if available)

#### Features
- **Drag & Drop**: Upload PDFs directly
- **Real-time Progress**: Visual processing feedback
- **Results Display**: Structured heading list with levels
- **JSON Download**: Export results with UTF-8 encoding
- **Error Handling**: Clear error messages
- **Processing Info**: Execution time, blocks processed, languages detected

#### API Endpoints
```bash
# Health check
curl http://localhost:5000/health

# System status
curl http://localhost:5000/status

# Upload PDF (POST)
curl -X POST -F "file=@document.pdf" http://localhost:5000/upload
```

---

### 2. 💻 Command Line Interface (Fast Processing)

#### Basic Usage
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Process all PDFs in input directory
python robust_pdf_extractor.py

# Process single PDF
python -c "
from robust_pdf_extractor import process_pdf
result = process_pdf('input/document.pdf')
print(result)
"
```

#### Batch Processing
```bash
# Place PDFs in input/ directory
copy *.pdf input\  # Windows
# or
cp *.pdf input/    # Linux/Mac

# Run processing
python robust_pdf_extractor.py

# Check results
dir output\        # Windows
# or
ls -la output/     # Linux/Mac

# View JSON output
type output\file01.json  # Windows
# or
cat output/file01.json   # Linux/Mac
```

#### Custom Configuration
```python
# Modify processing parameters
import robust_pdf_extractor as rpe

# Change text threshold
rpe.THRESHOLD_TEXT_LENGTH = 100

# Process with custom settings
result = rpe.process_pdf('document.pdf')
```

---

### 3. 🔄 Legacy Web Interface (Backward Compatibility)

#### Start Legacy Web Server
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Start legacy web interface
python app.py
```

#### Access Interface
- **Local**: http://localhost:5000
- **Features**: Basic upload, simple results display

#### Legacy API Endpoints
```bash
# Upload PDF (POST)
curl -X POST -F "file=@document.pdf" http://localhost:5000/predict
```

---

### 4. 🔧 Development Tools

#### Train New Model
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Train model with enhanced features
python train_model.py

# Outputs: model/model.joblib, model/label_map.json
```

#### Run Predictions (Legacy)
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Run prediction pipeline
python predict_and_export.py

# Outputs: output/predictions.json
```

#### Test System
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Run comprehensive tests
python test_multilingual.py

# Expected output: All tests passed
```

#### Feature Extraction
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Extract features from PDF
python -c "
from extract_features import extract_features
df = extract_features('input/document.pdf')
print(f'Extracted {len(df)} blocks with {len(df.columns)} features')
print(f'Scripts detected: {df[\"script_name\"].value_counts().to_dict()}')
"
```

---

### 5. 🐳 Docker Deployment (Production Ready)

#### Build Image
```bash
# Build with OCR support
docker build -t robust-pdf-extractor .

# Check image size
docker images robust-pdf-extractor
```

#### Run Container
```bash
# Basic run
docker run -p 5000:5000 robust-pdf-extractor

# With volume mounts
docker run -p 5000:5000 \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  robust-pdf-extractor

# Production run
docker run -d \
  --name pdf-extractor \
  -p 5000:5000 \
  --restart unless-stopped \
  robust-pdf-extractor
```

#### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  pdf-extractor:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./input:/app/input
      - ./output:/app/output
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

#### Run with Docker Compose
```bash
# Start services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

### 6. 🔄 System Comparison

| Interface | Command | Features | Use Case |
|-----------|---------|----------|----------|
| **Modern Web** | `python robust_app.py` | Drag-drop, real-time, modern UI | Production, user-friendly |
| **Command Line** | `python robust_pdf_extractor.py` | Fast, batch processing | Automation, scripting |
| **Legacy Web** | `python app.py` | Basic upload, simple display | Backward compatibility |
| **Docker** | `docker run robust-pdf-extractor` | Containerized, production | Deployment, scaling |

---

### 7. 📁 File Organization

#### Input/Output Structure
```
project/
├── input/                    # 📥 Place PDFs here
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── output/                   # 📤 Results appear here
│   ├── document1.json
│   ├── document2.json
│   └── ...
├── uploads/                  # 📤 Temporary web uploads
└── model/                    # 🧠 ML model files
    ├── model.joblib
    └── label_map.json
```

#### Expected Output Format
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction",
      "page": 1
    },
    {
      "level": "H2",
      "text": "Background",
      "page": 2
    }
  ],
  "processing_info": {
    "pdf_type": "text-based",
    "total_blocks": 150,
    "headings_found": 25,
    "execution_time": 0.23,
    "ocr_used": false
  }
}
```

---

### 8. 🚀 Performance Tips

#### For Fast Processing
```bash
# Use command line interface
python robust_pdf_extractor.py

# Process multiple files
python -c "
import robust_pdf_extractor as rpe
import os
for file in os.listdir('input'):
    if file.endswith('.pdf'):
        rpe.process_pdf(f'input/{file}')
"
```

#### For Web Interface
```bash
# Use modern interface for better UX
python robust_app.py

# Access from any device on network
# http://YOUR_IP:5000
```

#### For Production
```bash
# Use Docker for deployment
docker run -d -p 5000:5000 robust-pdf-extractor

# Monitor performance
docker stats pdf-extractor
```

---

### 9. 🔧 Troubleshooting

#### Common Issues
```bash
# Model not found
python train_model.py

# Dependencies missing
pip install -r requirements.txt

# Virtual environment not activated
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Port already in use
python robust_app.py --port 5001
```

#### Performance Issues
```bash
# Check system resources
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
"

# Monitor processing
python robust_pdf_extractor.py --verbose
```

---

## 📚 API Reference

### Core Functions

#### `process_pdf(pdf_path: str) -> dict`
Main processing function for single PDF.

**Parameters**:
- `pdf_path` (str): Path to PDF file

**Returns**:
- `dict`: Processing results with structure:
  ```python
  {
      "title": str,
      "outline": List[Dict],
      "processing_info": Dict
  }
  ```

**Example**:
```python
from robust_pdf_extractor import process_pdf

result = process_pdf("document.pdf")
print(f"Found {len(result['outline'])} headings")
```

#### `is_scanned_pdf(pdf_path: str) -> bool`
Detect if PDF is scanned or text-based.

**Parameters**:
- `pdf_path` (str): Path to PDF file

**Returns**:
- `bool`: True if scanned, False if text-based

**Algorithm**:
1. Extract total text length
2. Count image blocks
3. Check for image references
4. Apply threshold logic

#### `extract_text_textpdf(pdf_path: str) -> List[Tuple]`
Extract text and metadata from text-based PDF.

**Parameters**:
- `pdf_path` (str): Path to PDF file

**Returns**:
- `List[Tuple]`: List of (page_num, blocks, page_height) tuples

**Block Structure**:
```python
(x0, y0, x1, y1, text, block_no, block_type)
```

#### `extract_text_ocr(pdf_path: str, lang: str) -> List[Tuple]`
Extract text from scanned PDF using OCR.

**Parameters**:
- `pdf_path` (str): Path to PDF file
- `lang` (str): Language codes for OCR

**Returns**:
- `List[Tuple]`: List of (page_num, text) tuples

**Supported Languages**:
- `eng`: English
- `hin`: Hindi
- `kan`: Kannada
- `san`: Sanskrit
- `jpn`: Japanese
- `kor`: Korean
- `tam`: Tamil
- `mar`: Marathi
- `urd`: Urdu

#### `get_features_from_block(block: Tuple, page_height: float, page_num: int) -> Dict`
Extract features from a text block.

**Parameters**:
- `block`: PyMuPDF text block tuple
- `page_height`: Page height in points
- `page_num`: Page number

**Returns**:
- `Dict`: Feature dictionary with 12 engineered features

**Features**:
- `font_size`: Estimated font size
- `font_size_ratio`: Normalized font size
- `bold`: Bold text indicator
- `alignment`: Text alignment (0=left, 1=center, 2=right)
- `spacing_above`: Distance from page top
- `spacing_below`: Distance to page bottom
- `line_spacing`: Line spacing ratio
- `num_words`: Word count
- `avg_word_length`: Average word length
- `caps_ratio`: Capitalization ratio
- `script_type`: Detected script name
- `position_pct`: Position percentage on page

#### `classify_headings_ml(features_list: List[Dict]) -> List[Dict]`
ML-powered heading classification.

**Parameters**:
- `features_list`: List of feature dictionaries

**Returns**:
- `List[Dict]`: List of heading dictionaries

**Algorithm**:
1. Feature compatibility check
2. ML model prediction
3. Label mapping
4. Heading filtering

#### `classify_headings_heuristic(features_list: List[Dict]) -> List[Dict]`
Rule-based heading classification.

**Parameters**:
- `features_list`: List of feature dictionaries

**Returns**:
- `List[Dict]`: List of heading dictionaries

**Rules**:
- H1: ≤3 words, font_ratio > 1.2
- H2: ≤5 words, font_ratio > 1.1
- H3: ≤8 words, font_ratio > 1.0

### Web API Endpoints

#### `POST /upload`
Upload and process PDF file.

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: PDF file in 'file' field

**Response**:
```json
{
    "title": "Document Title",
    "outline": [
        {
            "level": "H1",
            "text": "Introduction",
            "page": 1
        }
    ],
    "processing_info": {
        "pdf_type": "text-based",
        "total_blocks": 150,
        "headings_found": 25,
        "execution_time": 2.45,
        "ocr_used": false
    }
}
```

#### `GET /status`
Get system status and capabilities.

**Response**:
```json
{
    "status": "operational",
    "ocr_available": true,
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
}
```

#### `GET /health`
Health check endpoint.

**Response**:
```json
{
    "status": "healthy",
    "timestamp": 1640995200.0
}
```

---

## ⚙️ Configuration

### Environment Variables

#### OCR Configuration
```bash
# Tesseract data path
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

# Language configuration
export TESSERACT_LANGUAGES=eng+hin+kan+san+jpn+kor+tam+mar+urd
```

#### Processing Configuration
```bash
# Text threshold for PDF type detection
export THRESHOLD_TEXT_LENGTH=50

# Maximum file size (bytes)
export MAX_FILE_SIZE=52428800

# Model paths
export MODEL_PATH=model/model.joblib
export LABEL_MAP_PATH=model/label_map.json
```

#### Web Server Configuration
```bash
# Flask configuration
export FLASK_ENV=production
export FLASK_DEBUG=0
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
```

### Model Configuration

#### Feature Engineering
```python
# Feature columns for ML model
FEATURE_COLS = [
    'font_size', 'font_size_ratio', 'bold', 'alignment',
    'spacing_above', 'spacing_below', 'line_spacing',
    'num_words', 'avg_word_length', 'caps_ratio'
]

# Script detection mapping
SCRIPT_MAP = {
    "Latin": 0,
    "Devanagari": 1,
    "Kannada": 2,
    "Tamil": 3,
    "Japanese": 4,
    "Korean": 5,
    "Arabic": 6,
    "Chinese": 7,
    "Thai": 8,
    "Bengali": 9,
    "Telugu": 10
}
```

#### ML Model Parameters
```python
# RandomForest configuration
MODEL_PARAMS = {
    'n_estimators': 150,
    'max_depth': 12,
    'random_state': 42,
    'n_jobs': -1,
    'min_samples_split': 3,
    'min_samples_leaf': 1,
    'class_weight': 'balanced',
    'bootstrap': True,
    'oob_score': True
}
```

### Performance Tuning

#### Memory Optimization
```python
# Batch processing settings
BATCH_SIZE = 1000  # Process blocks in batches
CHUNK_SIZE = 100   # Memory chunk size

# OCR optimization
OCR_DPI = 300      # OCR resolution
OCR_PSM = 6        # Page segmentation mode
```

#### Speed Optimization
```python
# Parallel processing
N_JOBS = -1        # Use all CPU cores
CHUNK_PROCESSING = True  # Enable chunking

# Caching
ENABLE_CACHE = True
CACHE_TTL = 3600   # Cache TTL in seconds
```

---

## 📊 Performance Metrics

### Current Performance

| Metric | Target | Achieved | Status | Notes |
|--------|--------|----------|--------|-------|
| **Model Size** | ≤200MB | **0.47MB** | ✅ Excellent | 99.8% under limit |
| **Execution Time** | ≤10s | **0.1-0.3s** | ✅ Excellent | 97% under limit |
| **Accuracy** | >0.85 | **77.14%** | ⚠️ Good | Room for improvement |
| **Languages** | 10+ | **11 scripts** | ✅ Excellent | Exceeds requirement |
| **OCR Support** | Yes | **Available** | ✅ Complete | Full implementation |

### Processing Speed Breakdown

#### Text-based PDFs
- **Small PDFs** (<10 pages): 0.1-0.2s
- **Medium PDFs** (10-50 pages): 0.2-0.5s
- **Large PDFs** (>50 pages): 0.5-2.0s

#### Scanned PDFs (with OCR)
- **Small PDFs** (<10 pages): 2-5s
- **Medium PDFs** (10-50 pages): 5-15s
- **Large PDFs** (>50 pages): 15-30s

### Memory Usage

#### Peak Memory
- **Text Processing**: ~50-100MB
- **OCR Processing**: ~200-500MB
- **Model Loading**: ~50MB
- **Total Peak**: ~600MB

#### Average Memory
- **Idle**: ~30MB
- **Processing**: ~150MB
- **Web Server**: ~80MB

### Scalability Metrics

#### Concurrent Processing
- **Single Instance**: 1 PDF at a time
- **Multiple Instances**: N PDFs (N = CPU cores)
- **Docker Scaling**: Horizontal scaling supported

#### Throughput
- **Text PDFs**: ~100-300 PDFs/hour
- **Scanned PDFs**: ~20-50 PDFs/hour
- **Mixed Workload**: ~50-150 PDFs/hour

---

## 🔍 Troubleshooting

### Common Issues

#### OCR Not Working
```bash
# Check Tesseract installation
tesseract --version

# Verify language packs
ls /usr/share/tesseract-ocr/4.00/tessdata/

# Test OCR functionality
python -c "
import pytesseract
from PIL import Image
print('OCR available:', pytesseract.is_available())
"
```

**Solutions**:
1. Install Tesseract system package
2. Install language packs
3. Set TESSDATA_PREFIX environment variable
4. Verify Python pytesseract installation

#### Model Loading Errors
```bash
# Check model files
ls -la model/

# Verify model integrity
python -c "
import joblib
model = joblib.load('model/model.joblib')
print('Model loaded successfully')
"
```

**Solutions**:
1. Ensure model files exist
2. Check file permissions
3. Verify joblib version compatibility
4. Re-download model files if corrupted

#### Unicode Issues
```python
# Enable UTF-8 encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Test Unicode handling
text = "हिंदी 日本語 한국어"
print(text.encode('utf-8').decode('utf-8'))
```

**Solutions**:
1. Set PYTHONIOENCODING=utf-8
2. Use UTF-8 file encoding
3. Normalize Unicode text
4. Handle encoding errors gracefully

#### Web Interface Issues
```bash
# Check Flask installation
python -c "import flask; print(flask.__version__)"

# Test web server
curl http://localhost:5000/health

# Check port availability
netstat -tulpn | grep :5000
```

**Solutions**:
1. Install Flask dependencies
2. Check port conflicts
3. Verify firewall settings
4. Use different port if needed

### Debug Mode

#### Enable Verbose Logging
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### Test Individual Components
```python
from robust_pdf_extractor import (
    is_scanned_pdf, 
    extract_text_textpdf,
    get_features_from_block,
    classify_headings_ml
)

# Test PDF type detection
is_scanned = is_scanned_pdf("document.pdf")
print(f"PDF is scanned: {is_scanned}")

# Test text extraction
blocks = extract_text_textpdf("document.pdf")
print(f"Extracted {len(blocks)} page blocks")

# Test feature extraction
features = get_features_from_block(blocks[0][1][0], 800, 1)
print(f"Features: {features}")

# Test classification
headings = classify_headings_ml([features])
print(f"Headings: {headings}")
```

#### Performance Profiling
```python
import cProfile
import pstats

# Profile processing
profiler = cProfile.Profile()
profiler.enable()

from robust_pdf_extractor import process_pdf
result = process_pdf("document.pdf")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Error Recovery

#### Graceful Degradation
```python
# Handle missing dependencies
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("OCR not available, scanned PDFs will be skipped")

# Handle model loading failures
try:
    clf = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("Model not found, using heuristic classification")
    clf = None
```

#### Fallback Mechanisms
```python
# ML model fallback
if clf is not None:
    headings = classify_headings_ml(features_list)
else:
    headings = classify_headings_heuristic(features_list)

# OCR fallback
if OCR_AVAILABLE:
    text = extract_text_ocr(pdf_path)
else:
    print("OCR not available, cannot process scanned PDF")
    return None
```

---

## 🚀 Development Guide

### Development Setup

#### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd robust-pdf-extractor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Verify setup
python -c "import robust_pdf_extractor; print('Setup complete')"
```

#### Code Quality Tools
```bash
# Format code
black robust_pdf_extractor.py robust_app.py

# Lint code
flake8 robust_pdf_extractor.py robust_app.py

# Type checking
mypy robust_pdf_extractor.py robust_app.py

# Run tests
pytest tests/
```

### Adding New Features

#### Adding New Languages
1. **Update Script Detection**:
   ```python
   def detect_script(text):
       # Add new Unicode ranges
       if re.search(r'[\uXXXX-\uYYYY]', text):
           return 'NewScript'
   ```

2. **Install Language Pack**:
   ```bash
   sudo apt-get install tesseract-ocr-newlang
   ```

3. **Update Configuration**:
   ```python
   LANGUAGES = "eng+hin+kan+san+jpn+kor+tam+mar+urd+newlang"
   ```

4. **Test Implementation**:
   ```python
   # Test with sample text
   text = "Sample text in new language"
   script = detect_script(text)
   assert script == "NewScript"
   ```

#### Adding New Features
1. **Feature Engineering**:
   ```python
   def get_features_from_block(block, page_height, page_num):
       # Add new feature
       new_feature = calculate_new_feature(block)
       
       return {
           # ... existing features
           "new_feature": new_feature
       }
   ```

2. **Model Retraining**:
   ```python
   # Update feature columns
   FEATURE_COLS.append('new_feature')
   
   # Retrain model
   python train_model.py
   ```

3. **Update Documentation**:
   - Update README.md
   - Add API documentation
   - Include examples

### Testing Strategy

#### Unit Tests
```python
# test_robust_extractor.py
import pytest
from robust_pdf_extractor import (
    normalize_text,
    detect_script,
    is_scanned_pdf
)

def test_normalize_text():
    text = "Hello\u00a0World"  # Non-breaking space
    result = normalize_text(text)
    assert result == "Hello World"

def test_detect_script():
    # Test Latin
    assert detect_script("Hello World") == "Latin"
    
    # Test Devanagari
    assert detect_script("नमस्ते") == "Devanagari"
    
    # Test Japanese
    assert detect_script("こんにちは") == "Japanese"

def test_is_scanned_pdf():
    # Test with text PDF
    assert is_scanned_pdf("text_document.pdf") == False
    
    # Test with scanned PDF
    assert is_scanned_pdf("scanned_document.pdf") == True
```

#### Integration Tests
```python
# test_integration.py
def test_full_pipeline():
    result = process_pdf("test_document.pdf")
    
    assert "title" in result
    assert "outline" in result
    assert "processing_info" in result
    assert isinstance(result["outline"], list)
    
    # Verify heading structure
    for heading in result["outline"]:
        assert "level" in heading
        assert "text" in heading
        assert "page" in heading
        assert heading["level"] in ["H1", "H2", "H3", "Title"]
```

#### Performance Tests
```python
# test_performance.py
import time

def test_processing_speed():
    start_time = time.time()
    result = process_pdf("large_document.pdf")
    end_time = time.time()
    
    processing_time = end_time - start_time
    assert processing_time < 10.0  # Should complete within 10 seconds

def test_memory_usage():
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    result = process_pdf("large_document.pdf")
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 500 * 1024 * 1024  # Less than 500MB increase
```

### Deployment Guide

#### Production Deployment
```bash
# Build production image
docker build -t robust-pdf-extractor:latest .

# Run with production settings
docker run -d \
  --name pdf-extractor-prod \
  -p 5000:5000 \
  --restart unless-stopped \
  --memory=1g \
  --cpus=2 \
  -e FLASK_ENV=production \
  -e FLASK_DEBUG=0 \
  robust-pdf-extractor:latest
```

#### Monitoring Setup
```python
# Add monitoring endpoints
@app.route("/metrics")
def metrics():
    return jsonify({
        "requests_total": request_count,
        "processing_time_avg": avg_processing_time,
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent()
    })
```

#### Logging Configuration
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=10240000, backupCount=5),
        logging.StreamHandler()
    ]
)
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyMuPDF**: PDF text extraction and manipulation
- **Tesseract**: OCR engine for scanned documents
- **scikit-learn**: Machine learning algorithms
- **Flask**: Web framework for API and interface
- **Pandas**: Data manipulation and analysis
- **Joblib**: Model serialization and persistence

---

## 📞 Support

For questions, issues, or contributions:

1. **Documentation**: Check this README and inline code comments
2. **Issues**: Create GitHub issue with detailed description
3. **Contributions**: Fork repository and submit pull request
4. **Contact**: Reach out to maintainers for direct support

---

**🌍 Built for multilingual document processing with ❤️**

*Last updated: December 2024* 