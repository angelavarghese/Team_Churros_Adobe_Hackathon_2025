#!/usr/bin/env python3
"""
Robust Multilingual PDF Heading Extractor
Supports both text-based and scanned PDFs with OCR fallback
Handles 10+ languages: English, Hindi, Kannada, Sanskrit, Japanese, Korean, Tamil, Marathi, Urdu, etc.
"""

import os
import time
import fitz  # PyMuPDF
import json
import unicodedata
import re
import pandas as pd
from joblib import load
import numpy as np

# Try to import OCR dependencies (optional)
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR dependencies not available. Install with: pip install pytesseract pdf2image")

# ---------- DYNAMIC CONFIGURATION ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "model.joblib")
LABEL_MAP_PATH = os.path.join(SCRIPT_DIR, "model", "label_map.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Language configuration
LANGUAGES = "eng+hin+kan+san+jpn+kor+tam+mar+urd"  # Supported languages
THRESHOLD_TEXT_LENGTH = 50  # Minimum text length to consider PDF as text-based

# Load model and label map
try:
    clf = load(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    
    # Calculate model size
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ Model loaded successfully (Size: {model_size_mb:.2f} MB)")
except FileNotFoundError:
    print("⚠️ Model not found. Using heuristic-based classification")
    clf = None
    label_map = None
    model_size_mb = 0.0

# ---------- HELPER FUNCTIONS ----------

def clean_ocr_text(text):
    """Clean OCR text by removing noise and adding placeholders for short text."""
    # Remove unwanted symbols but keep Latin, Devanagari, Japanese, Korean, spaces
    text = re.sub(r'[^A-Za-z\u0900-\u097F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF ]', '', text)
    text = text.strip()

    # Detect if Japanese characters exist
    if re.search(r'[\u3040-\u30FF\u4E00-\u9FFF]', text):
        if len(text) < 2:
            return "Japanese Text"  # Placeholder
        return text

    # Detect if Devanagari (Hindi/Sanskrit/Marathi)
    if re.search(r'[\u0900-\u097F]', text):
        if len(text) < 2:
            return "Hindi/Sanskrit Text"  # Placeholder
        return text

    # Detect if Korean
    if re.search(r'[\uAC00-\uD7AF]', text):
        if len(text) < 2:
            return "Korean Text"  # Placeholder
        return text

    # Remove very short text (noise)
    if len(text) < 3:
        return None

    return text

def normalize_text(text):
    """Normalize text using Unicode normalization."""
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8')
        except UnicodeDecodeError:
            text = text.decode('utf-8', errors='replace')
    
    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text).strip()
    
    # Clean up any remaining replacement characters
    text = text.replace('\ufffd', '')
    
    return text

def detect_script(text):
    """Detect script for multilingual support."""
    # Devanagari (Hindi, Sanskrit, Marathi)
    if re.search(r'[\u0900-\u097F]', text):
        return 'Devanagari'
    # Kannada
    elif re.search(r'[\u0C80-\u0CFF]', text):
        return 'Kannada'
    # Tamil
    elif re.search(r'[\u0B80-\u0BFF]', text):
        return 'Tamil'
    # Japanese (Hiragana, Katakana, Kanji)
    elif re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return 'Japanese'
    # Korean (Hangul)
    elif re.search(r'[\uAC00-\uD7AF]', text):
        return 'Korean'
    # Arabic (Urdu)
    elif re.search(r'[\u0600-\u06FF]', text):
        return 'Arabic'
    # Chinese (Simplified/Traditional)
    elif re.search(r'[\u4E00-\u9FFF]', text):
        return 'Chinese'
    # Thai
    elif re.search(r'[\u0E00-\u0E7F]', text):
        return 'Thai'
    # Bengali
    elif re.search(r'[\u0980-\u09FF]', text):
        return 'Bengali'
    # Telugu
    elif re.search(r'[\u0C00-\u0C7F]', text):
        return 'Telugu'
    # Default to Latin (English and other Latin-based scripts)
    else:
        return 'Latin'

def calculate_caps_ratio(text, script_type):
    """Calculate capitalization ratio only for scripts that support it."""
    if script_type in ['Latin']:
        if text:
            caps_count = sum(1 for c in text if c.isupper())
            return caps_count / len(text) if text else 0
    return 0.0

# ---------- PDF TYPE DETECTION ----------

def is_scanned_pdf(pdf_path):
    """Detect if PDF is scanned (image-based) or text-based."""
    doc = fitz.open(pdf_path)
    total_text = ""
    total_blocks = 0
    image_blocks = 0
    text_blocks = 0
    
    for page in doc:
        text = page.get_text()
        total_text += text
        blocks = page.get_text("blocks")
        total_blocks += len(blocks)
        
        # Count image vs text blocks
        for block in blocks:
            if block[6] == 1:  # Image block
                image_blocks += 1
            elif block[6] == 0:  # Text block
                text_blocks += 1
    
    doc.close()
    
    # Check if text is too short or contains mostly image blocks
    text_length = len(total_text.strip())
    print(f"📄 PDF Analysis: {text_length} characters, {total_blocks} blocks ({text_blocks} text, {image_blocks} images)")
    
    # Consider scanned if:
    # 1. Very little text (< threshold)
    # 2. Text contains mostly image references
    # 3. No meaningful text blocks
    # 4. High ratio of image blocks to total blocks (>70%)
    # 5. Low text-to-block ratio (<0.5 characters per block)
    is_scanned = (
        text_length < THRESHOLD_TEXT_LENGTH or
        total_text.count("<image") > len(total_text) * 0.1 or
        total_blocks == 0 or
        (total_blocks > 0 and image_blocks / total_blocks > 0.7) or
        (total_blocks > 0 and text_length / total_blocks < 0.5)
    )
    
    return is_scanned

# ---------- TEXT EXTRACTION ----------

def extract_text_textpdf(pdf_path):
    """Extract text and metadata from text-based PDF."""
    doc = fitz.open(pdf_path)
    page_blocks = []
    
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        page_blocks.append((page_num, blocks, page.rect.height))
    
    doc.close()
    return page_blocks

def extract_text_ocr(pdf_path, lang=LANGUAGES):
    """Extract text from scanned PDF using OCR."""
    if not OCR_AVAILABLE:
        print("❌ OCR not available. Cannot process scanned PDF.")
        return []
    
    try:
        print(f"🔍 Running OCR with languages: {lang}")
        images = convert_from_path(pdf_path)
        page_texts = []
        
        for page_num, img in enumerate(images, start=1):
            print(f"📄 Processing page {page_num} with OCR...")
            # Optimize OCR settings for better performance
            text = pytesseract.image_to_string(
                img, 
                lang=lang, 
                config='--psm 6 --oem 1 -c tessedit_do_invert=0'
            )
            page_texts.append((page_num, text))
        
        return page_texts
    except Exception as e:
        print(f"❌ OCR failed: {e}")
        return []

# ---------- FEATURE EXTRACTION ----------

def get_features_from_block(block, page_height, page_num):
    """Extract features from a text block."""
    # block: (x0, y0, x1, y1, text, block_no, block_type)
    text = normalize_text(block[4])
    
    # Skip image blocks or empty text
    if not text or text.startswith("<image") or len(text.strip()) < 2:
        return None
    
    # Basic features
    x0, y0, x1, y1 = block[:4]
    text_clean = text.strip()
    words = text_clean.split()
    
    # Font size estimation
    font_size = (y1 - y0) / max(len(words), 1)
    
    # Alignment detection
    page_center = page_height / 2
    block_center = (x0 + x1) / 2
    is_centered = abs(block_center - page_center) < 50
    
    # Position percentage
    position_pct = y0 / page_height if page_height > 0 else 0
    
    # Script detection
    script_name = detect_script(text_clean)
    
    # Word features
    num_words = len(words)
    avg_word_length = sum(len(w) for w in words) / num_words if num_words > 0 else 0
    
    # Capitalization ratio
    caps_ratio = calculate_caps_ratio(text_clean, script_name)
    
    return {
        "text": text_clean,
        "font_size": font_size,
        "font_size_ratio": font_size / 12.0,  # Normalize to default size
        "bold": 0,  # Default, can be enhanced
        "alignment": 1 if is_centered else 0,
        "spacing_above": y0,
        "spacing_below": page_height - y1 if page_height > 0 else 0,
        "line_spacing": 1.0,  # Default
        "num_words": num_words,
        "avg_word_length": avg_word_length,
        "caps_ratio": caps_ratio,
        "script_type": script_name,
        "position_pct": position_pct,
        "page": page_num
    }

def get_features_from_ocr_text(text, page_num, line_num, total_lines):
    """Extract features from OCR text line."""
    text_clean = normalize_text(text)
    
    if not text_clean or len(text_clean.strip()) < 2:
        return None
    
    words = text_clean.split()
    num_words = len(words)
    
    # More lenient word limit for OCR (some headings can be longer)
    if num_words > 25:
        return None
    
    # Estimate position based on line number
    position_pct = line_num / total_lines if total_lines > 0 else 0
    
    # Script detection
    script_name = detect_script(text_clean)
    
    # Enhanced heuristic features for OCR text
    # Consider short lines, lines with numbers, or lines with special characters as potential headings
    is_potential_heading = (
        num_words <= 8 or  # Short lines
        any(char.isdigit() for char in text_clean) or  # Contains numbers
        any(char in '一二三四五六七八九十百千万' for char in text_clean) or  # Japanese/Chinese numbers
        any(char in '१२३४५६७८९०' for char in text_clean) or  # Devanagari numbers
        text_clean.strip().endswith('.') or  # Ends with period
        text_clean.strip().endswith('：') or  # Ends with Japanese colon
        text_clean.strip().endswith(':')  # Ends with colon
    )
    
    if not is_potential_heading:
        return None
    
    # Font size estimation based on line characteristics
    font_size_ratio = 1.8 if num_words <= 3 else (1.4 if num_words <= 6 else 1.1)
    caps_ratio = calculate_caps_ratio(text_clean, script_name)
    
    return {
        "text": text_clean,
        "font_size": 12,  # Default for OCR
        "font_size_ratio": font_size_ratio,
        "bold": 0,
        "alignment": 0,  # Assume left-aligned for OCR
        "spacing_above": position_pct * 1000,  # Estimate
        "spacing_below": (1 - position_pct) * 1000,
        "line_spacing": 1.0,
        "num_words": num_words,
        "avg_word_length": sum(len(w) for w in words) / num_words if num_words > 0 else 0,
        "caps_ratio": caps_ratio,
        "script_type": script_name,
        "position_pct": position_pct,
        "page": page_num
    }

# ---------- HEADING CLASSIFICATION ----------

def classify_headings_ml(features_list):
    """Classify headings using trained ML model."""
    if not clf or not features_list:
        return []
    
    # Prepare features for ML model
    feature_cols = [
        'font_size', 'font_size_ratio', 'bold', 'alignment',
        'spacing_above', 'spacing_below', 'line_spacing',
        'num_words', 'avg_word_length', 'caps_ratio'
    ]
    
    # Filter available features
    available_features = [col for col in feature_cols if col in features_list[0]]
    
    if not available_features:
        print("⚠️ No compatible features found for ML model")
        return classify_headings_heuristic(features_list)
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    
    # Check if model expects specific features
    if hasattr(clf, 'feature_names_in_'):
        model_features = list(clf.feature_names_in_)
        available_features = [col for col in available_features if col in model_features]
    
    if not available_features:
        print("⚠️ Feature mismatch with model, using heuristic")
        return classify_headings_heuristic(features_list)
    
    try:
        # Make predictions
        predictions = clf.predict(df[available_features])
        
        # Create outline
        outline = []
        for idx, pred in enumerate(predictions):
            if label_map and str(pred) in label_map:
                label = label_map[str(pred)]
                if label in ["H1", "H2", "H3", "Title"]:
                    outline.append({
                        "level": label,
                        "text": features_list[idx]["text"],
                        "page": features_list[idx]["page"]
                    })
        
        return outline
    except Exception as e:
        print(f"⚠️ ML classification failed: {e}")
        return classify_headings_heuristic(features_list)

def classify_headings_heuristic(features_list):
    """Classify headings using heuristic rules."""
    outline = []
    
    for features in features_list:
        text = features["text"]
        num_words = features["num_words"]
        font_size_ratio = features.get("font_size_ratio", 1.0)
        position_pct = features.get("position_pct", 0.5)
        
        # Heuristic rules
        if num_words <= 3 and font_size_ratio > 1.2:
            level = "H1"
        elif num_words <= 5 and font_size_ratio > 1.1:
            level = "H2"
        elif num_words <= 8 and font_size_ratio > 1.0:
            level = "H3"
        else:
            continue  # Skip body text
        
        outline.append({
            "level": level,
            "text": text,
            "page": features["page"]
        })
    
    return outline

# ---------- MAIN PIPELINE ----------

def process_pdf(pdf_path):
    """Main pipeline for processing PDF and extracting headings."""
    start_time = time.time()
    
    print(f"\n🔍 Processing: {os.path.basename(pdf_path)}")
    
    # Step 1: Detect PDF type
    scanned = is_scanned_pdf(pdf_path)
    
    # Step 2: Extract text based on type
    if scanned:
        print("📄 Scanned PDF detected. Using OCR...")
        pages_text = extract_text_ocr(pdf_path)
        if not pages_text:
            print("❌ OCR extraction failed")
            return None
        
        # Extract features from OCR text
        features_list = []
        for page_num, text in pages_text:
            lines = text.split('\n')
            for line_num, line in enumerate(lines):
                cleaned_line = clean_ocr_text(line)
                if cleaned_line:
                    features = get_features_from_ocr_text(cleaned_line, page_num, line_num, len(lines))
                    if features:
                        features_list.append(features)
        
        print(f"📊 Extracted {len(features_list)} text blocks from OCR")
        
    else:
        print("📄 Text-based PDF detected. Extracting structured blocks...")
        page_blocks = extract_text_textpdf(pdf_path)
        
        # Extract features from text blocks
        features_list = []
        for page_num, blocks, page_height in page_blocks:
            for block in blocks:
                features = get_features_from_block(block, page_height, page_num)
                if features:
                    features_list.append(features)
        
        print(f"📊 Extracted {len(features_list)} text blocks")
    
    # Step 3: Classify headings
    if features_list:
        outline = classify_headings_ml(features_list)
        print(f"📊 Found {len(outline)} headings")
    else:
        outline = []
        print("⚠️ No text blocks found")
    
    # Step 4: Generate output (Hackathon compliant format)
    title = os.path.basename(pdf_path).replace(".pdf", "")
    output = {
        "title": title,
        "outline": outline
    }
    
    # Step 5: Save output
    output_path = os.path.join(OUTPUT_DIR, f"{title}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    exec_time = time.time() - start_time
    print(f"✅ Completed in {exec_time:.2f}s")
    print(f"📊 Headings detected: {len(outline)}")
    print(f"📁 Output saved to: {output_path}")
    
    # Check hackathon constraints
    if exec_time > 10.0:
        print(f"⚠️ Warning: Execution time ({exec_time:.2f}s) exceeds 10s limit")
    if model_size_mb > 200.0:
        print(f"⚠️ Warning: Model size ({model_size_mb:.2f}MB) exceeds 200MB limit")
    
    return output

# ---------- BATCH PROCESSING ----------

def process_directory(input_dir="input"):
    """Process all PDFs in a directory."""
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_dir}")
        return
    
    print(f"🔍 Found {len(pdf_files)} PDF files to process")
    
    results = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        try:
            result = process_pdf(pdf_path)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
    
    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"   Total files: {len(pdf_files)}")
    print(f"   Successful: {len(results)}")
    print(f"   Failed: {len(pdf_files) - len(results)}")
    
    return results

# ---------- MAIN EXECUTION ----------

if __name__ == "__main__":
    print("Robust Multilingual PDF Heading Extractor")
    print("Adobe Hackathon Round 1A - Compliant Version")
    print("=" * 50)
    
    # Display model information
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Supported languages: {LANGUAGES}")
    print(f"OCR available: {OCR_AVAILABLE}")
    
    # Check dependencies
    if not OCR_AVAILABLE:
        print("Warning: OCR dependencies not available. Scanned PDFs will not be processed.")
        print("Install with: pip install pytesseract pdf2image")
    
    # Process PDFs from input directory
    input_dir = os.path.join(SCRIPT_DIR, "input")
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        print("Creating input directory...")
        os.makedirs(input_dir, exist_ok=True)
        print(f"Please place PDF files in: {input_dir}")
        exit(1)
    
    results = process_directory(input_dir)
    
    if results:
        print(f"\nSuccessfully processed {len(results)} PDFs!")
        print(f"Check the '{OUTPUT_DIR}' directory for results.")
    else:
        print("\nNo PDFs were successfully processed.")
        print(f"Please ensure PDF files are placed in: {input_dir}") 