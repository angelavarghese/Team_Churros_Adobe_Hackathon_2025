# extract_features.py

import fitz  # PyMuPDF
import re
import pandas as pd
import unicodedata

def normalize_text(text):
    """Normalize text using Unicode normalization to handle mixed scripts."""
    # First, try to decode if it's bytes
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = text.decode('latin-1')
            except UnicodeDecodeError:
                text = text.decode('utf-8', errors='replace')
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Clean up any remaining replacement characters
    text = text.replace('\ufffd', '')  # Remove replacement characters
    
    return text.strip()

def detect_script(text):
    """
    Detect script for 10 languages using Unicode ranges.
    Returns script name for multilingual awareness.
    """
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
    
    # Chinese (Simplified/Traditional) - only if no Japanese characters
    elif re.search(r'[\u4E00-\u9FFF]', text):
        # Check if it's mixed with Japanese
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
            return 'Japanese'
        else:
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

def script_to_numeric(script_name):
    """Convert script name to numeric for ML training."""
    script_map = {
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
    return script_map.get(script_name, 0)

def calculate_caps_ratio(text, script_type):
    """Calculate capitalization ratio only for scripts that support it."""
    # Only Latin-based scripts support traditional capitalization
    if script_type in ['Latin']:
        if text:
            caps_count = sum(1 for c in text if c.isupper())
            return caps_count / len(text) if text else 0
    return 0.0

def extract_text_with_enhanced_features(page, page_num):
    """
    Extract text from page with enhanced features.
    Returns list of text blocks with enhanced features.
    """
    blocks = []
    
    # Get text blocks
    text_blocks = page.get_text("blocks")
    
    # Process normal text blocks
    for block in text_blocks:
        x0, y0, x1, y1, text, *_ = block
        
        if text.strip():
            # Extract font information
            font_size = 12
            is_bold = False
            
            try:
                text_dict = page.get_text("dict")
                for b in text_dict["blocks"]:
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            if span["text"] in text:
                                font_size = span["size"]
                                font_name = span.get("font", "").lower()
                                is_bold = any(bold_indicator in font_name for bold_indicator in 
                                            ["bold", "black", "heavy", "semibold"])
                                break
                        if font_size != 12:
                            break
                    if font_size != 12:
                        break
            except Exception:
                pass
            
            blocks.append({
                'text': text.strip(),
                'bbox': [x0, y0, x1, y1],
                'font_size': font_size,
                'is_bold': is_bold,
                'is_centered': abs((x0 + x1) / 2 - page.rect.width / 2) < 20,
                'page_num': page_num,
                'source': 'text'
            })
    
    return blocks

def extract_features(pdf_path):
    """
    Extract structural features from PDF blocks for ML training.
    Now supports enhanced multilingual features with improved accuracy.
    Returns: pandas DataFrame with features for each text block.
    """
    doc = fitz.open(pdf_path)
    rows = []
    
    print(f"🔍 Processing PDF: {pdf_path}")
    print(f"📄 Total pages: {len(doc)}")
    
    for page_num, page in enumerate(doc, start=1):
        # Extract text blocks with enhanced features
        blocks = extract_text_with_enhanced_features(page, page_num)
        
        for block in blocks:
            text = block['text']
            
            # Skip empty blocks
            if not text:
                continue
                
            # Clean and normalize text with improved Unicode handling
            text = normalize_text(text)
            
            # Skip if text is empty after normalization
            if not text:
                continue
            
            # Detect script type
            script_name = detect_script(text)
            script_type = script_to_numeric(script_name)
            
            # Get block properties
            bbox = block['bbox']
            x0, y0, x1, y1 = bbox
            font_size = block['font_size']
            is_bold = block['is_bold']
            is_centered = block['is_centered']
            
            # Calculate enhanced features
            words = text.split()
            word_count = len(words)
            avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
            
            # Determine alignment (left=0, center=1, right=2)
            if is_centered:
                alignment = 1
            elif x0 < page.rect.width * 0.3:  # left-aligned
                alignment = 0
            else:  # right-aligned
                alignment = 2
            
            # Calculate caps ratio (language-aware)
            caps_ratio = calculate_caps_ratio(text, script_name)
            
            # Position percentage (relative to page height)
            page_height = page.rect.height
            position_pct = y0 / page_height if page_height > 0 else 0
            
            # Enhanced spacing features
            spacing_above = y0
            spacing_below = page_height - y1 if page_height > 0 else 0
            
            # Font size ratio (relative to median)
            font_size_ratio = font_size / 12.0  # Normalize to default size
            
            # Line spacing (simplified)
            line_spacing = 1.0  # Default, can be enhanced
            
            rows.append({
                "text": text,
                "font_size": font_size,
                "font_size_ratio": font_size_ratio,
                "bold": int(is_bold),
                "alignment": alignment,
                "spacing_above": spacing_above,
                "spacing_below": spacing_below,
                "line_spacing": line_spacing,
                "num_words": word_count,
                "avg_word_length": avg_word_length,
                "caps_ratio": caps_ratio,
                "script_type": script_type,
                "script_name": script_name,
                "page_num": page_num,
                "source": block['source']
            })
    
    df = pd.DataFrame(rows)
    
    # Print processing summary
    if not df.empty:
        print(f"✅ Extracted {len(df)} blocks")
        if 'script_name' in df.columns:
            script_counts = df['script_name'].value_counts()
            print(f"📊 Script distribution: {script_counts.to_dict()}")
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            print(f"📊 Source distribution: {source_counts.to_dict()}")
    
    return df

def extract_features_from_pdf(pdf_path):
    """
    Legacy function for backward compatibility.
    Returns: List of dicts with features for each block.
    """
    df = extract_features(pdf_path)
    
    blocks_with_features = []
    for idx, row in df.iterrows():
        features = {
            "font_size": row["font_size"],
            "font_size_ratio": row.get("font_size_ratio", 1.0),
            "bold": row["bold"],
            "alignment": row["alignment"],
            "spacing_above": row["spacing_above"],
            "spacing_below": row["spacing_below"],
            "line_spacing": row.get("line_spacing", 1.0),
            "num_words": row["num_words"],
            "avg_word_length": row["avg_word_length"],
            "caps_ratio": row["caps_ratio"],
            "script_type": row["script_type"]
        }
        
        blocks_with_features.append({
            "text": row["text"],
            "page": row["page_num"],
            "bbox": [0, 0, 0, 0],  # simplified
            "features": features
        })
    
    return blocks_with_features

if __name__ == "__main__":
    pdf_file = "data/labeled_blocks_sample.pdf"
    df = extract_features(pdf_file)
    print(f"Extracted {len(df)} blocks with enhanced multilingual features.")
    print(f"Script types found: {df['script_name'].value_counts().to_dict()}")
    print(f"Languages detected: {set(df['script_name'].values)}")
    print(f"Source types: {df['source'].value_counts().to_dict()}")
    print(df.head()) 