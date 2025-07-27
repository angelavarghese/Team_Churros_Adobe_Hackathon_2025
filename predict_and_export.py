# predict_and_export.py

import os
import json
import joblib
import fitz  # PyMuPDF
import pandas as pd
import time
from collections import defaultdict

from extract_features import extract_features  # Updated to use enhanced function

MODEL_PATH = "model/model.joblib"  # Updated path
LABEL_MAP_PATH = "model/label_map.json"
PDF_PATH = "data/labeled_blocks_sample.pdf"         # 🔁 Set your input PDF path here
OUTPUT_PATH = "output/predictions.json"    # 🔁 Output JSON path

print("🔄 Loading enhanced model and label map...")
start_time = time.time()

# Load model and label map
clf = joblib.load(MODEL_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

# Invert label_map to get numeric -> label
label_decoder = {int(k): v for k, v in label_map.items()}

print(f"✅ Enhanced model loaded in {time.time() - start_time:.2f}s")

# Load and extract features from PDF
print("🔍 Extracting enhanced features from PDF...")
extract_start = time.time()

df = extract_features(PDF_PATH)

extract_time = time.time() - extract_start
print(f"✅ Enhanced feature extraction completed in {extract_time:.2f}s")

# Define enhanced feature columns (must match training)
feature_cols = [
    "font_size", "font_size_ratio", "bold", "alignment", 
    "spacing_above", "spacing_below", "line_spacing",
    "num_words", "avg_word_length", "caps_ratio", "script_type"
]

# Check which features are available and match the model
available_features = [col for col in feature_cols if col in df.columns]

# Handle feature compatibility with trained model
if hasattr(clf, 'feature_names_in_'):
    # Model was trained with specific features, use only those
    model_features = list(clf.feature_names_in_)
    available_features = [col for col in available_features if col in model_features]
    print(f"Model expects features: {model_features}")
    print(f"Available features: {available_features}")
else:
    # Fallback to old feature names if needed
    if not available_features:
        available_features = [
            "font_size", "is_bold", "is_centered", "word_count", 
            "caps_ratio", "spacing_above", "position_pct"
        ]
        available_features = [col for col in available_features if col in df.columns]

print(f"Using enhanced features: {available_features}")

# Show script types found (for multilingual support)
if 'script_name' in df.columns:
    print(f"Script types found: {df['script_name'].value_counts().to_dict()}")
    print(f"Languages detected: {set(df['script_name'].values)}")

# Show OCR usage if available
if 'source' in df.columns:
    source_counts = df['source'].value_counts()
    print(f"Processing source: {source_counts.to_dict()}")
    if 'ocr' in source_counts:
        print(f"📄 OCR used for {source_counts['ocr']} blocks")

# Predict
print("🔄 Making enhanced predictions...")
predict_start = time.time()

predictions = clf.predict(df[available_features])
labels = [label_decoder[pred] for pred in predictions]

predict_time = time.time() - predict_start
print(f"✅ Enhanced predictions completed in {predict_time:.2f}s")

# Combine predictions with original data
df["predicted_label"] = labels

# Create structured output in the expected format
print("🔄 Creating enhanced structured output...")
output_start = time.time()

# Group by page and create outline
outline = []
for idx, row in df.iterrows():
    if row["predicted_label"] in ["H1", "H2", "H3", "Title"]:  # Only include headings
        # Ensure text is properly encoded
        text = row["text"]
        if isinstance(text, str):
            # Clean up any remaining encoding issues
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        outline.append({
            "level": row["predicted_label"],
            "text": text,
            "page": row["page_num"]
        })

# Create final output structure
structured_output = {
    "title": "Enhanced Multilingual Document",  # You can extract this from the first H1
    "outline": outline
}

output_time = time.time() - output_start
print(f"✅ Enhanced output structured in {output_time:.2f}s")

# Write to JSON with proper UTF-8 encoding
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(structured_output, f, indent=2, ensure_ascii=False)

# Performance summary
total_time = time.time() - start_time
print(f"✅ Enhanced predictions saved to {OUTPUT_PATH}")
print(f"📊 Total headings predicted: {len(outline)}")
print(f"📊 Total blocks processed: {len(df)}")
print(f"📊 Total execution time: {total_time:.2f}s")

# Enhanced performance breakdown
print("\n=== Enhanced Performance Breakdown ===")
print(f"📊 Model loading: {extract_start - start_time:.2f}s")
print(f"📊 Feature extraction: {extract_time:.2f}s")
print(f"📊 Prediction: {predict_time:.2f}s")
print(f"📊 Output structuring: {output_time:.2f}s")
print(f"📊 Total time: {total_time:.2f}s")

# Check if performance meets requirements
if total_time < 10:
    print("✅ Execution time is within 10s limit")
else:
    print("⚠️ Execution time exceeds 10s limit")

# Show sample of predictions
print("\n=== Enhanced Sample Predictions ===")
for i, item in enumerate(outline[:5]):
    print(f"{i+1}. [{item['level']}] {item['text']} (Page {item['page']})")

# Enhanced multilingual statistics
if 'script_name' in df.columns:
    print(f"\n=== Enhanced Multilingual Statistics ===")
    script_counts = df['script_name'].value_counts()
    print(f"Scripts detected: {len(script_counts)}")
    for script, count in script_counts.items():
        print(f"  {script}: {count} blocks")

# OCR statistics
if 'source' in df.columns:
    print(f"\n=== OCR Processing Statistics ===")
    source_counts = df['source'].value_counts()
    for source, count in source_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {source.upper()}: {count} blocks ({percentage:.1f}%)")

# Model performance metrics
if hasattr(clf, 'feature_importances_'):
    print(f"\n=== Model Feature Importance (Top 5) ===")
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

print(f"\n🎉 Enhanced multilingual PDF processing completed successfully!") 