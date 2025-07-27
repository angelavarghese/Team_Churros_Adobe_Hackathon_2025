import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import joblib
import pandas as pd
import time
from extract_features import extract_features
from flask import Flask, request, jsonify, render_template

# Load model and label map
MODEL_PATH = "model/model.joblib"  # Updated path
LABEL_MAP_PATH = "model/label_map.json"

try:
    clf = joblib.load(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    label_decoder = {int(k): v for k, v in label_map.items()}
    print("✅ Enhanced model loaded successfully")
except FileNotFoundError:
    print("⚠️ Model not found. Please train the model first.")
    clf = None
    label_decoder = {}

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    if clf is None:
        return jsonify({"error": "Model not trained. Please train the model first."}), 500

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        start_time = time.time()
        
        # Extract features with OCR fallback
        print(f"🔄 Processing {filename} with enhanced features...")
        df = extract_features(file_path)
        
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
        ocr_used = False
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            print(f"Processing source: {source_counts.to_dict()}")
            if 'ocr' in source_counts:
                ocr_used = True
                print(f"📄 OCR used for {source_counts['ocr']} blocks")

        # Predict
        predictions = clf.predict(df[available_features])
        labels = [label_decoder[pred] for pred in predictions]
        
        # Combine with text and create outline
        outline = []
        for idx, row in df.iterrows():
            if labels[idx] in ["H1", "H2", "H3", "Title"]:  # Only include headings
                # Ensure text is properly encoded
                text = row["text"]
                if isinstance(text, str):
                    # Clean up any remaining encoding issues
                    text = text.encode('utf-8', errors='ignore').decode('utf-8')
                
                outline.append({
                    "level": labels[idx],
                    "text": text,
                    "page": row.get("page_num", 1)
                })
        
        # Create result structure
        result = {
            "title": "Enhanced Multilingual Document",
            "outline": outline,
            "performance": {
                "total_blocks": len(df),
                "headings_found": len(outline),
                "processing_time": time.time() - start_time
            }
        }
        
        # Add enhanced multilingual statistics if available
        if 'script_name' in df.columns:
            script_counts = df['script_name'].value_counts()
            result["multilingual_stats"] = {
                "scripts_detected": len(script_counts),
                "script_distribution": script_counts.to_dict()
            }
        
        # Add OCR statistics if available
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            result["ocr_stats"] = {
                "ocr_used": ocr_used,
                "source_distribution": source_counts.to_dict(),
                "ocr_percentage": (source_counts.get('ocr', 0) / len(df)) * 100
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    app.run(debug=True, port=5000) 