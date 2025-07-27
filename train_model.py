import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import json
import os
import time

# Paths
CSV_PATH = "data/labeled_blocks.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

print("🔄 Loading enhanced multilingual dataset...")
start_time = time.time()

# Load labeled data
df = pd.read_csv(CSV_PATH)

# Drop rows with missing labels
df = df[df['label'].notnull() & (df['label'] != '')]

print(f"✅ Loaded {len(df)} samples in {time.time() - start_time:.2f}s")

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Save reverse label map for predictions
reverse_label_map = {int(k): v for k, v in dict(zip(le.transform(le.classes_), le.classes_)).items()}

with open(LABEL_MAP_PATH, "w") as f:
    json.dump(reverse_label_map, f)

print(f"✅ Saved label map to {LABEL_MAP_PATH}")

# Define enhanced feature columns for improved accuracy
features = [
    'font_size', 'font_size_ratio', 'bold', 'alignment', 
    'spacing_above', 'spacing_below', 'line_spacing',
    'num_words', 'avg_word_length', 'caps_ratio', 'script_type'
]

# Check if all features exist in CSV, if not use available ones
available_features = [f for f in features if f in df.columns]
if not available_features:
    # Fallback to old feature names
    available_features = [
        'font_size', 'is_bold', 'is_centered', 'word_count', 
        'caps_ratio', 'spacing_above', 'position_pct'
    ]
    available_features = [f for f in available_features if f in df.columns]

print(f"Using enhanced features: {available_features}")

# Show script types if available
if 'script_type' in df.columns:
    print(f"Script types found: {df['script_type'].value_counts().to_dict()}")
elif 'script_name' in df.columns:
    print(f"Script names found: {df['script_name'].value_counts().to_dict()}")

# Show source distribution if available
if 'source' in df.columns:
    print(f"Source distribution: {df['source'].value_counts().to_dict()}")

# Features and target
X = df[available_features]
y = df['label_encoded']

print(f"📊 Dataset shape: {X.shape}")
print(f"📊 Classes: {list(le.classes_)}")

# Train/test split with stratification (if possible)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("✅ Used stratified split")
except ValueError as e:
    print(f"⚠️ Stratified split failed: {e}")
    print("📊 Using regular split instead")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"📊 Training samples: {len(X_train)}")
print(f"📊 Test samples: {len(X_test)}")

# Train enhanced classifier with optimized parameters
print("🔄 Training enhanced RandomForest classifier...")
train_start = time.time()

# Calculate class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

clf = RandomForestClassifier(
    n_estimators=150,  # Increased for better accuracy
    max_depth=12,      # Slightly deeper
    random_state=42, 
    n_jobs=-1,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight=class_weight_dict,  # Handle class imbalance
    bootstrap=True,
    oob_score=True     # Out-of-bag scoring
)
clf.fit(X_train, y_train)

train_time = time.time() - train_start
print(f"✅ Training completed in {train_time:.2f}s")

# Evaluate model
print("🔄 Evaluating enhanced model...")
y_pred = clf.predict(X_test)

# Get unique classes that actually appear in test data
unique_test_classes = sorted(set(y_test))
test_target_names = [le.classes_[i] for i in unique_test_classes]

print("\n=== Enhanced Classification Report ===")
print(classification_report(y_test, y_pred, target_names=test_target_names))

# Calculate accuracy metrics
accuracy = (y_pred == y_test).mean()
oob_score = clf.oob_score_ if hasattr(clf, 'oob_score_') else None

print(f"📊 Overall Accuracy: {accuracy:.4f}")
if oob_score:
    print(f"📊 Out-of-bag Score: {oob_score:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Feature Importance ===")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save model with compression
print("🔄 Saving enhanced model...")
save_start = time.time()
dump(clf, MODEL_PATH, compress=3)
save_time = time.time() - save_start

# Check model size
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"✅ Model saved to {MODEL_PATH}")
print(f"📊 Model size: {model_size_mb:.2f} MB")
print(f"📊 Save time: {save_time:.2f}s")

# Performance summary
print("\n=== Enhanced Performance Summary ===")
print(f"📊 Training time: {train_time:.2f}s")
print(f"📊 Model size: {model_size_mb:.2f} MB")
print(f"📊 Accuracy: {accuracy:.4f}")
if oob_score:
    print(f"📊 OOB Score: {oob_score:.4f}")
print(f"📊 Features used: {len(available_features)}")
print(f"📊 Classes: {len(le.classes_)}")

# Check constraints
if model_size_mb < 200:
    print("✅ Model size is within 200MB limit")
else:
    print("⚠️ Model size exceeds 200MB limit")

if model_size_mb < 10:
    print("✅ Model size is excellent (< 10MB)")
else:
    print("⚠️ Model size is larger than ideal")

if accuracy > 0.85:
    print("✅ Accuracy meets goal (> 0.85)")
else:
    print("⚠️ Accuracy below goal (< 0.85)")

print(f"✅ Multilingual support: {len(set(df.get('script_type', [0])))} script types")

# Show OCR support if available
if 'source' in df.columns:
    ocr_samples = df[df['source'] == 'ocr'].shape[0]
    print(f"✅ OCR support: {ocr_samples} OCR-processed samples")

print("\n=== Model Configuration ===")
print(f"📊 Estimators: {clf.n_estimators}")
print(f"📊 Max Depth: {clf.max_depth}")
print(f"📊 Class Weights: {class_weight_dict}")
print(f"📊 OOB Score: {clf.oob_score_ if hasattr(clf, 'oob_score_') else 'N/A'}") 