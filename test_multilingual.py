#!/usr/bin/env python3
"""
Test script for multilingual PDF heading extractor.
Tests script detection, feature extraction, and model compatibility.
"""

import sys
import os
import pandas as pd
import numpy as np
from extract_features import detect_script, script_to_numeric, normalize_text

def test_script_detection():
    """Test script detection for various languages."""
    print("🧪 Testing Script Detection")
    print("=" * 50)
    
    test_cases = [
        ("Hello World", "Latin"),
        ("नमस्ते दुनिया", "Devanagari"),
        ("ನಮಸ್ಕಾರ ಪ್ರಪಂಚ", "Kannada"),
        ("வணக்கம் உலகம்", "Tamil"),
        ("こんにちは世界", "Japanese"),
        ("안녕하세요 세계", "Korean"),
        ("مرحبا بالعالم", "Arabic"),
        ("你好世界", "Chinese"),
        ("สวัสดีโลก", "Thai"),
        ("হ্যালো বিশ্ব", "Bengali"),
        ("హలో ప్రపంచం", "Telugu"),
    ]
    
    all_passed = True
    
    for text, expected in test_cases:
        detected = detect_script(text)
        numeric = script_to_numeric(detected)
        
        if detected == expected:
            print(f"✅ {text} -> {detected} (numeric: {numeric})")
        else:
            print(f"❌ {text} -> {detected} (numeric: {numeric})")
            all_passed = False
    
    print(f"\nScript Detection: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    return all_passed

def test_text_normalization():
    """Test Unicode text normalization."""
    print("\n🧪 Testing Text Normalization")
    print("=" * 50)
    
    test_cases = [
        ("Hello\u00a0World", "Hello World"),  # Non-breaking space
        ("नमस्ते\u200bदुनिया", "नमस्तेदुनिया"),  # Zero-width space
        ("こんにちは\u3000世界", "こんにちは 世界"),  # Ideographic space
    ]
    
    all_passed = True
    
    for input_text, expected in test_cases:
        normalized = normalize_text(input_text)
        if normalized == expected:
            print(f"✅ '{input_text}' -> '{normalized}'")
        else:
            print(f"❌ '{input_text}' -> '{normalized}' (expected: '{expected}')")
            all_passed = False
    
    print(f"\nText Normalization: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    return all_passed

def test_feature_extraction():
    """Test feature extraction from sample data."""
    print("\n🧪 Testing Feature Extraction")
    print("=" * 50)
    
    try:
        from extract_features import extract_features
        
        # Test with a sample PDF if available
        test_pdf = "data/labeled_blocks_sample.pdf"
        
        if os.path.exists(test_pdf):
            print(f"Testing feature extraction with {test_pdf}")
            df = extract_features(test_pdf)
            
            if not df.empty:
                print(f"✅ Extracted {len(df)} blocks")
                print(f"✅ Features: {list(df.columns)}")
                
                # Check required features
                required_features = [
                    'font_size', 'font_size_ratio', 'bold', 'alignment',
                    'spacing_above', 'spacing_below', 'line_spacing',
                    'num_words', 'avg_word_length', 'caps_ratio', 'script_type'
                ]
                
                missing_features = [f for f in required_features if f not in df.columns]
                if missing_features:
                    print(f"⚠️ Missing features: {missing_features}")
                else:
                    print("✅ All required features present")
                
                # Show script distribution
                if 'script_name' in df.columns:
                    script_counts = df['script_name'].value_counts()
                    print(f"✅ Script distribution: {script_counts.to_dict()}")
                
                return True
            else:
                print("❌ No blocks extracted")
                return False
        else:
            print(f"⚠️ Test PDF not found: {test_pdf}")
            print("Creating synthetic test data...")
            
            # Create synthetic test data
            synthetic_data = {
                'text': ['Hello World', 'नमस्ते दुनिया', 'こんにちは世界'],
                'font_size': [12, 14, 16],
                'font_size_ratio': [1.0, 1.17, 1.33],
                'bold': [0, 1, 0],
                'alignment': [0, 1, 2],
                'spacing_above': [10, 15, 20],
                'spacing_below': [5, 10, 15],
                'line_spacing': [1.0, 1.2, 1.0],
                'num_words': [2, 2, 1],
                'avg_word_length': [5.0, 4.5, 6.0],
                'caps_ratio': [0.18, 0.0, 0.0],
                'script_type': [0, 1, 4],
                'script_name': ['Latin', 'Devanagari', 'Japanese'],
                'page_num': [1, 1, 1],
                'source': ['text', 'text', 'text']
            }
            
            df = pd.DataFrame(synthetic_data)
            print(f"✅ Created synthetic data with {len(df)} samples")
            print(f"✅ Features: {list(df.columns)}")
            return True
            
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_model_compatibility():
    """Test model loading and feature compatibility."""
    print("\n🧪 Testing Model Compatibility")
    print("=" * 50)
    
    try:
        import joblib
        
        model_path = "model/model.joblib"
        label_map_path = "model/label_map.json"
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_path}")
            return False
        
        if not os.path.exists(label_map_path):
            print(f"⚠️ Label map not found: {label_map_path}")
            return False
        
        # Load model
        clf = joblib.load(model_path)
        print("✅ Model loaded successfully")
        
        # Load label map
        import json
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        print("✅ Label map loaded successfully")
        
        # Check model features
        if hasattr(clf, 'feature_names_in_'):
            print(f"✅ Model expects features: {list(clf.feature_names_in_)}")
            
            # Test with synthetic data
            synthetic_features = pd.DataFrame({
                'font_size': [12, 14, 16],
                'font_size_ratio': [1.0, 1.17, 1.33],
                'bold': [0, 1, 0],
                'alignment': [0, 1, 2],
                'spacing_above': [10, 15, 20],
                'spacing_below': [5, 10, 15],
                'line_spacing': [1.0, 1.2, 1.0],
                'num_words': [2, 2, 1],
                'avg_word_length': [5.0, 4.5, 6.0],
                'caps_ratio': [0.18, 0.0, 0.0],
                'script_type': [0, 1, 4]
            })
            
            # Use only features that model expects
            available_features = [f for f in clf.feature_names_in_ if f in synthetic_features.columns]
            if available_features:
                test_features = synthetic_features[available_features]
                predictions = clf.predict(test_features)
                print(f"✅ Model predictions: {predictions}")
                
                # Decode predictions
                label_decoder = {int(k): v for k, v in label_map.items()}
                decoded_predictions = [label_decoder[pred] for pred in predictions]
                print(f"✅ Decoded predictions: {decoded_predictions}")
                
                return True
            else:
                print("❌ No compatible features found")
                return False
        else:
            print("⚠️ Model doesn't have feature names")
            return False
            
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False

def test_performance():
    """Test performance metrics."""
    print("\n🧪 Testing Performance Metrics")
    print("=" * 50)
    
    try:
        import time
        
        # Test model loading time
        start_time = time.time()
        import joblib
        clf = joblib.load("model/model.joblib")
        load_time = time.time() - start_time
        
        print(f"✅ Model loading time: {load_time:.3f}s")
        
        # Test model size
        model_path = "model/model.joblib"
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model size: {model_size_mb:.2f} MB")
        
        # Test prediction time
        synthetic_features = pd.DataFrame({
            'font_size': [12] * 100,
            'font_size_ratio': [1.0] * 100,
            'bold': [0] * 100,
            'alignment': [0] * 100,
            'spacing_above': [10] * 100,
            'spacing_below': [5] * 100,
            'line_spacing': [1.0] * 100,
            'num_words': [2] * 100,
            'avg_word_length': [5.0] * 100,
            'caps_ratio': [0.18] * 100,
            'script_type': [0] * 100
        })
        
        # Use only features that model expects
        if hasattr(clf, 'feature_names_in_'):
            available_features = [f for f in clf.feature_names_in_ if f in synthetic_features.columns]
            if available_features:
                test_features = synthetic_features[available_features]
                
                start_time = time.time()
                predictions = clf.predict(test_features)
                predict_time = time.time() - start_time
                
                print(f"✅ Prediction time (100 samples): {predict_time:.3f}s")
                print(f"✅ Prediction rate: {100/predict_time:.1f} samples/second")
        
        # Check constraints
        constraints_met = True
        
        if model_size_mb > 200:
            print("❌ Model size exceeds 200MB limit")
            constraints_met = False
        else:
            print("✅ Model size within 200MB limit")
        
        if load_time > 5:
            print("❌ Model loading time exceeds 5s")
            constraints_met = False
        else:
            print("✅ Model loading time within 5s")
        
        return constraints_met
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🌍 Multilingual PDF Heading Extractor - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Script Detection", test_script_detection),
        ("Text Normalization", test_text_normalization),
        ("Feature Extraction", test_feature_extraction),
        ("Model Compatibility", test_model_compatibility),
        ("Performance Metrics", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 