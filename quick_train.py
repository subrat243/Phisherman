#!/usr/bin/env python3
"""
Quick ML Model Training Script
Trains a lightweight ML model with sample data for immediate use
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from feature_extractor import URLFeatureExtractor

print("=" * 70)
print("🤖 Quick ML Model Training")
print("=" * 70)
print("\nThis script will create a trained ML model using sample data")
print("The model will be ready to use immediately!\n")

# Sample training data - known phishing and legitimate URLs
SAMPLE_URLS = {
    # Legitimate URLs (label = 0)
    "legitimate": [
        "https://www.google.com",
        "https://www.github.com",
        "https://www.microsoft.com",
        "https://www.amazon.com",
        "https://www.facebook.com",
        "https://www.twitter.com",
        "https://www.linkedin.com",
        "https://www.apple.com",
        "https://www.youtube.com",
        "https://www.netflix.com",
        "https://www.wikipedia.org",
        "https://www.reddit.com",
        "https://www.stackoverflow.com",
        "https://www.cnn.com",
        "https://www.bbc.com",
        "https://www.paypal.com",
        "https://www.chase.com",
        "https://www.wellsfargo.com",
        "https://www.bankofamerica.com",
        "https://www.citibank.com",
        "https://accounts.google.com/login",
        "https://login.microsoftonline.com",
        "https://www.linkedin.com/login",
        "https://signin.aws.amazon.com",
        "https://login.live.com",
    ],
    # Phishing URLs (label = 1)
    "phishing": [
        "http://paypal-verify-account.tk/login.php",
        "http://secure-login-paypal.ml/verify.php",
        "http://amazon-account-locked.ga/confirm.php",
        "http://apple-id-unlock.cf/verify.php",
        "http://microsoft-account-suspend.gq/login.php",
        "http://192.168.1.1/banking/login.php",
        "http://secure-facebook-login.tk/verify.html",
        "http://netflix-billing-update.ml/payment.php",
        "http://account-verification-needed.ga/confirm.asp",
        "http://urgent-security-alert.cf/verify.php",
        "http://bit.ly/suspicious-link",
        "http://paypa1.com/login",
        "http://g00gle.com/verify",
        "http://micros0ft.com/account",
        "http://bank-0f-america.com/login",
        "http://paypal.com-verify.suspicious.tk/login",
        "http://secure-payment-paypal.ml/confirm",
        "http://amazon.com.phishing.ga/verify",
        "http://apple-account-locked.cf/unlock.php",
        "http://facebook.com-security.tk/login.html",
        "http://twitter.com.verify.ml/account.php",
        "http://instagram-copyright.ga/appeal.php",
        "http://netflix.billing-problem.cf/update.php",
        "http://chase-bank-alert.tk/verify.asp",
        "http://wellsfargo-suspicious-activity.ml/confirm.php",
    ],
}


def create_sample_dataset():
    """Create sample dataset from URLs"""
    print("Step 1: Creating sample dataset...")

    extractor = URLFeatureExtractor()

    features_list = []
    labels = []

    # Process legitimate URLs
    print(f"  - Processing {len(SAMPLE_URLS['legitimate'])} legitimate URLs...")
    for url in SAMPLE_URLS["legitimate"]:
        try:
            features = extractor.extract_all_features(url)
            features_list.append(features)
            labels.append(0)  # Legitimate
        except Exception as e:
            print(f"    Warning: Failed to extract from {url}: {e}")

    # Process phishing URLs
    print(f"  - Processing {len(SAMPLE_URLS['phishing'])} phishing URLs...")
    for url in SAMPLE_URLS["phishing"]:
        try:
            features = extractor.extract_all_features(url)
            features_list.append(features)
            labels.append(1)  # Phishing
        except Exception as e:
            print(f"    Warning: Failed to extract from {url}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    df["is_phishing"] = labels

    # Fill missing values
    df = df.fillna(-1)

    print(f"✓ Created dataset with {len(df)} samples")
    print(f"  - Legitimate: {sum(df['is_phishing'] == 0)}")
    print(f"  - Phishing: {sum(df['is_phishing'] == 1)}")

    return df


def train_model(df):
    """Train Random Forest model"""
    print("\nStep 2: Training Random Forest model...")

    # Separate features and labels
    X = df.drop(columns=["is_phishing"])
    y = df["is_phishing"]

    feature_names = list(X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  - Training set: {len(X_train)} samples")
    print(f"  - Test set: {len(X_test)} samples")
    print(f"  - Features: {len(feature_names)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("  - Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    print("\nStep 3: Evaluating model...")
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Model Performance:")
    print(f"  - Accuracy: {accuracy:.2%}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0][0]}  FP: {cm[0][1]}")
    print(f"    FN: {cm[1][0]}  TP: {cm[1][1]}")

    print(f"\n  Classification Report:")
    print(
        classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"])
    )

    # Feature importance
    importances = model.feature_importances_
    top_features = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )[:10]

    print(f"\n  Top 10 Important Features:")
    for feat, imp in top_features:
        print(f"    {feat:30s}: {imp:.4f}")

    return model, scaler, feature_names


def save_model(model, scaler, feature_names):
    """Save trained model"""
    print("\nStep 4: Saving model...")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Save model
    model_path = models_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved to {model_path}")

    # Save scaler
    scaler_path = models_dir / "model_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Scaler saved to {scaler_path}")

    # Save feature names
    features_path = models_dir / "model_features.json"
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  ✓ Feature names saved to {features_path}")

    return model_path, scaler_path, features_path


def test_model(model, scaler, feature_names):
    """Test model with sample URLs"""
    print("\nStep 5: Testing model...")

    extractor = URLFeatureExtractor()

    test_urls = [
        ("https://www.google.com", "Legitimate"),
        ("http://paypal-verify.tk/login", "Phishing"),
        ("https://www.github.com", "Legitimate"),
        ("http://192.168.1.1/bank/login", "Phishing"),
    ]

    print("\n  Test Results:")
    for url, expected in test_urls:
        try:
            features = extractor.extract_all_features(url)
            feature_vector = np.array(
                [features.get(f, 0) for f in feature_names]
            ).reshape(1, -1)
            feature_vector_scaled = scaler.transform(feature_vector)

            prediction = model.predict(feature_vector_scaled)[0]
            probability = model.predict_proba(feature_vector_scaled)[0]

            result = "Phishing" if prediction == 1 else "Legitimate"
            confidence = probability[prediction] * 100
            status = "✓" if result == expected else "✗"

            print(f"    {status} {url[:50]:50s} -> {result:12s} ({confidence:.1f}%)")
        except Exception as e:
            print(f"    ✗ {url[:50]:50s} -> Error: {e}")


def main():
    try:
        # Create dataset
        df = create_sample_dataset()

        # Train model
        model, scaler, feature_names = train_model(df)

        # Save model
        model_path, scaler_path, features_path = save_model(
            model, scaler, feature_names
        )

        # Test model
        test_model(model, scaler, feature_names)

        print("\n" + "=" * 70)
        print("✅ ML MODEL TRAINING COMPLETE!")
        print("=" * 70)

        print(f"\nModel files saved:")
        print(f"  - {model_path}")
        print(f"  - {scaler_path}")
        print(f"  - {features_path}")

        print(f"\nTo use the ML model:")
        print(f"  python3 detect.py -u https://example.com -m {model_path}")

        print(f"\nOr use it in your code:")
        print(f"  from src.phishing_detector import PhishingDetector")
        print(f"  detector = PhishingDetector(")
        print(f"      model_path='{model_path}',")
        print(f"      scaler_path='{scaler_path}',")
        print(f"      feature_names_path='{features_path}'")
        print(f"  )")
        print(f"  result = detector.predict_url('https://example.com')")

        print("\n🎉 Your ML-powered phishing detector is ready!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
