"""
Phishing Detector - Real-time URL and Email Analysis
Main detection engine that combines all analysis components
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import warnings

from feature_extractor import URLFeatureExtractor
from email_analyzer import EmailPhishingAnalyzer

warnings.filterwarnings("ignore")


class PhishingDetector:
    """
    Main phishing detection system that combines URL analysis,
    email analysis, and ML predictions
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        feature_names_path: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the phishing detector

        Args:
            model_path: Path to trained model file
            scaler_path: Path to scaler file
            feature_names_path: Path to feature names JSON
            threshold: Classification threshold (0-1)
        """
        self.threshold = threshold
        self.model = None
        self.scaler = None
        self.feature_names = None

        # Initialize feature extractors
        self.url_extractor = URLFeatureExtractor()
        self.email_analyzer = EmailPhishingAnalyzer()

        # Load model if provided
        if model_path:
            self.load_model(model_path, scaler_path, feature_names_path)

    def load_model(
        self,
        model_path: str,
        scaler_path: Optional[str] = None,
        feature_names_path: Optional[str] = None,
    ):
        """
        Load trained model and preprocessing components

        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
            feature_names_path: Path to feature names JSON
        """
        try:
            # Load model
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")

            # Load scaler
            if scaler_path:
                self.scaler = joblib.load(scaler_path)
                print(f"Scaler loaded from {scaler_path}")

            # Load feature names
            if feature_names_path:
                with open(feature_names_path, "r") as f:
                    self.feature_names = json.load(f)
                print(f"Feature names loaded ({len(self.feature_names)} features)")

            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict_url(self, url: str) -> Dict[str, any]:
        """
        Predict if a URL is phishing

        Args:
            url: URL to analyze

        Returns:
            Dictionary containing prediction results
        """
        try:
            # Extract features
            features = self.url_extractor.extract_all_features(url)

            # Prepare for prediction
            if self.model and self.feature_names:
                # Create feature vector in correct order
                feature_vector = np.array(
                    [features.get(fname, 0.0) for fname in self.feature_names]
                ).reshape(1, -1)

                # Scale features
                if self.scaler:
                    feature_vector = self.scaler.transform(feature_vector)

                # Predict
                prediction = self.model.predict(feature_vector)[0]
                probability = (
                    self.model.predict_proba(feature_vector)[0]
                    if hasattr(self.model, "predict_proba")
                    else None
                )

                # Calculate risk score
                if probability is not None:
                    phishing_probability = probability[1]
                    risk_score = phishing_probability * 100
                else:
                    phishing_probability = 1.0 if prediction == 1 else 0.0
                    risk_score = phishing_probability * 100

            else:
                # Rule-based detection if no model loaded
                risk_score = self._calculate_rule_based_score(features)
                phishing_probability = risk_score / 100.0
                prediction = 1 if phishing_probability >= self.threshold else 0

            # Determine classification
            classification = self._classify_risk(risk_score)

            # Generate warnings
            warnings = self._generate_url_warnings(features)

            # Identify suspicious features
            suspicious_features = self._identify_suspicious_features(features)

            return {
                "url": url,
                "is_phishing": bool(prediction),
                "phishing_probability": float(phishing_probability),
                "risk_score": float(risk_score),
                "classification": classification,
                "warnings": warnings,
                "suspicious_features": suspicious_features,
                "features": features,
            }

        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "is_phishing": None,
                "risk_score": None,
            }

    def predict_email(self, email_content: bytes) -> Dict[str, any]:
        """
        Predict if an email is phishing

        Args:
            email_content: Raw email content as bytes

        Returns:
            Dictionary containing prediction results
        """
        try:
            # Analyze email
            email_analysis = self.email_analyzer.analyze_email(email_content)

            return {
                "is_phishing": email_analysis["classification"]
                in ["HIGH_RISK", "MEDIUM_RISK"],
                "risk_score": email_analysis["risk_score"],
                "classification": email_analysis["classification"],
                "warnings": email_analysis["warnings"],
                "features": email_analysis["features"],
            }

        except Exception as e:
            return {
                "error": str(e),
                "is_phishing": None,
                "risk_score": None,
            }

    def predict_batch(self, urls: List[str]) -> List[Dict[str, any]]:
        """
        Predict multiple URLs in batch

        Args:
            urls: List of URLs to analyze

        Returns:
            List of prediction results
        """
        results = []

        for url in urls:
            result = self.predict_url(url)
            results.append(result)

        return results

    def predict_from_features(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Make prediction from pre-extracted features

        Args:
            features: Dictionary of feature values

        Returns:
            Prediction results
        """
        if not self.model or not self.feature_names:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Create feature vector
            feature_vector = np.array(
                [features.get(fname, 0.0) for fname in self.feature_names]
            ).reshape(1, -1)

            # Scale features
            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)

            # Predict
            prediction = self.model.predict(feature_vector)[0]
            probability = (
                self.model.predict_proba(feature_vector)[0]
                if hasattr(self.model, "predict_proba")
                else None
            )

            if probability is not None:
                phishing_probability = probability[1]
                risk_score = phishing_probability * 100
            else:
                phishing_probability = 1.0 if prediction == 1 else 0.0
                risk_score = phishing_probability * 100

            classification = self._classify_risk(risk_score)

            return {
                "is_phishing": bool(prediction),
                "phishing_probability": float(phishing_probability),
                "risk_score": float(risk_score),
                "classification": classification,
            }

        except Exception as e:
            return {"error": str(e), "is_phishing": None}

    def _calculate_rule_based_score(self, features: Dict[str, float]) -> float:
        """
        Calculate risk score using rule-based heuristics
        Used when ML model is not available
        """
        score = 0.0
        weights = {
            "has_ip_address": 15.0,
            "has_suspicious_keyword": 10.0,
            "is_shortened": 8.0,
            "has_at": 12.0,
            "url_length": 0.05,  # per character over 75
            "num_dots": 5.0,  # per dot over 3
            "has_suspicious_tld": 12.0,
            "has_embedded_domain": 15.0,
            "has_https": -10.0,  # negative = good
            "has_valid_cert": -10.0,
            "domain_age_months": -0.5,  # per month (negative = good)
            "num_hyphens": 3.0,  # per hyphen over 1
            "num_underscores": 4.0,
            "num_params": 2.0,  # per parameter over 2
            "has_port": 8.0,
            "domain_has_digits": 7.0,
        }

        # Base score
        score = 50.0

        # Apply weighted features
        for feature, weight in weights.items():
            if feature in features:
                value = features[feature]

                if feature == "url_length":
                    if value > 75:
                        score += (value - 75) * weight
                elif feature == "num_dots":
                    if value > 3:
                        score += (value - 3) * weight
                elif feature == "num_hyphens":
                    if value > 1:
                        score += (value - 1) * weight
                elif feature == "num_params":
                    if value > 2:
                        score += (value - 2) * weight
                elif feature == "domain_age_months":
                    if value > 0:
                        score += value * weight
                else:
                    score += value * weight

        # Normalize to 0-100
        score = max(0.0, min(100.0, score))

        return score

    def _classify_risk(self, risk_score: float) -> str:
        """
        Classify risk level based on score

        Args:
            risk_score: Risk score (0-100)

        Returns:
            Risk classification string
        """
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "SAFE"

    def _generate_url_warnings(self, features: Dict[str, float]) -> List[str]:
        """Generate human-readable warnings from features"""
        warnings = []

        if features.get("has_ip_address", 0) > 0:
            warnings.append("⚠️ URL uses IP address instead of domain name")

        if features.get("has_suspicious_keyword", 0) > 0:
            warnings.append(
                "⚠️ URL contains suspicious keywords (login, verify, account, etc.)"
            )

        if features.get("is_shortened", 0) > 0:
            warnings.append("⚠️ URL uses a shortening service")

        if features.get("has_at", 0) > 0 and features["has_at"] > 0:
            warnings.append("⚠️ URL contains '@' symbol (possible obfuscation)")

        if features.get("url_length", 0) > 100:
            warnings.append("⚠️ URL is unusually long")

        if features.get("has_suspicious_tld", 0) > 0:
            warnings.append("⚠️ URL uses suspicious top-level domain")

        if features.get("has_embedded_domain", 0) > 0:
            warnings.append("⚠️ URL appears to have embedded domain name")

        if features.get("has_https", 0) == 0:
            warnings.append("⚠️ URL does not use HTTPS")

        if features.get("has_valid_cert", 0) == 0 and features.get("has_https", 0) > 0:
            warnings.append("⚠️ Invalid or missing SSL certificate")

        if (
            features.get("domain_age_months", -1) >= 0
            and features["domain_age_months"] < 6
        ):
            warnings.append("⚠️ Domain is very new (less than 6 months old)")

        if features.get("num_subdomain_parts", 0) > 3:
            warnings.append("⚠️ URL has many subdomains")

        if features.get("domain_has_digits", 0) > 0:
            warnings.append("⚠️ Domain name contains digits")

        if features.get("is_known_brand", 0) == 0.5:
            warnings.append("🚨 Domain contains brand name but is not official domain")

        if features.get("has_port", 0) > 0:
            warnings.append("⚠️ URL specifies non-standard port")

        if features.get("prefix_suffix_dash", 0) > 0:
            warnings.append("⚠️ Domain has prefix or suffix with dash")

        return warnings

    def _identify_suspicious_features(
        self, features: Dict[str, float]
    ) -> List[Dict[str, any]]:
        """Identify and rank suspicious features"""
        suspicious = []

        # Define thresholds
        suspicion_rules = [
            ("has_ip_address", ">", 0, "high"),
            ("has_at", ">", 0, "high"),
            ("has_embedded_domain", ">", 0, "high"),
            ("is_known_brand", "==", 0.5, "critical"),
            ("has_suspicious_tld", ">", 0, "medium"),
            ("is_shortened", ">", 0, "medium"),
            ("has_suspicious_keyword", ">", 0, "medium"),
            ("url_length", ">", 100, "medium"),
            ("num_dots", ">", 4, "low"),
            ("num_hyphens", ">", 2, "low"),
            ("has_https", "==", 0, "medium"),
            ("has_valid_cert", "==", 0, "high"),
            ("domain_age_months", "<", 3, "high"),
            ("domain_has_digits", ">", 0, "low"),
        ]

        for feature_name, operator, threshold, severity in suspicion_rules:
            if feature_name in features:
                value = features[feature_name]
                is_suspicious = False

                if operator == ">" and value > threshold:
                    is_suspicious = True
                elif operator == "<" and value < threshold and value >= 0:
                    is_suspicious = True
                elif operator == "==" and value == threshold:
                    is_suspicious = True

                if is_suspicious:
                    suspicious.append(
                        {"feature": feature_name, "value": value, "severity": severity}
                    )

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suspicious.sort(key=lambda x: severity_order[x["severity"]])

        return suspicious

    def analyze_url_comprehensive(self, url: str) -> Dict[str, any]:
        """
        Comprehensive URL analysis with detailed report

        Args:
            url: URL to analyze

        Returns:
            Detailed analysis report
        """
        # Get prediction
        prediction = self.predict_url(url)

        # Additional analysis
        from urllib.parse import urlparse

        parsed = urlparse(url)

        report = {
            "url": url,
            "prediction": prediction,
            "url_components": {
                "scheme": parsed.scheme,
                "netloc": parsed.netloc,
                "path": parsed.path,
                "params": parsed.params,
                "query": parsed.query,
                "fragment": parsed.fragment,
            },
            "risk_analysis": {
                "overall_risk": prediction.get("classification", "UNKNOWN"),
                "risk_score": prediction.get("risk_score", 0),
                "phishing_probability": prediction.get("phishing_probability", 0),
            },
            "warnings": prediction.get("warnings", []),
            "suspicious_features": prediction.get("suspicious_features", []),
            "recommendations": self._generate_recommendations(prediction),
        }

        return report

    def _generate_recommendations(self, prediction: Dict[str, any]) -> List[str]:
        """Generate security recommendations based on prediction"""
        recommendations = []

        risk_score = prediction.get("risk_score", 0)

        if risk_score >= 60:
            recommendations.append(
                "🚫 DO NOT visit this URL or enter any personal information"
            )
            recommendations.append("🚫 DO NOT download any files from this website")
            recommendations.append("📧 Report this URL to your IT security team")

        elif risk_score >= 40:
            recommendations.append("⚠️ Exercise extreme caution with this URL")
            recommendations.append("🔍 Verify the URL matches the official website")
            recommendations.append("🔐 Do not enter passwords or sensitive data")

        elif risk_score >= 20:
            recommendations.append("⚠️ Be cautious and verify the legitimacy")
            recommendations.append("🔍 Check for HTTPS and valid SSL certificate")

        else:
            recommendations.append(
                "✅ URL appears legitimate, but always stay vigilant"
            )
            recommendations.append(
                "🔐 Ensure HTTPS is enabled before entering sensitive data"
            )

        # Add specific recommendations
        features = prediction.get("features", {})

        if features.get("has_https", 0) == 0:
            recommendations.append("🔒 Avoid entering sensitive information (no HTTPS)")

        if (
            features.get("domain_age_months", -1) >= 0
            and features["domain_age_months"] < 6
        ):
            recommendations.append(
                "🆕 Domain is very new - verify legitimacy before trusting"
            )

        return recommendations

    def get_model_info(self) -> Dict[str, any]:
        """Get information about loaded model"""
        if not self.model:
            return {"loaded": False, "message": "No model loaded"}

        info = {
            "loaded": True,
            "model_type": type(self.model).__name__,
            "num_features": len(self.feature_names) if self.feature_names else None,
            "threshold": self.threshold,
            "has_scaler": self.scaler is not None,
            "features": self.feature_names,
        }

        return info


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("Phishing Detector - Real-time Analysis")
    print("=" * 70)

    # Initialize detector
    detector = PhishingDetector()

    # Test URLs
    test_urls = [
        "https://www.google.com",
        "http://paypa1-secure-login.tk/verify.php?id=12345",
        "https://192.168.1.1/admin/login",
        "http://bit.ly/suspicious",
        "https://secure-banking-update.com/account/verify",
    ]

    print("\nTesting URLs (Rule-based detection):")
    print("-" * 70)

    for url in test_urls:
        print(f"\nURL: {url}")
        result = detector.predict_url(url)

        print(f"  Is Phishing: {result['is_phishing']}")
        print(f"  Risk Score: {result['risk_score']:.2f}/100")
        print(f"  Classification: {result['classification']}")

        if result.get("warnings"):
            print(f"  Warnings:")
            for warning in result["warnings"][:3]:
                print(f"    - {warning}")

    print("\n" + "=" * 70)
    print("Detection complete!")
