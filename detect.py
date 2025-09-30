#!/usr/bin/env python3
"""
CLI Tool for Phishing Detection
Simple command-line interface for analyzing URLs and emails
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from phishing_detector import PhishingDetector
from feature_extractor import URLFeatureExtractor


class PhishingCLI:
    """Command-line interface for phishing detection"""

    def __init__(self):
        self.detector = None
        self.feature_extractor = URLFeatureExtractor()

    def load_model(
        self, model_path: str, scaler_path: str = None, features_path: str = None
    ):
        """Load the ML model"""
        print("Loading model...")
        self.detector = PhishingDetector(
            model_path=model_path,
            scaler_path=scaler_path,
            feature_names_path=features_path,
        )
        print("✓ Model loaded successfully\n")

    def analyze_url(self, url: str, verbose: bool = False):
        """Analyze a single URL"""
        print(f"\n{'=' * 70}")
        print(f"Analyzing URL: {url}")
        print(f"{'=' * 70}\n")

        start_time = time.time()

        if self.detector:
            result = self.detector.predict_url(url)
        else:
            # Use rule-based detection
            print("⚠️  No model loaded - using rule-based detection")
            detector = PhishingDetector()
            result = detector.predict_url(url)

        elapsed_time = time.time() - start_time

        # Display results
        self._display_result(result, verbose)
        print(f"\n⏱️  Analysis completed in {elapsed_time:.3f} seconds")

    def analyze_batch(self, file_path: str, output: Optional[str] = None):
        """Analyze multiple URLs from a file"""
        print(f"\n{'=' * 70}")
        print(f"Batch Analysis")
        print(f"{'=' * 70}\n")

        # Read URLs from file
        try:
            with open(file_path, "r") as f:
                urls = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"❌ Error: File '{file_path}' not found")
            return

        print(f"Found {len(urls)} URLs to analyze\n")

        if not self.detector:
            print("⚠️  No model loaded - using rule-based detection")
            self.detector = PhishingDetector()

        # Analyze all URLs
        results = []
        phishing_count = 0
        safe_count = 0

        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] Analyzing: {url[:60]}...")
            result = self.detector.predict_url(url)
            results.append(result)

            if result.get("is_phishing"):
                phishing_count += 1
            else:
                safe_count += 1

        # Display summary
        print(f"\n{'=' * 70}")
        print("BATCH ANALYSIS SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total URLs analyzed: {len(urls)}")
        print(
            f"Phishing detected: {phishing_count} ({phishing_count / len(urls) * 100:.1f}%)"
        )
        print(f"Safe URLs: {safe_count} ({safe_count / len(urls) * 100:.1f}%)")
        print(f"{'=' * 70}\n")

        # Save results if output file specified
        if output:
            self._save_results(results, output)

    def interactive_mode(self):
        """Interactive mode for continuous URL checking"""
        print("\n" + "=" * 70)
        print("INTERACTIVE PHISHING DETECTION")
        print("=" * 70)
        print("Enter URLs to analyze (type 'quit' or 'exit' to stop)")
        print("=" * 70 + "\n")

        if not self.detector:
            print("⚠️  No model loaded - using rule-based detection\n")
            self.detector = PhishingDetector()

        while True:
            try:
                url = input("Enter URL: ").strip()

                if url.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                if not url:
                    continue

                if not url.startswith(("http://", "https://")):
                    url = "http://" + url

                result = self.detector.predict_url(url)
                self._display_result(result, verbose=False)
                print()

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

    def check_features(self, url: str):
        """Display extracted features from a URL"""
        print(f"\n{'=' * 70}")
        print(f"Feature Extraction: {url}")
        print(f"{'=' * 70}\n")

        features = self.feature_extractor.extract_all_features(url)

        print(f"Extracted {len(features)} features:\n")

        # Group features by category
        categories = {
            "URL Structure": [
                "url_length",
                "num_dots",
                "num_hyphens",
                "num_slashes",
                "num_question_marks",
                "num_params",
                "url_entropy",
            ],
            "Domain": [
                "domain_length",
                "subdomain_length",
                "num_subdomain_parts",
                "domain_has_digits",
                "domain_entropy",
            ],
            "Security": [
                "has_https",
                "has_valid_cert",
                "has_ip_address",
                "cert_age_days",
                "has_port",
            ],
            "Suspicious Indicators": [
                "has_suspicious_keyword",
                "is_shortened",
                "has_suspicious_tld",
                "has_embedded_domain",
            ],
            "Domain Info": [
                "domain_age_months",
                "days_until_expiration",
                "has_registrar",
            ],
        }

        for category, feature_names in categories.items():
            print(f"{category}:")
            for fname in feature_names:
                if fname in features:
                    value = features[fname]
                    print(f"  {fname:30s}: {value}")
            print()

    def _display_result(self, result: dict, verbose: bool = False):
        """Display analysis result in a formatted way"""
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            return

        # Status indicator
        is_phishing = result.get("is_phishing", False)
        risk_score = result.get("risk_score", 0)
        classification = result.get("classification", "UNKNOWN")

        if is_phishing or risk_score >= 60:
            status_icon = "🚨"
            status_text = "PHISHING DETECTED"
            color = "\033[91m"  # Red
        elif risk_score >= 40:
            status_icon = "⚠️"
            status_text = "SUSPICIOUS"
            color = "\033[93m"  # Yellow
        else:
            status_icon = "✅"
            status_text = "SAFE"
            color = "\033[92m"  # Green

        reset_color = "\033[0m"

        print(f"{color}{status_icon} {status_text}{reset_color}")
        print(f"\nRisk Score: {risk_score:.1f}/100")
        print(f"Classification: {classification}")

        if "phishing_probability" in result:
            prob = result["phishing_probability"] * 100
            print(f"Confidence: {prob:.1f}%")

        # Risk meter
        self._display_risk_meter(risk_score)

        # Warnings
        warnings = result.get("warnings", [])
        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for i, warning in enumerate(warnings[:10], 1):
                print(f"  {i}. {warning}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more")

        # Suspicious features
        if verbose and "suspicious_features" in result:
            suspicious = result["suspicious_features"]
            if suspicious:
                print(f"\n🔍 Suspicious Features:")
                for feature in suspicious[:5]:
                    print(
                        f"  • {feature['feature']}: {feature['value']} (severity: {feature['severity']})"
                    )

        # Recommendations
        if verbose and "warnings" in result and result["warnings"]:
            print(f"\n💡 Recommendations:")
            if risk_score >= 60:
                print("  • DO NOT visit this website or enter any information")
                print("  • DO NOT download any files from this site")
                print("  • Report this URL to your IT security team")
            elif risk_score >= 40:
                print("  • Exercise extreme caution")
                print("  • Verify the URL matches the official website")
                print("  • Do not enter passwords or sensitive information")
            else:
                print(
                    "  • Always verify HTTPS is enabled before entering sensitive data"
                )
                print("  • Stay vigilant for any suspicious behavior")

    def _display_risk_meter(self, risk_score: float):
        """Display a visual risk meter"""
        bar_length = 40
        filled_length = int(bar_length * risk_score / 100)

        if risk_score >= 80:
            color = "\033[91m"  # Red
            char = "█"
        elif risk_score >= 60:
            color = "\033[91m"  # Red
            char = "▓"
        elif risk_score >= 40:
            color = "\033[93m"  # Yellow
            char = "▒"
        else:
            color = "\033[92m"  # Green
            char = "░"

        reset_color = "\033[0m"

        bar = color + char * filled_length + reset_color
        bar += "·" * (bar_length - filled_length)

        print(f"\nRisk Meter: [{bar}] {risk_score:.0f}%")

    def _save_results(self, results: list, output_path: str):
        """Save results to JSON file"""
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✓ Results saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AI-Powered Phishing Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single URL
  python detect.py -u https://example.com

  # Analyze with ML model
  python detect.py -u https://example.com -m models/best_model.pkl

  # Batch analysis
  python detect.py -f urls.txt -o results.json

  # Interactive mode
  python detect.py -i

  # Check features only
  python detect.py -u https://example.com --features

  # Verbose output
  python detect.py -u https://example.com -v
        """,
    )

    parser.add_argument("-u", "--url", type=str, help="URL to analyze")
    parser.add_argument(
        "-f", "--file", type=str, help="File containing URLs (one per line)"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output file for results (JSON)"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive mode"
    )
    parser.add_argument("-m", "--model", type=str, help="Path to trained model")
    parser.add_argument("-s", "--scaler", type=str, help="Path to scaler file")
    parser.add_argument("--features", type=str, help="Path to feature names JSON")
    parser.add_argument(
        "--check-features", action="store_true", help="Display extracted features only"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Display banner
    print("\n" + "=" * 70)
    print("🛡️  AI-POWERED PHISHING DETECTION TOOL")
    print("=" * 70)

    # Initialize CLI
    cli = PhishingCLI()

    # Load model if specified
    if args.model:
        try:
            cli.load_model(args.model, args.scaler, args.features)
        except Exception as e:
            print(f"⚠️  Warning: Could not load model: {e}")
            print("Falling back to rule-based detection\n")

    # Execute based on mode
    try:
        if args.interactive:
            cli.interactive_mode()
        elif args.file:
            cli.analyze_batch(args.file, args.output)
        elif args.url:
            if args.check_features:
                cli.check_features(args.url)
            else:
                cli.analyze_url(args.url, args.verbose)
        else:
            # No arguments - show help
            parser.print_help()
            print("\n💡 Tip: Use -i for interactive mode or -u to analyze a URL")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
