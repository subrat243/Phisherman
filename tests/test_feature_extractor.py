"""
Unit Tests for Feature Extractor
Tests the URL feature extraction functionality
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from feature_extractor import URLFeatureExtractor


class TestURLFeatureExtractor(unittest.TestCase):
    """Test cases for URL feature extraction"""

    def setUp(self):
        """Set up test fixtures"""
        self.extractor = URLFeatureExtractor()

    def test_legitimate_url(self):
        """Test feature extraction from legitimate URL"""
        url = "https://www.google.com"
        features = self.extractor.extract_all_features(url)

        # Check that features are extracted
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)

        # Check specific features
        self.assertEqual(features.get("has_https", 0), 1.0)
        self.assertEqual(features.get("has_ip_address", 0), 0.0)
        self.assertLess(features.get("url_length", 0), 50)

    def test_suspicious_url(self):
        """Test feature extraction from suspicious URL"""
        url = "http://paypal-verify.tk/login.php?id=12345"
        features = self.extractor.extract_all_features(url)

        # Check suspicious indicators
        self.assertEqual(features.get("has_https", 0), 0.0)
        self.assertEqual(features.get("has_suspicious_tld", 0), 1.0)
        self.assertGreater(features.get("has_suspicious_keyword", 0), 0)

    def test_ip_based_url(self):
        """Test detection of IP-based URLs"""
        url = "http://192.168.1.1/login"
        features = self.extractor.extract_all_features(url)

        self.assertEqual(features.get("has_ip_address", 0), 1.0)

    def test_url_length(self):
        """Test URL length feature"""
        short_url = "https://example.com"
        long_url = "https://very-long-subdomain.example.com/very/long/path/with/many/segments?param1=value1&param2=value2"

        short_features = self.extractor.extract_all_features(short_url)
        long_features = self.extractor.extract_all_features(long_url)

        self.assertLess(
            short_features.get("url_length", 0), long_features.get("url_length", 0)
        )

    def test_subdomain_detection(self):
        """Test subdomain feature extraction"""
        url = "https://mail.google.com"
        features = self.extractor.extract_all_features(url)

        self.assertGreater(features.get("subdomain_length", 0), 0)
        self.assertGreater(features.get("num_subdomain_parts", 0), 0)

    def test_has_ip_address(self):
        """Test IP address detection"""
        # Test with IP
        self.assertEqual(self.extractor._has_ip_address("192.168.1.1"), 1.0)

        # Test with domain
        self.assertEqual(self.extractor._has_ip_address("example.com"), 0.0)

    def test_calculate_entropy(self):
        """Test entropy calculation"""
        # Low entropy (repeated characters)
        low_entropy = self.extractor._calculate_entropy("aaaaaaa")

        # High entropy (random characters)
        high_entropy = self.extractor._calculate_entropy("a1b2c3d4e5")

        self.assertLess(low_entropy, high_entropy)

    def test_is_shortened_url(self):
        """Test shortened URL detection"""
        shortened = "http://bit.ly/abc123"
        normal = "https://example.com/page"

        self.assertEqual(self.extractor._is_shortened_url(shortened), 1.0)
        self.assertEqual(self.extractor._is_shortened_url(normal), 0.0)

    def test_has_embedded_domain(self):
        """Test embedded domain detection"""
        # Suspicious: has embedded domain
        suspicious = "http://paypal.com.fake-site.com"
        normal = "https://www.paypal.com"

        self.assertEqual(self.extractor._has_embedded_domain(suspicious), 1.0)
        self.assertEqual(self.extractor._has_embedded_domain(normal), 0.0)

    def test_url_with_parameters(self):
        """Test URL parameter counting"""
        url = "https://example.com/page?param1=value1&param2=value2&param3=value3"
        features = self.extractor.extract_all_features(url)

        self.assertEqual(features.get("num_params", 0), 3)
        self.assertGreater(features.get("num_equals", 0), 0)
        self.assertGreater(features.get("num_ampersands", 0), 0)

    def test_https_detection(self):
        """Test HTTPS protocol detection"""
        https_url = "https://secure-site.com"
        http_url = "http://insecure-site.com"

        https_features = self.extractor.extract_all_features(https_url)
        http_features = self.extractor.extract_all_features(http_url)

        self.assertEqual(https_features.get("has_https", 0), 1.0)
        self.assertEqual(http_features.get("has_https", 0), 0.0)

    def test_suspicious_keywords(self):
        """Test suspicious keyword detection"""
        suspicious_url = "https://account-verify-login-update.com"
        features = self.extractor.extract_all_features(suspicious_url)

        self.assertEqual(features.get("has_suspicious_keyword", 0), 1.0)

    def test_special_characters(self):
        """Test special character counting"""
        url = "https://example.com/path?a=b&c=d#fragment"
        features = self.extractor.extract_all_features(url)

        self.assertGreater(features.get("num_dots", 0), 0)
        self.assertGreater(features.get("num_slashes", 0), 0)
        self.assertGreater(features.get("num_question_marks", 0), 0)
        self.assertGreater(features.get("num_equals", 0), 0)

    def test_domain_features(self):
        """Test domain-specific features"""
        url = "https://test-123.example.com"
        features = self.extractor.extract_all_features(url)

        self.assertGreater(features.get("domain_length", 0), 0)
        self.assertEqual(
            features.get("domain_has_digits", 0), 0.0
        )  # Domain is "example", no digits
        self.assertEqual(
            features.get("domain_has_hyphens", 0), 0.0
        )  # Domain is "example", no hyphens

    def test_port_detection(self):
        """Test port number detection"""
        url_with_port = "https://example.com:8080/path"
        url_without_port = "https://example.com/path"

        with_port_features = self.extractor.extract_all_features(url_with_port)
        without_port_features = self.extractor.extract_all_features(url_without_port)

        self.assertEqual(with_port_features.get("has_port", 0), 1.0)
        self.assertEqual(without_port_features.get("has_port", 0), 0.0)

    def test_fragment_detection(self):
        """Test URL fragment detection"""
        url_with_fragment = "https://example.com/page#section"
        url_without_fragment = "https://example.com/page"

        with_fragment = self.extractor.extract_all_features(url_with_fragment)
        without_fragment = self.extractor.extract_all_features(url_without_fragment)

        self.assertEqual(with_fragment.get("has_fragment", 0), 1.0)
        self.assertEqual(without_fragment.get("has_fragment", 0), 0.0)

    def test_error_handling(self):
        """Test error handling with invalid URLs"""
        invalid_urls = [
            "not-a-url",
            "",
            "javascript:alert(1)",
            "ftp://example.com",  # FTP might have different handling
        ]

        for url in invalid_urls:
            try:
                features = self.extractor.extract_all_features(url)
                # Should return a dict even on error
                self.assertIsInstance(features, dict)
            except Exception as e:
                self.fail(
                    f"Feature extractor raised unexpected exception for '{url}': {e}"
                )


class TestHelperMethods(unittest.TestCase):
    """Test helper methods in feature extractor"""

    def setUp(self):
        """Set up test fixtures"""
        self.extractor = URLFeatureExtractor()

    def test_is_known_brand(self):
        """Test known brand detection"""
        # Exact match
        self.assertEqual(self.extractor._is_known_brand("google"), 1.0)

        # Contains but not exact (suspicious)
        self.assertEqual(self.extractor._is_known_brand("google-login"), 0.5)

        # Not a known brand
        self.assertEqual(self.extractor._is_known_brand("randomsite"), 0.0)

    def test_calculate_entropy_edge_cases(self):
        """Test entropy calculation edge cases"""
        # Empty string
        self.assertEqual(self.extractor._calculate_entropy(""), 0.0)

        # Single character
        entropy = self.extractor._calculate_entropy("a")
        self.assertEqual(entropy, 0.0)

        # All same characters
        entropy = self.extractor._calculate_entropy("aaaa")
        self.assertEqual(entropy, 0.0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
