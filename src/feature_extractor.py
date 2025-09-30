"""
Feature Extractor for Phishing URL Detection
Extracts various features from URLs for ML model training and prediction
"""

import re
import socket
import ssl
import whois
import dns.resolver
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import tldextract
import ipaddress
import requests
from typing import Dict, List, Tuple, Optional
import hashlib
from collections import Counter
import math


class URLFeatureExtractor:
    """Extract comprehensive features from URLs for phishing detection"""

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.suspicious_tlds = ["tk", "ml", "ga", "cf", "gq", "zip", "review"]
        self.suspicious_keywords = [
            "login",
            "signin",
            "account",
            "verify",
            "secure",
            "update",
            "banking",
            "paypal",
            "ebay",
            "amazon",
            "apple",
            "microsoft",
            "confirm",
            "suspended",
            "locked",
            "unusual",
            "click",
        ]

    def extract_all_features(self, url: str) -> Dict[str, float]:
        """
        Extract all features from a URL

        Args:
            url: The URL to analyze

        Returns:
            Dictionary containing all extracted features
        """
        features = {}

        # Basic URL features
        features.update(self._extract_url_based_features(url))

        # Domain-based features
        features.update(self._extract_domain_features(url))

        # Address bar based features
        features.update(self._extract_address_bar_features(url))

        # HTML and content-based features (if accessible)
        features.update(self._extract_content_features(url))

        # WHOIS and DNS features
        features.update(self._extract_whois_features(url))

        # SSL/Certificate features
        features.update(self._extract_ssl_features(url))

        return features

    def _extract_url_based_features(self, url: str) -> Dict[str, float]:
        """Extract features directly from URL string"""
        features = {}

        try:
            parsed = urlparse(url)

            # 1. URL Length
            features["url_length"] = len(url)

            # 2. Number of dots in URL
            features["num_dots"] = url.count(".")

            # 3. Number of hyphens
            features["num_hyphens"] = url.count("-")

            # 4. Number of underscores
            features["num_underscores"] = url.count("_")

            # 5. Number of slashes
            features["num_slashes"] = url.count("/")

            # 6. Number of question marks
            features["num_question_marks"] = url.count("?")

            # 7. Number of equal signs
            features["num_equals"] = url.count("=")

            # 8. Number of at symbols
            features["num_at"] = url.count("@")

            # 9. Number of ampersands
            features["num_ampersands"] = url.count("&")

            # 10. Number of pipe symbols
            features["num_pipes"] = url.count("|")

            # 11. Number of semicolons
            features["num_semicolons"] = url.count(";")

            # 12. Number of percent symbols (encoding)
            features["num_percent"] = url.count("%")

            # 13. Has IP address instead of domain
            features["has_ip_address"] = self._has_ip_address(parsed.netloc)

            # 14. Number of digits in URL
            features["num_digits"] = sum(c.isdigit() for c in url)

            # 15. Number of letters
            features["num_letters"] = sum(c.isalpha() for c in url)

            # 16. Ratio of digits to total characters
            features["digit_ratio"] = (
                features["num_digits"] / len(url) if len(url) > 0 else 0
            )

            # 17. Number of parameters
            features["num_params"] = len(parse_qs(parsed.query))

            # 18. Has suspicious keywords
            features["has_suspicious_keyword"] = any(
                keyword in url.lower() for keyword in self.suspicious_keywords
            )

            # 19. URL entropy (randomness measure)
            features["url_entropy"] = self._calculate_entropy(url)

            # 20. Length of hostname
            features["hostname_length"] = len(parsed.netloc)

            # 21. Path length
            features["path_length"] = len(parsed.path)

            # 22. Number of subdirectories
            features["num_subdirs"] = len([p for p in parsed.path.split("/") if p])

            # 23. Has HTTPS
            features["has_https"] = 1.0 if parsed.scheme == "https" else 0.0

            # 24. Has port number
            features["has_port"] = 1.0 if parsed.port else 0.0

            # 25. Number of fragments
            features["has_fragment"] = 1.0 if parsed.fragment else 0.0

        except Exception as e:
            print(f"Error extracting URL features: {e}")
            # Return default values
            for key in [
                "url_length",
                "num_dots",
                "num_hyphens",
                "num_underscores",
                "num_slashes",
                "num_question_marks",
                "num_equals",
                "num_at",
                "num_ampersands",
                "num_pipes",
                "num_semicolons",
                "num_percent",
                "has_ip_address",
                "num_digits",
                "num_letters",
                "digit_ratio",
                "num_params",
                "has_suspicious_keyword",
                "url_entropy",
                "hostname_length",
                "path_length",
                "num_subdirs",
                "has_https",
                "has_port",
                "has_fragment",
            ]:
                features[key] = 0.0

        return features

    def _extract_domain_features(self, url: str) -> Dict[str, float]:
        """Extract domain-specific features"""
        features = {}

        try:
            extracted = tldextract.extract(url)
            domain = extracted.domain
            subdomain = extracted.subdomain
            suffix = extracted.suffix

            # 26. Domain length
            features["domain_length"] = len(domain)

            # 27. Subdomain length
            features["subdomain_length"] = len(subdomain)

            # 28. Number of subdomain parts
            features["num_subdomain_parts"] = (
                len(subdomain.split(".")) if subdomain else 0
            )

            # 29. Has suspicious TLD
            features["has_suspicious_tld"] = (
                1.0 if suffix in self.suspicious_tlds else 0.0
            )

            # 30. TLD length
            features["tld_length"] = len(suffix)

            # 31. Domain has digits
            features["domain_has_digits"] = (
                1.0 if any(c.isdigit() for c in domain) else 0.0
            )

            # 32. Domain has hyphens
            features["domain_has_hyphens"] = 1.0 if "-" in domain else 0.0

            # 33. Subdomain has hyphens
            features["subdomain_has_hyphens"] = 1.0 if "-" in subdomain else 0.0

            # 34. Domain entropy
            features["domain_entropy"] = self._calculate_entropy(domain)

            # 35. Is domain a known brand (simple check)
            features["is_known_brand"] = self._is_known_brand(domain)

            # 36. Has www prefix
            features["has_www"] = 1.0 if subdomain.startswith("www") else 0.0

        except Exception as e:
            print(f"Error extracting domain features: {e}")
            for key in [
                "domain_length",
                "subdomain_length",
                "num_subdomain_parts",
                "has_suspicious_tld",
                "tld_length",
                "domain_has_digits",
                "domain_has_hyphens",
                "subdomain_has_hyphens",
                "domain_entropy",
                "is_known_brand",
                "has_www",
            ]:
                features[key] = 0.0

        return features

    def _extract_address_bar_features(self, url: str) -> Dict[str, float]:
        """Extract features visible in browser address bar"""
        features = {}

        try:
            parsed = urlparse(url)

            # 37. Using URL shortener
            features["is_shortened"] = self._is_shortened_url(url)

            # 38. Has double slash in path
            features["has_double_slash"] = 1.0 if "//" in parsed.path else 0.0

            # 39. Has embedded domain
            features["has_embedded_domain"] = self._has_embedded_domain(url)

            # 40. Prefix or suffix with dash
            domain_part = parsed.netloc.split(":")[0]
            features["prefix_suffix_dash"] = (
                1.0 if domain_part.startswith("-") or domain_part.endswith("-") else 0.0
            )

            # 41. Number of special characters
            special_chars = set("!@#$%^&*()_+-=[]{}|;:,.<>?")
            features["num_special_chars"] = sum(1 for c in url if c in special_chars)

        except Exception as e:
            print(f"Error extracting address bar features: {e}")
            for key in [
                "is_shortened",
                "has_double_slash",
                "has_embedded_domain",
                "prefix_suffix_dash",
                "num_special_chars",
            ]:
                features[key] = 0.0

        return features

    def _extract_content_features(self, url: str) -> Dict[str, float]:
        """Extract features from webpage content (if accessible)"""
        features = {}

        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                verify=False,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            content = response.text

            # 42. Page rank (simplified - based on response time)
            features["response_time"] = response.elapsed.total_seconds()

            # 43. Has form
            features["has_form"] = 1.0 if "<form" in content.lower() else 0.0

            # 44. Has input fields
            features["num_input_fields"] = content.lower().count("<input")

            # 45. Has password field
            features["has_password_field"] = (
                1.0
                if 'type="password"' in content.lower()
                or "type='password'" in content.lower()
                else 0.0
            )

            # 46. External resources ratio
            features["external_resources_ratio"] = self._calculate_external_resources(
                content, url
            )

            # 47. Has hidden elements
            features["has_hidden_elements"] = (
                1.0 if "hidden" in content.lower() else 0.0
            )

            # 48. Number of external links
            features["num_external_links"] = self._count_external_links(content, url)

            # 49. Has suspicious meta tags
            features["has_suspicious_meta"] = self._has_suspicious_meta_tags(content)

            # 50. Content length
            features["content_length"] = len(content)

        except Exception as e:
            # If page is not accessible, set default values
            for key in [
                "response_time",
                "has_form",
                "num_input_fields",
                "has_password_field",
                "external_resources_ratio",
                "has_hidden_elements",
                "num_external_links",
                "has_suspicious_meta",
                "content_length",
            ]:
                features[key] = -1.0  # -1 indicates feature could not be extracted

        return features

    def _extract_whois_features(self, url: str) -> Dict[str, float]:
        """Extract WHOIS-based features"""
        features = {}

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.split(":")[0]

            w = whois.whois(domain)

            # 51. Domain age
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]

            if creation_date:
                age_days = (datetime.now() - creation_date).days
                features["domain_age_days"] = age_days
                features["domain_age_months"] = age_days / 30.0
            else:
                features["domain_age_days"] = -1.0
                features["domain_age_months"] = -1.0

            # 52. Domain expiration date
            expiration_date = w.expiration_date
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]

            if expiration_date:
                days_until_expiration = (expiration_date - datetime.now()).days
                features["days_until_expiration"] = days_until_expiration
            else:
                features["days_until_expiration"] = -1.0

            # 53. Has registrar
            features["has_registrar"] = 1.0 if w.registrar else 0.0

            # 54. WHOIS privacy enabled
            if w.registrar and "privacy" in str(w.registrar).lower():
                features["whois_privacy"] = 1.0
            else:
                features["whois_privacy"] = 0.0

        except Exception as e:
            for key in [
                "domain_age_days",
                "domain_age_months",
                "days_until_expiration",
                "has_registrar",
                "whois_privacy",
            ]:
                features[key] = -1.0

        return features

    def _extract_ssl_features(self, url: str) -> Dict[str, float]:
        """Extract SSL certificate features"""
        features = {}

        try:
            parsed = urlparse(url)
            hostname = parsed.netloc.split(":")[0]

            if parsed.scheme == "https":
                try:
                    context = ssl.create_default_context()
                    with socket.create_connection(
                        (hostname, 443), timeout=self.timeout
                    ) as sock:
                        with context.wrap_socket(
                            sock, server_hostname=hostname
                        ) as secure_sock:
                            cert = secure_sock.getpeercert()

                            # 55. Certificate issuer
                            features["has_valid_cert"] = 1.0

                            # 56. Certificate age
                            not_before = ssl.cert_time_to_seconds(cert["notBefore"])
                            cert_age = (
                                datetime.now().timestamp() - not_before
                            ) / 86400  # days
                            features["cert_age_days"] = cert_age

                            # 57. Certificate remaining validity
                            not_after = ssl.cert_time_to_seconds(cert["notAfter"])
                            remaining_validity = (
                                not_after - datetime.now().timestamp()
                            ) / 86400
                            features["cert_remaining_days"] = remaining_validity

                            # 58. Certificate has SAN
                            san = cert.get("subjectAltName", [])
                            features["cert_has_san"] = 1.0 if san else 0.0
                except (ssl.SSLError, socket.timeout, socket.error, OSError) as ssl_err:
                    # SSL connection failed - mark as having issues but not necessarily invalid
                    # Some legitimate sites may have temporary SSL issues
                    features["has_valid_cert"] = 0.5  # Uncertain rather than definitely invalid
                    features["cert_age_days"] = -1.0
                    features["cert_remaining_days"] = -1.0
                    features["cert_has_san"] = 0.0
            else:
                # HTTP (not HTTPS)
                features["has_valid_cert"] = 0.0
                features["cert_age_days"] = -1.0
                features["cert_remaining_days"] = -1.0
                features["cert_has_san"] = 0.0

        except Exception as e:
            # General error - don't penalize too much as it could be network issues
            features["has_valid_cert"] = 0.5
            features["cert_age_days"] = -1.0
            features["cert_remaining_days"] = -1.0
            features["cert_has_san"] = 0.0

        return features

    # Helper methods

    def _has_ip_address(self, hostname: str) -> float:
        """Check if hostname is an IP address"""
        try:
            ipaddress.ip_address(hostname.split(":")[0])
            return 1.0
        except ValueError:
            return 0.0

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0

        prob = [float(text.count(c)) / len(text) for c in set(text)]
        entropy = -sum([p * math.log2(p) for p in prob if p > 0])
        return entropy

    def _is_known_brand(self, domain: str) -> float:
        """Check if domain matches known brand names"""
        known_brands = [
            "google",
            "facebook",
            "amazon",
            "microsoft",
            "apple",
            "paypal",
            "ebay",
            "netflix",
            "linkedin",
            "twitter",
            "instagram",
            "youtube",
            "chase",
            "wellsfargo",
            "bankofamerica",
            "citibank",
        ]

        domain_lower = domain.lower()
        for brand in known_brands:
            if brand in domain_lower and domain_lower != brand:
                # Domain contains brand name but is not exact match (suspicious)
                return 0.5
            elif domain_lower == brand:
                return 1.0

        return 0.0

    def _is_shortened_url(self, url: str) -> float:
        """Check if URL uses a shortening service"""
        shorteners = [
            "bit.ly",
            "goo.gl",
            "tinyurl.com",
            "ow.ly",
            "t.co",
            "buff.ly",
            "is.gd",
            "tiny.cc",
            "cli.gs",
            "short.to",
        ]

        return 1.0 if any(shortener in url.lower() for shortener in shorteners) else 0.0

    def _has_embedded_domain(self, url: str) -> float:
        """Check if URL has embedded domain name (e.g., paypal.com.fake.com)"""
        suspicious_patterns = [
            r"https?://[^/]*\.(com|net|org)\.[^/]*",
            r"@[^/]*\.(com|net|org)",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, url):
                return 1.0

        return 0.0

    def _calculate_external_resources(self, html: str, base_url: str) -> float:
        """Calculate ratio of external resources to total resources"""
        try:
            parsed_base = urlparse(base_url)
            base_domain = parsed_base.netloc

            # Find all src and href attributes
            src_pattern = r'(?:src|href)=["\']([^"\']+)["\']'
            resources = re.findall(src_pattern, html, re.IGNORECASE)

            if not resources:
                return 0.0

            external_count = 0
            for resource in resources:
                if resource.startswith("http"):
                    parsed_resource = urlparse(resource)
                    if parsed_resource.netloc != base_domain:
                        external_count += 1

            return external_count / len(resources)

        except Exception:
            return 0.0

    def _count_external_links(self, html: str, base_url: str) -> float:
        """Count number of external links"""
        try:
            parsed_base = urlparse(base_url)
            base_domain = parsed_base.netloc

            href_pattern = r'<a\s+(?:[^>]*?\s+)?href=["\']([^"\']+)["\']'
            links = re.findall(href_pattern, html, re.IGNORECASE)

            external_count = 0
            for link in links:
                if link.startswith("http"):
                    parsed_link = urlparse(link)
                    if parsed_link.netloc != base_domain:
                        external_count += 1

            return float(external_count)

        except Exception:
            return 0.0

    def _has_suspicious_meta_tags(self, html: str) -> float:
        """Check for suspicious meta tags"""
        suspicious_content = ["noindex", "nofollow", "noarchive"]
        meta_pattern = r'<meta[^>]*content=["\']([^"\']*)["\']'

        matches = re.findall(meta_pattern, html, re.IGNORECASE)

        for match in matches:
            if any(susp in match.lower() for susp in suspicious_content):
                return 1.0

        return 0.0


if __name__ == "__main__":
    # Example usage
    extractor = URLFeatureExtractor()

    # Test with a legitimate URL
    legitimate_url = "https://www.google.com"
    print("Extracting features from legitimate URL...")
    features = extractor.extract_all_features(legitimate_url)
    print(f"Extracted {len(features)} features")

    # Test with a suspicious URL
    suspicious_url = "http://paypa1-secure-login.tk/verify-account.php?id=12345"
    print("\nExtracting features from suspicious URL...")
    features = extractor.extract_all_features(suspicious_url)
    print(f"Extracted {len(features)} features")

    # Print some key features
    print("\nKey features:")
    for key, value in list(features.items())[:10]:
        print(f"  {key}: {value}")
