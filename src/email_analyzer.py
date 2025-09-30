"""
Email Phishing Analyzer
Analyzes email content, headers, and attachments for phishing indicators
"""

import re
import email
from email import policy
from email.parser import BytesParser
import hashlib
import base64
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import dns.resolver
from bs4 import BeautifulSoup
import magic
import validators


class EmailPhishingAnalyzer:
    """Analyze emails for phishing indicators"""

    def __init__(self):
        self.suspicious_keywords = [
            "verify",
            "urgent",
            "suspended",
            "locked",
            "confirm",
            "update",
            "billing",
            "payment",
            "account",
            "security",
            "unusual",
            "activity",
            "click here",
            "act now",
            "limited time",
            "expire",
            "immediate action",
            "congratulations",
            "winner",
            "prize",
            "refund",
            "tax",
            "claim",
            "reset password",
            "validate",
            "credentials",
            "ssn",
            "social security",
        ]

        self.legitimate_domains = [
            "google.com",
            "microsoft.com",
            "apple.com",
            "amazon.com",
            "facebook.com",
            "linkedin.com",
            "twitter.com",
            "paypal.com",
        ]

    def analyze_email(self, email_content: bytes) -> Dict[str, any]:
        """
        Comprehensive email analysis

        Args:
            email_content: Raw email content as bytes

        Returns:
            Dictionary containing analysis results and risk score
        """
        # Parse email
        msg = BytesParser(policy=policy.default).parsebytes(email_content)

        features = {}

        # Header analysis
        features.update(self._analyze_headers(msg))

        # Body analysis
        features.update(self._analyze_body(msg))

        # Link analysis
        features.update(self._analyze_links(msg))

        # Attachment analysis
        features.update(self._analyze_attachments(msg))

        # Sender reputation
        features.update(self._analyze_sender_reputation(msg))

        # Calculate risk score
        risk_score = self._calculate_risk_score(features)

        return {
            "features": features,
            "risk_score": risk_score,
            "classification": self._classify_risk(risk_score),
            "warnings": self._generate_warnings(features),
        }

    def _analyze_headers(self, msg) -> Dict[str, float]:
        """Analyze email headers for suspicious patterns"""
        features = {}

        try:
            # 1. Sender email analysis
            sender = msg.get("From", "")
            features["sender_has_display_name_mismatch"] = (
                self._check_display_name_mismatch(sender)
            )

            # 2. Reply-To differs from From
            reply_to = msg.get("Reply-To", "")
            from_addr = self._extract_email_address(sender)
            reply_to_addr = self._extract_email_address(reply_to)
            features["reply_to_differs"] = (
                1.0 if reply_to_addr and reply_to_addr != from_addr else 0.0
            )

            # 3. Return-Path analysis
            return_path = msg.get("Return-Path", "")
            return_path_addr = self._extract_email_address(return_path)
            features["return_path_differs"] = (
                1.0 if return_path_addr and return_path_addr != from_addr else 0.0
            )

            # 4. X-Mailer header (some phishing emails lack this)
            features["has_x_mailer"] = 1.0 if msg.get("X-Mailer") else 0.0

            # 5. Received headers count (fewer = more suspicious)
            received_headers = msg.get_all("Received", [])
            features["num_received_headers"] = len(received_headers)

            # 6. Suspicious received headers
            features["has_suspicious_received"] = self._check_suspicious_received(
                received_headers
            )

            # 7. SPF, DKIM, DMARC authentication
            auth_results = msg.get("Authentication-Results", "")
            features["spf_pass"] = 1.0 if "spf=pass" in auth_results.lower() else 0.0
            features["dkim_pass"] = 1.0 if "dkim=pass" in auth_results.lower() else 0.0
            features["dmarc_pass"] = (
                1.0 if "dmarc=pass" in auth_results.lower() else 0.0
            )

            # 8. Message-ID format
            message_id = msg.get("Message-ID", "")
            features["has_valid_message_id"] = self._validate_message_id(
                message_id, from_addr
            )

            # 9. Date header analysis
            date_header = msg.get("Date", "")
            features["has_date_anomaly"] = self._check_date_anomaly(date_header)

            # 10. X-Priority or Importance (phishing often uses high priority)
            priority = msg.get("X-Priority", "")
            importance = msg.get("Importance", "")
            features["marked_high_priority"] = (
                1.0 if priority == "1" or importance.lower() == "high" else 0.0
            )

            # 11. Sender domain reputation
            sender_domain = self._extract_domain(from_addr)
            features["sender_domain_suspicious"] = self._check_domain_reputation(
                sender_domain
            )

            # 12. BCC used (often suspicious in phishing)
            features["has_bcc"] = 1.0 if msg.get("Bcc") else 0.0

            # 13. Content-Type analysis
            content_type = msg.get("Content-Type", "")
            features["is_html_email"] = (
                1.0 if "text/html" in content_type.lower() else 0.0
            )

        except Exception as e:
            print(f"Error analyzing headers: {e}")
            # Set default values
            for key in [
                "sender_has_display_name_mismatch",
                "reply_to_differs",
                "return_path_differs",
                "has_x_mailer",
                "num_received_headers",
                "has_suspicious_received",
                "spf_pass",
                "dkim_pass",
                "dmarc_pass",
                "has_valid_message_id",
                "has_date_anomaly",
                "marked_high_priority",
                "sender_domain_suspicious",
                "has_bcc",
                "is_html_email",
            ]:
                if key not in features:
                    features[key] = 0.0

        return features

    def _analyze_body(self, msg) -> Dict[str, float]:
        """Analyze email body content"""
        features = {}

        try:
            # Get body content
            body_text = ""
            body_html = ""

            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        body_text = part.get_content()
                    elif content_type == "text/html":
                        body_html = part.get_content()
            else:
                content_type = msg.get_content_type()
                if content_type == "text/plain":
                    body_text = msg.get_content()
                elif content_type == "text/html":
                    body_html = msg.get_content()

            # Combine for text analysis
            body_combined = body_text + " " + self._html_to_text(body_html)

            # 14. Suspicious keywords count
            features["num_suspicious_keywords"] = sum(
                1
                for keyword in self.suspicious_keywords
                if keyword.lower() in body_combined.lower()
            )

            # 15. Urgency indicators
            urgency_words = [
                "urgent",
                "immediate",
                "act now",
                "expire",
                "deadline",
                "hurry",
            ]
            features["has_urgency"] = (
                1.0
                if any(word in body_combined.lower() for word in urgency_words)
                else 0.0
            )

            # 16. Financial indicators
            financial_words = [
                "payment",
                "bank",
                "credit card",
                "account",
                "refund",
                "tax",
                "invoice",
            ]
            features["has_financial_terms"] = (
                1.0
                if any(word in body_combined.lower() for word in financial_words)
                else 0.0
            )

            # 17. Credential request
            credential_words = [
                "password",
                "username",
                "login",
                "credentials",
                "pin",
                "ssn",
                "social security",
            ]
            features["requests_credentials"] = (
                1.0
                if any(word in body_combined.lower() for word in credential_words)
                else 0.0
            )

            # 18. Greeting analysis (generic vs personalized)
            features["has_generic_greeting"] = self._check_generic_greeting(
                body_combined
            )

            # 19. Spelling and grammar errors (simple heuristic)
            features["has_many_caps"] = (
                1.0
                if sum(1 for c in body_combined if c.isupper())
                / max(len(body_combined), 1)
                > 0.15
                else 0.0
            )

            # 20. Body length
            features["body_length"] = len(body_combined)

            # 21. HTML complexity (if HTML email)
            if body_html:
                features["html_complexity"] = self._calculate_html_complexity(body_html)
                features["has_hidden_text"] = self._check_hidden_text(body_html)
                features["has_suspicious_css"] = self._check_suspicious_css(body_html)
            else:
                features["html_complexity"] = 0.0
                features["has_hidden_text"] = 0.0
                features["has_suspicious_css"] = 0.0

            # 22. Contains forms
            features["contains_form"] = 1.0 if "<form" in body_html.lower() else 0.0

            # 23. Contains JavaScript
            features["contains_javascript"] = (
                1.0
                if "<script" in body_html.lower() or "javascript:" in body_html.lower()
                else 0.0
            )

            # 24. Threatening language
            threat_words = [
                "suspended",
                "locked",
                "terminated",
                "legal action",
                "consequences",
                "unauthorized",
            ]
            features["has_threatening_language"] = (
                1.0
                if any(word in body_combined.lower() for word in threat_words)
                else 0.0
            )

            # 25. Exclamation marks count
            features["num_exclamation_marks"] = body_combined.count("!")

        except Exception as e:
            print(f"Error analyzing body: {e}")
            for key in [
                "num_suspicious_keywords",
                "has_urgency",
                "has_financial_terms",
                "requests_credentials",
                "has_generic_greeting",
                "has_many_caps",
                "body_length",
                "html_complexity",
                "has_hidden_text",
                "has_suspicious_css",
                "contains_form",
                "contains_javascript",
                "has_threatening_language",
                "num_exclamation_marks",
            ]:
                if key not in features:
                    features[key] = 0.0

        return features

    def _analyze_links(self, msg) -> Dict[str, float]:
        """Analyze links in email"""
        features = {}

        try:
            # Extract all links
            links = self._extract_links(msg)

            # 26. Number of links
            features["num_links"] = len(links)

            # 27. Has shortened URLs
            features["has_shortened_urls"] = self._check_shortened_urls(links)

            # 28. Has IP-based URLs
            features["has_ip_urls"] = self._check_ip_urls(links)

            # 29. Link-text mismatch (anchor text vs actual URL)
            features["has_link_text_mismatch"] = self._check_link_text_mismatch(msg)

            # 30. Links to different domains
            unique_domains = len(set(self._extract_domain(link) for link in links))
            features["num_unique_link_domains"] = unique_domains

            # 31. Suspicious TLDs in links
            features["has_suspicious_tld_links"] = self._check_suspicious_tlds(links)

            # 32. Has homograph attack (IDN)
            features["has_homograph_attack"] = self._check_homograph(links)

            # 33. Links contain suspicious keywords
            features["links_contain_suspicious_keywords"] = sum(
                1
                for link in links
                if any(keyword in link.lower() for keyword in self.suspicious_keywords)
            )

            # 34. HTTPS vs HTTP ratio
            if links:
                https_count = sum(1 for link in links if link.startswith("https://"))
                features["https_ratio"] = https_count / len(links)
            else:
                features["https_ratio"] = 0.0

            # 35. Links to known phishing domains
            features["links_to_known_phishing"] = self._check_known_phishing_domains(
                links
            )

        except Exception as e:
            print(f"Error analyzing links: {e}")
            for key in [
                "num_links",
                "has_shortened_urls",
                "has_ip_urls",
                "has_link_text_mismatch",
                "num_unique_link_domains",
                "has_suspicious_tld_links",
                "has_homograph_attack",
                "links_contain_suspicious_keywords",
                "https_ratio",
                "links_to_known_phishing",
            ]:
                if key not in features:
                    features[key] = 0.0

        return features

    def _analyze_attachments(self, msg) -> Dict[str, float]:
        """Analyze email attachments"""
        features = {}

        try:
            attachments = []

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_disposition() == "attachment":
                        attachments.append(part)

            # 36. Number of attachments
            features["num_attachments"] = len(attachments)

            # 37. Has executable attachments
            features["has_executable_attachment"] = self._check_executable_attachments(
                attachments
            )

            # 38. Has suspicious file extensions
            features["has_suspicious_extension"] = self._check_suspicious_extensions(
                attachments
            )

            # 39. Has double extensions
            features["has_double_extension"] = self._check_double_extensions(
                attachments
            )

            # 40. Attachment size anomalies
            features["has_large_attachment"] = self._check_large_attachments(
                attachments
            )

            # 41. Has password-protected archives
            features["has_protected_archive"] = self._check_protected_archives(
                attachments
            )

        except Exception as e:
            print(f"Error analyzing attachments: {e}")
            for key in [
                "num_attachments",
                "has_executable_attachment",
                "has_suspicious_extension",
                "has_double_extension",
                "has_large_attachment",
                "has_protected_archive",
            ]:
                if key not in features:
                    features[key] = 0.0

        return features

    def _analyze_sender_reputation(self, msg) -> Dict[str, float]:
        """Analyze sender reputation and domain"""
        features = {}

        try:
            sender = msg.get("From", "")
            sender_addr = self._extract_email_address(sender)
            sender_domain = self._extract_domain(sender_addr)

            # 42. Domain has MX records
            features["has_mx_records"] = self._check_mx_records(sender_domain)

            # 43. Domain age (approximated by checking if in known domains)
            features["is_established_domain"] = (
                1.0 if sender_domain in self.legitimate_domains else 0.0
            )

            # 44. Free email provider
            free_providers = [
                "gmail.com",
                "yahoo.com",
                "hotmail.com",
                "outlook.com",
                "aol.com",
                "mail.com",
            ]
            features["is_free_email"] = 1.0 if sender_domain in free_providers else 0.0

            # 45. Sender domain matches link domains
            links = self._extract_links(msg)
            if links:
                link_domains = [self._extract_domain(link) for link in links]
                features["sender_domain_matches_links"] = (
                    1.0 if sender_domain in link_domains else 0.0
                )
            else:
                features["sender_domain_matches_links"] = 0.0

            # 46. Similar to known brand (typosquatting)
            features["is_typosquatting"] = self._check_typosquatting(sender_domain)

        except Exception as e:
            print(f"Error analyzing sender reputation: {e}")
            for key in [
                "has_mx_records",
                "is_established_domain",
                "is_free_email",
                "sender_domain_matches_links",
                "is_typosquatting",
            ]:
                if key not in features:
                    features[key] = 0.0

        return features

    # Helper methods

    def _extract_email_address(self, field: str) -> str:
        """Extract email address from email field"""
        match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", field)
        return match.group(0) if match else ""

    def _extract_domain(self, email_or_url: str) -> str:
        """Extract domain from email address or URL"""
        if "@" in email_or_url:
            return email_or_url.split("@")[-1]
        elif "://" in email_or_url:
            match = re.search(r"://([^/]+)", email_or_url)
            return match.group(1) if match else ""
        return email_or_url

    def _check_display_name_mismatch(self, sender: str) -> float:
        """Check if display name suggests a different organization"""
        # Extract display name and email
        match = re.match(r"([^<]+)<([^>]+)>", sender)
        if match:
            display_name = match.group(1).strip().lower()
            email_addr = match.group(2).strip().lower()
            domain = self._extract_domain(email_addr)

            # Check if display name contains brand that doesn't match domain
            for brand in self.legitimate_domains:
                brand_name = brand.split(".")[0]
                if brand_name in display_name and brand_name not in domain:
                    return 1.0

        return 0.0

    def _check_suspicious_received(self, received_headers: List[str]) -> float:
        """Check for suspicious patterns in Received headers"""
        suspicious_indicators = ["unknown", "localhost", "dynamic", "dialup"]

        for header in received_headers:
            header_lower = header.lower()
            if any(indicator in header_lower for indicator in suspicious_indicators):
                return 1.0

        return 0.0

    def _validate_message_id(self, message_id: str, sender_email: str) -> float:
        """Validate Message-ID format matches sender domain"""
        if not message_id:
            return 0.0

        sender_domain = self._extract_domain(sender_email)
        if sender_domain in message_id:
            return 1.0

        return 0.0

    def _check_date_anomaly(self, date_header: str) -> float:
        """Check for date/time anomalies"""
        # For now, simple check if date exists
        return 0.0 if date_header else 1.0

    def _check_domain_reputation(self, domain: str) -> float:
        """Check if domain is suspicious"""
        suspicious_tlds = ["tk", "ml", "ga", "cf", "gq", "zip", "top"]

        if any(domain.endswith(f".{tld}") for tld in suspicious_tlds):
            return 1.0

        return 0.0

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text"""
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        except:
            return html

    def _check_generic_greeting(self, body: str) -> float:
        """Check for generic greetings"""
        generic_greetings = [
            "dear customer",
            "dear user",
            "dear member",
            "dear sir/madam",
            "hello customer",
            "valued customer",
            "dear account holder",
        ]

        body_lower = body.lower()[:200]  # Check first 200 chars
        return (
            1.0
            if any(greeting in body_lower for greeting in generic_greetings)
            else 0.0
        )

    def _calculate_html_complexity(self, html: str) -> float:
        """Calculate HTML complexity score"""
        # Count tags as a simple measure
        tag_count = html.count("<")
        return min(tag_count / 100.0, 10.0)  # Normalize to 0-10

    def _check_hidden_text(self, html: str) -> float:
        """Check for hidden text in HTML"""
        hidden_patterns = [
            r'style="[^"]*display:\s*none',
            r'style="[^"]*visibility:\s*hidden',
            r'style="[^"]*opacity:\s*0',
            r'style="[^"]*font-size:\s*0',
        ]

        for pattern in hidden_patterns:
            if re.search(pattern, html, re.IGNORECASE):
                return 1.0

        return 0.0

    def _check_suspicious_css(self, html: str) -> float:
        """Check for suspicious CSS"""
        # Check for CSS that might hide real URLs or content
        suspicious_patterns = [
            r"position:\s*absolute.*left:\s*-\d+px",
            r"text-indent:\s*-\d+px",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, html, re.IGNORECASE):
                return 1.0

        return 0.0

    def _extract_links(self, msg) -> List[str]:
        """Extract all links from email"""
        links = []

        # Get HTML body
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    html = part.get_content()
                    soup = BeautifulSoup(html, "html.parser")
                    for a_tag in soup.find_all("a", href=True):
                        links.append(a_tag["href"])
        else:
            if msg.get_content_type() == "text/html":
                html = msg.get_content()
                soup = BeautifulSoup(html, "html.parser")
                for a_tag in soup.find_all("a", href=True):
                    links.append(a_tag["href"])

        return links

    def _check_shortened_urls(self, links: List[str]) -> float:
        """Check if any links are shortened URLs"""
        shorteners = ["bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "buff.ly"]

        for link in links:
            if any(shortener in link.lower() for shortener in shorteners):
                return 1.0

        return 0.0

    def _check_ip_urls(self, links: List[str]) -> float:
        """Check if any URLs use IP addresses"""
        ip_pattern = r"://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"

        for link in links:
            if re.search(ip_pattern, link):
                return 1.0

        return 0.0

    def _check_link_text_mismatch(self, msg) -> float:
        """Check if link anchor text doesn't match URL"""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    html = part.get_content()
                    soup = BeautifulSoup(html, "html.parser")

                    for a_tag in soup.find_all("a", href=True):
                        anchor_text = a_tag.get_text().strip()
                        href = a_tag["href"]

                        # Check if anchor text looks like a URL but differs from href
                        if validators.url(anchor_text) and anchor_text != href:
                            return 1.0

        return 0.0

    def _check_suspicious_tlds(self, links: List[str]) -> float:
        """Check for suspicious TLDs in links"""
        suspicious_tlds = ["tk", "ml", "ga", "cf", "gq", "zip", "top", "work"]

        for link in links:
            domain = self._extract_domain(link)
            if any(domain.endswith(f".{tld}") for tld in suspicious_tlds):
                return 1.0

        return 0.0

    def _check_homograph(self, links: List[str]) -> float:
        """Check for IDN homograph attacks"""
        # Simple check for non-ASCII characters in domain
        for link in links:
            domain = self._extract_domain(link)
            if any(ord(char) > 127 for char in domain):
                return 1.0

        return 0.0

    def _check_known_phishing_domains(self, links: List[str]) -> float:
        """Check against known phishing domains (placeholder)"""
        # In production, this would check against a database
        return 0.0

    def _check_executable_attachments(self, attachments: List) -> float:
        """Check for executable attachments"""
        executable_extensions = [
            ".exe",
            ".bat",
            ".cmd",
            ".com",
            ".pif",
            ".scr",
            ".vbs",
            ".js",
        ]

        for attachment in attachments:
            filename = attachment.get_filename() or ""
            if any(filename.lower().endswith(ext) for ext in executable_extensions):
                return 1.0

        return 0.0

    def _check_suspicious_extensions(self, attachments: List) -> float:
        """Check for suspicious file extensions"""
        suspicious_extensions = [
            ".zip",
            ".rar",
            ".7z",
            ".iso",
            ".doc",
            ".docm",
            ".xlsm",
            ".pptm",
        ]

        for attachment in attachments:
            filename = attachment.get_filename() or ""
            if any(filename.lower().endswith(ext) for ext in suspicious_extensions):
                return 1.0

        return 0.0

    def _check_double_extensions(self, attachments: List) -> float:
        """Check for double file extensions"""
        for attachment in attachments:
            filename = attachment.get_filename() or ""
            # Count number of dots
            if filename.count(".") > 1:
                # Check if has suspicious pattern like .pdf.exe
                parts = filename.split(".")
                if len(parts) >= 3:
                    return 1.0

        return 0.0

    def _check_large_attachments(self, attachments: List) -> float:
        """Check for unusually large attachments"""
        for attachment in attachments:
            try:
                content = attachment.get_content()
                if len(content) > 10 * 1024 * 1024:  # > 10MB
                    return 1.0
            except:
                pass

        return 0.0

    def _check_protected_archives(self, attachments: List) -> float:
        """Check for password-protected archives"""
        # This is a placeholder - would need proper archive analysis
        for attachment in attachments:
            filename = attachment.get_filename() or ""
            if "password" in filename.lower() or "protected" in filename.lower():
                return 1.0

        return 0.0

    def _check_mx_records(self, domain: str) -> float:
        """Check if domain has valid MX records"""
        try:
            dns.resolver.resolve(domain, "MX")
            return 1.0
        except:
            return 0.0

    def _check_typosquatting(self, domain: str) -> float:
        """Check for typosquatting of known brands"""
        # Simple Levenshtein-like check
        for legit_domain in self.legitimate_domains:
            legit_name = legit_domain.split(".")[0]
            domain_name = domain.split(".")[0]

            # Check for single character difference
            if len(legit_name) == len(domain_name):
                diff_count = sum(1 for a, b in zip(legit_name, domain_name) if a != b)
                if diff_count == 1:
                    return 1.0

            # Check for character substitution (l->1, o->0, etc)
            substitutions = {"l": "1", "i": "1", "o": "0", "s": "5", "a": "4"}
            for char, sub in substitutions.items():
                if legit_name.replace(char, sub) == domain_name:
                    return 1.0

        return 0.0

    def _calculate_risk_score(self, features: Dict[str, float]) -> float:
        """Calculate overall risk score from features"""
        # Weighted scoring system
        weights = {
            "sender_has_display_name_mismatch": 5.0,
            "reply_to_differs": 3.0,
            "spf_pass": -5.0,
            "dkim_pass": -5.0,
            "dmarc_pass": -5.0,
            "num_suspicious_keywords": 2.0,
            "has_urgency": 4.0,
            "requests_credentials": 8.0,
            "has_threatening_language": 5.0,
            "has_link_text_mismatch": 7.0,
            "has_ip_urls": 6.0,
            "has_shortened_urls": 3.0,
            "has_executable_attachment": 10.0,
            "has_suspicious_extension": 5.0,
            "is_typosquatting": 8.0,
            "has_homograph_attack": 8.0,
            "sender_domain_suspicious": 6.0,
        }

        score = 50.0  # Base score

        for feature, value in features.items():
            if feature in weights:
                score += weights[feature] * value

        # Normalize to 0-100
        return max(0.0, min(100.0, score))

    def _classify_risk(self, risk_score: float) -> str:
        """Classify risk level based on score"""
        if risk_score >= 75:
            return "HIGH_RISK"
        elif risk_score >= 50:
            return "MEDIUM_RISK"
        elif risk_score >= 25:
            return "LOW_RISK"
        else:
            return "SAFE"

    def _generate_warnings(self, features: Dict[str, float]) -> List[str]:
        """Generate human-readable warnings"""
        warnings = []

        if features.get("sender_has_display_name_mismatch", 0) > 0:
            warnings.append("Display name doesn't match sender domain")

        if features.get("reply_to_differs", 0) > 0:
            warnings.appen
