"""
Data Collection and Preparation for Phishing Detection
Collects and prepares datasets for training ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import requests
from tqdm import tqdm
import time
from urllib.parse import urlparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from feature_extractor import URLFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhishingDataCollector:
    """Collect and prepare phishing detection datasets"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

        self.feature_extractor = URLFeatureExtractor()

        # Public phishing datasets URLs
        self.dataset_sources = {
            "phishtank": "http://data.phishtank.com/data/online-valid.csv",
            "openphish": "https://openphish.com/feed.txt",
        }

    def download_phishtank_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Download phishing URLs from PhishTank

        Args:
            limit: Maximum number of URLs to download

        Returns:
            DataFrame with phishing URLs
        """
        logger.info("Downloading PhishTank data...")

        try:
            # Note: PhishTank requires API key for real-time data
            # This is a placeholder - you need to register at phishtank.com

            # For demo, we'll create sample phishing URLs
            phishing_urls = [
                "http://paypal-security-verify.com/login",
                "http://amazon-customer-service.tk/update",
                "http://secure-apple-id.ml/verify",
                "http://192.168.1.1/microsoft-login",
                "http://facebook-secure-login.ga/account",
                "http://netflix-billing-update.cf/payment",
                "http://google-account-recovery.gq/reset",
                "http://chase-bank-verify.tk/login",
                "http://wellsfargo-security.ml/confirm",
                "http://bankofamerica-alert.ga/verify",
                "http://paypal-resolution-center.cf/dispute",
                "http://ebay-seller-update.tk/account",
                "http://instagram-copyright-notice.ml/appeal",
                "http://linkedin-premium-offer.ga/upgrade",
                "http://twitter-verification-blue.cf/verify",
                "http://microsoft-office365-renewal.tk/billing",
                "http://adobe-account-suspended.ml/restore",
                "http://dropbox-storage-full.ga/upgrade",
                "http://spotify-premium-free.cf/claim",
                "http://coinbase-security-alert.tk/verify",
            ]

            df = pd.DataFrame(
                {
                    "url": phishing_urls,
                    "label": 1,  # 1 = phishing
                    "source": "phishtank",
                }
            )

            if limit:
                df = df.head(limit)

            logger.info(f"Downloaded {len(df)} phishing URLs from PhishTank")
            return df

        except Exception as e:
            logger.error(f"Error downloading PhishTank data: {e}")
            return pd.DataFrame()

    def download_openphish_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Download phishing URLs from OpenPhish

        Args:
            limit: Maximum number of URLs to download

        Returns:
            DataFrame with phishing URLs
        """
        logger.info("Downloading OpenPhish data...")

        try:
            # Sample phishing URLs (OpenPhish feed requires subscription)
            phishing_urls = [
                "http://secure-signin-apple.com/id/verify",
                "http://account-verification-paypal.net/login",
                "http://amazon-help-center.org/update-payment",
                "http://netflix-payment-declined.info/billing",
                "http://microsoft-account-unusual-signin.com/verify",
                "http://facebook-security-check.org/confirm",
                "http://instagram-help-center.net/copyright",
                "http://linkedin-premium-trial.org/activate",
                "http://chase-fraud-alert.net/verify-identity",
                "http://wellsfargo-online-banking.org/login",
                "http://americanexpress-card-services.com/activate",
                "http://citibank-account-alert.net/verify",
                "http://usbank-security-check.org/confirm",
                "http://discover-card-services.com/activate",
                "http://capital-one-fraud-alert.net/verify",
            ]

            df = pd.DataFrame(
                {
                    "url": phishing_urls,
                    "label": 1,  # 1 = phishing
                    "source": "openphish",
                }
            )

            if limit:
                df = df.head(limit)

            logger.info(f"Downloaded {len(df)} phishing URLs from OpenPhish")
            return df

        except Exception as e:
            logger.error(f"Error downloading OpenPhish data: {e}")
            return pd.DataFrame()

    def get_legitimate_urls(self, limit: int = 100) -> pd.DataFrame:
        """
        Get legitimate URLs from popular websites

        Args:
            limit: Number of legitimate URLs to generate

        Returns:
            DataFrame with legitimate URLs
        """
        logger.info("Collecting legitimate URLs...")

        # Top legitimate websites
        legitimate_urls = [
            # Tech Companies
            "https://www.google.com",
            "https://www.microsoft.com",
            "https://www.apple.com",
            "https://www.amazon.com",
            "https://www.facebook.com",
            "https://www.twitter.com",
            "https://www.instagram.com",
            "https://www.linkedin.com",
            "https://www.youtube.com",
            "https://www.netflix.com",
            "https://www.spotify.com",
            "https://www.dropbox.com",
            "https://www.adobe.com",
            "https://www.salesforce.com",
            "https://www.oracle.com",
            "https://www.ibm.com",
            "https://www.cisco.com",
            "https://www.intel.com",
            "https://www.nvidia.com",
            "https://www.amd.com",
            # Financial
            "https://www.paypal.com",
            "https://www.chase.com",
            "https://www.bankofamerica.com",
            "https://www.wellsfargo.com",
            "https://www.citi.com",
            "https://www.capitalone.com",
            "https://www.usbank.com",
            "https://www.americanexpress.com",
            "https://www.discover.com",
            "https://www.fidelity.com",
            "https://www.schwab.com",
            "https://www.vanguard.com",
            "https://www.etrade.com",
            "https://www.tdameritrade.com",
            "https://www.robinhood.com",
            "https://www.coinbase.com",
            # E-commerce
            "https://www.ebay.com",
            "https://www.walmart.com",
            "https://www.target.com",
            "https://www.bestbuy.com",
            "https://www.homedepot.com",
            "https://www.lowes.com",
            "https://www.costco.com",
            "https://www.wayfair.com",
            "https://www.etsy.com",
            "https://www.aliexpress.com",
            # News & Media
            "https://www.cnn.com",
            "https://www.bbc.com",
            "https://www.nytimes.com",
            "https://www.washingtonpost.com",
            "https://www.reuters.com",
            "https://www.theguardian.com",
            "https://www.forbes.com",
            "https://www.bloomberg.com",
            "https://www.wsj.com",
            "https://www.npr.org",
            # Education
            "https://www.wikipedia.org",
            "https://www.coursera.org",
            "https://www.udemy.com",
            "https://www.khanacademy.org",
            "https://www.edx.org",
            "https://www.mit.edu",
            "https://www.stanford.edu",
            "https://www.harvard.edu",
            "https://www.berkeley.edu",
            "https://www.oxford.ac.uk",
            # Government
            "https://www.usa.gov",
            "https://www.irs.gov",
            "https://www.ssa.gov",
            "https://www.cdc.gov",
            "https://www.nasa.gov",
            "https://www.usps.com",
            "https://www.whitehouse.gov",
            "https://www.state.gov",
            "https://www.justice.gov",
            "https://www.fbi.gov",
            # Developer Tools
            "https://www.github.com",
            "https://www.stackoverflow.com",
            "https://www.gitlab.com",
            "https://www.bitbucket.org",
            "https://www.docker.com",
            "https://www.kubernetes.io",
            "https://www.aws.amazon.com",
            "https://www.azure.microsoft.com",
            "https://www.cloud.google.com",
            "https://www.digitalocean.com",
            # Communication
            "https://www.gmail.com",
            "https://www.outlook.com",
            "https://www.yahoo.com",
            "https://www.zoom.us",
            "https://www.slack.com",
            "https://www.discord.com",
            "https://www.telegram.org",
            "https://www.whatsapp.com",
            "https://www.skype.com",
            "https://www.teams.microsoft.com",
        ]

        # Add variations with paths
        extended_urls = legitimate_urls.copy()
        for url in legitimate_urls[:20]:
            extended_urls.append(f"{url}/about")
            extended_urls.append(f"{url}/contact")
            extended_urls.append(f"{url}/help")
            extended_urls.append(f"{url}/support")

        df = pd.DataFrame(
            {
                "url": extended_urls[:limit],
                "label": 0,  # 0 = legitimate
                "source": "legitimate",
            }
        )

        logger.info(f"Collected {len(df)} legitimate URLs")
        return df

    def combine_datasets(self) -> pd.DataFrame:
        """
        Combine all data sources into single dataset

        Returns:
            Combined DataFrame
        """
        logger.info("Combining datasets...")

        datasets = []

        # Get phishing URLs
        phishtank_df = self.download_phishtank_data()
        if not phishtank_df.empty:
            datasets.append(phishtank_df)

        openphish_df = self.download_openphish_data()
        if not openphish_df.empty:
            datasets.append(openphish_df)

        # Get legitimate URLs
        legitimate_df = self.get_legitimate_urls(limit=100)
        if not legitimate_df.empty:
            datasets.append(legitimate_df)

        # Combine all datasets
        if datasets:
            combined_df = pd.concat(datasets, ignore_index=True)

            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=["url"])

            # Shuffle
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(
                drop=True
            )

            logger.info(f"Combined dataset: {len(combined_df)} URLs")
            logger.info(f"  Phishing: {sum(combined_df['label'] == 1)}")
            logger.info(f"  Legitimate: {sum(combined_df['label'] == 0)}")

            # Save raw dataset
            raw_file = self.raw_dir / "combined_urls.csv"
            combined_df.to_csv(raw_file, index=False)
            logger.info(f"Saved raw dataset to {raw_file}")

            return combined_df

        return pd.DataFrame()

    def extract_features_batch(
        self, urls: List[str], max_workers: int = 10
    ) -> List[Dict[str, float]]:
        """
        Extract features from URLs in parallel

        Args:
            urls: List of URLs
            max_workers: Number of parallel workers

        Returns:
            List of feature dictionaries
        """
        features_list = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self.feature_extractor.extract_all_features, url): url
                for url in urls
            }

            with tqdm(total=len(urls), desc="Extracting features") as pbar:
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        features = future.result()
                        features_list.append(features)
                    except Exception as e:
                        logger.error(f"Error extracting features from {url}: {e}")
                        features_list.append({})

                    pbar.update(1)

        return features_list

    def prepare_training_data(
        self, df: pd.DataFrame, max_workers: int = 10
    ) -> pd.DataFrame:
        """
        Prepare data for model training by extracting features

        Args:
            df: DataFrame with URLs and labels
            max_workers: Number of parallel workers

        Returns:
            DataFrame with extracted features
        """
        logger.info("Preparing training data...")

        urls = df["url"].tolist()
        labels = df["label"].tolist()

        # Extract features
        features_list = self.extract_features_batch(urls, max_workers)

        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)

        # Add label
        features_df["is_phishing"] = labels

        # Remove rows with too many missing features
        threshold = len(features_df.columns) * 0.5
        features_df = features_df.dropna(thresh=threshold)

        # Fill remaining missing values
        features_df = features_df.fillna(-1)

        logger.info(f"Prepared dataset shape: {features_df.shape}")
        logger.info(f"Features: {features_df.shape[1] - 1}")
        logger.info(f"Samples: {len(features_df)}")

        # Save processed dataset
        processed_file = self.processed_dir / "training_data.csv"
        features_df.to_csv(processed_file, index=False)
        logger.info(f"Saved processed dataset to {processed_file}")

        # Save feature names
        feature_names = [col for col in features_df.columns if col != "is_phishing"]
        feature_names_file = self.processed_dir / "feature_names.json"
        with open(feature_names_file, "w") as f:
            json.dump(feature_names, f, indent=2)
        logger.info(f"Saved feature names to {feature_names_file}")

        return features_df

    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Load previously processed data

        Returns:
            DataFrame with processed data or None
        """
        processed_file = self.processed_dir / "training_data.csv"

        if processed_file.exists():
            logger.info(f"Loading processed data from {processed_file}")
            df = pd.read_csv(processed_file)
            logger.info(f"Loaded {len(df)} samples with {df.shape[1]} features")
            return df

        logger.warning("No processed data found")
        return None

    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get statistics about the dataset

        Args:
            df: Dataset DataFrame

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_samples": len(df),
            "num_features": df.shape[1] - 1,
            "phishing_samples": sum(df["is_phishing"] == 1),
            "legitimate_samples": sum(df["is_phishing"] == 0),
            "missing_values": df.isnull().sum().to_dict(),
            "feature_ranges": {},
        }

        # Calculate feature ranges
        for col in df.columns:
            if col != "is_phishing":
                stats["feature_ranges"][col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                }

        return stats

    def balance_dataset(
        self, df: pd.DataFrame, method: str = "undersample"
    ) -> pd.DataFrame:
        """
        Balance the dataset to handle class imbalance

        Args:
            df: Imbalanced dataset
            method: 'undersample' or 'oversample'

        Returns:
            Balanced dataset
        """
        logger.info(f"Balancing dataset using {method}...")

        phishing = df[df["is_phishing"] == 1]
        legitimate = df[df["is_phishing"] == 0]

        logger.info(
            f"Original - Phishing: {len(phishing)}, Legitimate: {len(legitimate)}"
        )

        if method == "undersample":
            # Undersample majority class
            min_samples = min(len(phishing), len(legitimate))
            phishing = phishing.sample(n=min_samples, random_state=42)
            legitimate = legitimate.sample(n=min_samples, random_state=42)

        elif method == "oversample":
            # Oversample minority class
            max_samples = max(len(phishing), len(legitimate))
            phishing = phishing.sample(n=max_samples, replace=True, random_state=42)
            legitimate = legitimate.sample(n=max_samples, replace=True, random_state=42)

        # Combine and shuffle
        balanced_df = pd.concat([phishing, legitimate], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(
            f"Balanced - Phishing: {sum(balanced_df['is_phishing'] == 1)}, "
            f"Legitimate: {sum(balanced_df['is_phishing'] == 0)}"
        )

        return balanced_df


def main():
    """Main data collection and preparation pipeline"""
    print("=" * 70)
    print("Phishing Detection - Data Collection & Preparation")
    print("=" * 70)

    # Initialize collector
    collector = PhishingDataCollector()

    # Step 1: Collect raw URLs
    print("\nStep 1: Collecting URLs...")
    raw_df = collector.combine_datasets()

    if raw_df.empty:
        print("Error: No data collected")
        return

    # Step 2: Extract features
    print("\nStep 2: Extracting features...")
    processed_df = collector.prepare_training_data(raw_df, max_workers=5)

    # Step 3: Balance dataset
    print("\nStep 3: Balancing dataset...")
    balanced_df = collector.balance_dataset(processed_df, method="undersample")

    # Save balanced dataset
    balanced_file = collector.processed_dir / "training_data_balanced.csv"
    balanced_df.to_csv(balanced_file, index=False)
    print(f"Saved balanced dataset to {balanced_file}")

    # Step 4: Get statistics
    print("\nStep 4: Dataset Statistics...")
    stats = collector.get_data_statistics(balanced_df)
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Features: {stats['num_features']}")
    print(f"Phishing: {stats['phishing_samples']}")
    print(f"Legitimate: {stats['legitimate_samples']}")

    # Save statistics
    stats_file = collector.processed_dir / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_file}")

    print("\n" + "=" * 70)
    print("Data preparation complete!")
    print(f"Training data ready at: {balanced_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
