#!/usr/bin/env python3
"""
Dataset Loader - Load and process various phishing datasets
Supports CSV, JSON, and common dataset formats
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and process phishing datasets from various formats"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'uci', 'kaggle']
    
    def load_csv(
        self,
        path: str,
        url_column: str = 'url',
        label_column: str = 'label',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Args:
            path: Path to CSV file
            url_column: Name of URL column
            label_column: Name of label column (0=legitimate, 1=phishing)
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with 'url' and 'is_phishing' columns
        """
        logger.info(f"Loading CSV dataset from {path}")
        
        try:
            df = pd.read_csv(path, **kwargs)
            
            # Validate columns exist
            if url_column not in df.columns:
                raise ValueError(f"URL column '{url_column}' not found in dataset")
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in dataset")
            
            # Standardize column names
            df = df.rename(columns={
                url_column: 'url',
                label_column: 'is_phishing'
            })
            
            # Keep only necessary columns
            df = df[['url', 'is_phishing']]
            
            # Validate labels
            df = self._validate_labels(df)
            
            logger.info(f"Loaded {len(df)} samples from CSV")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def load_json(
        self,
        path: str,
        url_field: str = 'url',
        label_field: str = 'label'
    ) -> pd.DataFrame:
        """
        Load dataset from JSON file
        
        Args:
            path: Path to JSON file
            url_field: Name of URL field
            label_field: Name of label field
            
        Returns:
            DataFrame with 'url' and 'is_phishing' columns
        """
        logger.info(f"Loading JSON dataset from {path}")
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")
            
            # Standardize column names
            df = df.rename(columns={
                url_field: 'url',
                label_field: 'is_phishing'
            })
            
            # Keep only necessary columns
            df = df[['url', 'is_phishing']]
            
            # Validate labels
            df = self._validate_labels(df)
            
            logger.info(f"Loaded {len(df)} samples from JSON")
            return df
            
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise
    
    def load_uci_format(self, path: str) -> pd.DataFrame:
        """
        Load UCI Phishing Websites Dataset format
        
        Args:
            path: Path to UCI dataset file
            
        Returns:
            DataFrame with 'url' and 'is_phishing' columns
        """
        logger.info(f"Loading UCI format dataset from {path}")
        
        try:
            # UCI dataset typically has features + label in last column
            df = pd.read_csv(path)
            
            # Assume last column is the label
            label_col = df.columns[-1]
            
            # If there's a URL column, use it; otherwise create placeholder
            if 'url' in df.columns or 'URL' in df.columns:
                url_col = 'url' if 'url' in df.columns else 'URL'
            else:
                # Create placeholder URLs
                df['url'] = [f"sample_{i}" for i in range(len(df))]
                url_col = 'url'
            
            # Standardize
            df = df.rename(columns={label_col: 'is_phishing'})
            df = df[['url', 'is_phishing']]
            
            # Validate labels
            df = self._validate_labels(df)
            
            logger.info(f"Loaded {len(df)} samples from UCI format")
            return df
            
        except Exception as e:
            logger.error(f"Error loading UCI format: {e}")
            raise
    
    def load_dataset(
        self,
        path: str,
        format: str = 'auto',
        **kwargs
    ) -> pd.DataFrame:
        """
        Auto-detect and load dataset
        
        Args:
            path: Path to dataset file
            format: Dataset format ('auto', 'csv', 'json', 'uci')
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            DataFrame with 'url' and 'is_phishing' columns
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        # Auto-detect format
        if format == 'auto':
            suffix = path_obj.suffix.lower()
            if suffix == '.csv':
                format = 'csv'
            elif suffix == '.json':
                format = 'json'
            else:
                format = 'csv'  # Default to CSV
        
        # Load based on format
        if format == 'csv':
            return self.load_csv(path, **kwargs)
        elif format == 'json':
            return self.load_json(path, **kwargs)
        elif format == 'uci':
            return self.load_uci_format(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _validate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean labels"""
        # Convert labels to 0/1
        df['is_phishing'] = df['is_phishing'].astype(int)
        
        # Ensure only 0 and 1
        valid_labels = df['is_phishing'].isin([0, 1])
        if not valid_labels.all():
            logger.warning(f"Removing {(~valid_labels).sum()} samples with invalid labels")
            df = df[valid_labels]
        
        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['url'])
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} duplicate URLs")
        
        # Remove empty URLs
        df = df[df['url'].notna() & (df['url'] != '')]
        
        return df.reset_index(drop=True)
    
    def merge_datasets(self, *dataframes: pd.DataFrame) -> pd.DataFrame:
        """
        Merge multiple datasets
        
        Args:
            *dataframes: Variable number of DataFrames to merge
            
        Returns:
            Merged DataFrame
        """
        if not dataframes:
            raise ValueError("No dataframes provided")
        
        merged = pd.concat(dataframes, ignore_index=True)
        merged = self._validate_labels(merged)
        
        logger.info(f"Merged {len(dataframes)} datasets into {len(merged)} samples")
        return merged


if __name__ == "__main__":
    # Example usage
    loader = DatasetLoader()
    
    # Example: Load CSV
    # df = loader.load_csv('phishing_data.csv', url_column='URL', label_column='Label')
    
    # Example: Load JSON
    # df = loader.load_json('phishing_data.json')
    
    # Example: Auto-detect
    # df = loader.load_dataset('phishing_data.csv')
    
    print("Dataset Loader ready!")
