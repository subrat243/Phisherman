#!/usr/bin/env python3
"""
Reinforcement Learning Trainer
Implements online learning and feedback-based model improvement
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class ReinforcementTrainer:
    """Reinforcement learning trainer for continuous model improvement"""
    
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        feedback_dir: str = "data/feedback",
        checkpoint_dir: str = "models/checkpoints"
    ):
        """
        Initialize reinforcement trainer
        
        Args:
            model_path: Path to base model
            scaler_path: Path to feature scaler
            feedback_dir: Directory for feedback storage
            checkpoint_dir: Directory for model checkpoints
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.feedback_dir = Path(feedback_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create directories
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and scaler
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        # Feedback buffer
        self.feedback_buffer = []
        self.feedback_file = self.feedback_dir / "feedback_log.json"
        
        # Load existing feedback
        self._load_feedback()
        
        # Reward configuration
        self.rewards = {
            'correct_detection': 1.0,
            'false_positive': -0.5,
            'false_negative': -1.0,
            'user_confirmation': 0.5
        }
        
        # Performance tracking
        self.performance_history = []
        self.performance_file = self.feedback_dir / "performance_history.json"
        self._load_performance_history()
    
    def _load_feedback(self):
        """Load existing feedback from file"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    self.feedback_buffer = json.load(f)
                logger.info(f"Loaded {len(self.feedback_buffer)} feedback entries")
            except Exception as e:
                logger.error(f"Error loading feedback: {e}")
                self.feedback_buffer = []
    
    def _save_feedback(self):
        """Save feedback to file"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_buffer, f, indent=2)
            logger.info(f"Saved {len(self.feedback_buffer)} feedback entries")
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def _load_performance_history(self):
        """Load performance history"""
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    self.performance_history = json.load(f)
                logger.info(f"Loaded {len(self.performance_history)} performance records")
            except Exception as e:
                logger.error(f"Error loading performance history: {e}")
                self.performance_history = []
    
    def _save_performance_history(self):
        """Save performance history"""
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
    
    def add_feedback(
        self,
        url: str,
        features: Dict[str, float],
        predicted_label: int,
        predicted_prob: float,
        actual_label: Optional[int] = None,
        user_feedback: Optional[str] = None
    ):
        """
        Add feedback for a prediction
        
        Args:
            url: The URL that was analyzed
            features: Extracted features
            predicted_label: Model's prediction (0=legitimate, 1=phishing)
            predicted_prob: Prediction probability
            actual_label: True label if known
            user_feedback: User's feedback ('correct', 'incorrect', 'unsure')
        """
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'url': url,
            'features': features,
            'predicted_label': int(predicted_label),
            'predicted_prob': float(predicted_prob),
            'actual_label': int(actual_label) if actual_label is not None else None,
            'user_feedback': user_feedback,
            'reward': self._calculate_reward(predicted_label, actual_label, user_feedback)
        }
        
        self.feedback_buffer.append(feedback_entry)
        self._save_feedback()
        
        logger.info(f"Added feedback for {url}: reward={feedback_entry['reward']}")
    
    def _calculate_reward(
        self,
        predicted: int,
        actual: Optional[int],
        user_feedback: Optional[str]
    ) -> float:
        """
        Calculate reward based on prediction accuracy
        
        Args:
            predicted: Predicted label
            actual: Actual label if known
            user_feedback: User feedback
            
        Returns:
            Reward value
        """
        # If we have actual label, use it
        if actual is not None:
            if predicted == actual:
                return self.rewards['correct_detection']
            else:
                if predicted == 1:  # False positive
                    return self.rewards['false_positive']
                else:  # False negative
                    return self.rewards['false_negative']
        
        # If we have user feedback
        if user_feedback:
            if user_feedback == 'correct':
                return self.rewards['user_confirmation']
            elif user_feedback == 'incorrect':
                if predicted == 1:
                    return self.rewards['false_positive']
                else:
                    return self.rewards['false_negative']
        
        # No feedback available
        return 0.0
    
    def get_training_samples(self, min_samples: int = 10) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get training samples from feedback buffer
        
        Args:
            min_samples: Minimum number of samples required
            
        Returns:
            (X, y) tuple or None if insufficient samples
        """
        # Filter feedback with actual labels
        labeled_feedback = [
            f for f in self.feedback_buffer 
            if f['actual_label'] is not None
        ]
        
        if len(labeled_feedback) < min_samples:
            logger.warning(f"Insufficient labeled samples: {len(labeled_feedback)}/{min_samples}")
            return None
        
        # Extract features and labels
        X = []
        y = []
        
        for feedback in labeled_feedback:
            features = feedback['features']
            # Convert features dict to array (maintain order)
            feature_array = [features.get(f, -1) for f in sorted(features.keys())]
            X.append(feature_array)
            y.append(feedback['actual_label'])
        
        return np.array(X), np.array(y)
    
    def update_model(self, batch_size: int = 100, learning_rate: float = 0.1):
        """
        Update model with feedback
        
        Args:
            batch_size: Number of samples to use for update
            learning_rate: Learning rate for update
        """
        logger.info("Updating model with feedback...")
        
        # Get training samples
        training_data = self.get_training_samples(min_samples=batch_size)
        
        if training_data is None:
            logger.warning("Not enough feedback for model update")
            return False
        
        X, y = training_data
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Evaluate current performance
        y_pred_before = self.model.predict(X_scaled)
        acc_before = accuracy_score(y, y_pred_before)
        
        # Partial fit (online learning)
        try:
            if hasattr(self.model, 'partial_fit'):
                self.model.partial_fit(X_scaled, y)
            else:
                # For models without partial_fit, retrain on combined data
                self.model.fit(X_scaled, y)
            
            # Evaluate new performance
            y_pred_after = self.model.predict(X_scaled)
            acc_after = accuracy_score(y, y_pred_after)
            
            # Record performance
            performance_record = {
                'timestamp': datetime.now().isoformat(),
                'samples_used': len(X),
                'accuracy_before': float(acc_before),
                'accuracy_after': float(acc_after),
                'improvement': float(acc_after - acc_before)
            }
            
            self.performance_history.append(performance_record)
            self._save_performance_history()
            
            logger.info(f"Model updated: {acc_before:.3f} → {acc_after:.3f}")
            
            # Save updated model if improvement
            if acc_after >= acc_before:
                self.save_checkpoint()
                return True
            else:
                logger.warning("Model performance decreased, not saving")
                return False
                
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            return False
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"model_checkpoint_{timestamp}.pkl"
        
        try:
            joblib.dump(self.model, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Also update main model
            joblib.dump(self.model, self.model_path)
            logger.info(f"Updated main model at {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def rollback_to_checkpoint(self, checkpoint_path: str):
        """
        Rollback to a previous checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            self.model = joblib.load(checkpoint_path)
            logger.info(f"Rolled back to checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error rolling back: {e}")
    
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        stats = {
            'total_feedback': len(self.feedback_buffer),
            'labeled_feedback': len([f for f in self.feedback_buffer if f['actual_label'] is not None]),
            'average_reward': np.mean([f['reward'] for f in self.feedback_buffer]) if self.feedback_buffer else 0,
            'performance_history': self.performance_history[-10:],  # Last 10 updates
        }
        
        return stats
    
    def clear_feedback(self, keep_last_n: int = 1000):
        """
        Clear old feedback, keeping only recent entries
        
        Args:
            keep_last_n: Number of recent entries to keep
        """
        if len(self.feedback_buffer) > keep_last_n:
            self.feedback_buffer = self.feedback_buffer[-keep_last_n:]
            self._save_feedback()
            logger.info(f"Cleared old feedback, kept last {keep_last_n} entries")


if __name__ == "__main__":
    # Example usage
    print("Reinforcement Learning Trainer ready!")
    
    # Example: Initialize trainer
    # trainer = ReinforcementTrainer(
    #     model_path="models/best_model.pkl",
    #     scaler_path="models/random_forest_scaler.pkl"
    # )
    
    # Example: Add feedback
    # trainer.add_feedback(
    #     url="http://example.com",
    #     features={'url_length': 20, 'has_ip': 0},
    #     predicted_label=0,
    #     predicted_prob=0.95,
    #     actual_label=0,
    #     user_feedback='correct'
    # )
    
    # Example: Update model
    # trainer.update_model(batch_size=50)
