"""
ML Model Trainer for Phishing Detection
Trains and evaluates various ML models for phishing detection
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
from typing import Dict, Tuple, List, Any
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class PhishingModelTrainer:
    """Train and evaluate ML models for phishing detection"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.metrics = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "is_phishing",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            test_size: Proportion of test set
            val_size: Proportion of validation set
            random_state: Random seed

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle missing values
        X = X.fillna(-1)

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp,
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Save feature names
        self.feature_names = list(X.columns)

        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Number of features: {X_train.shape[1]}")

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

    def train_random_forest(
        self, X_train: np.ndarray, y_train: np.ndarray, **kwargs
    ) -> RandomForestClassifier:
        """Train Random Forest model"""
        print("\n" + "=" * 50)
        print("Training Random Forest...")
        print("=" * 50)

        default_params = {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        default_params.update(kwargs)

        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)

        self.models["random_forest"] = model
        return model

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        print("\n" + "=" * 50)
        print("Training XGBoost...")
        print("=" * 50)

        default_params = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }
        default_params.update(kwargs)

        model = xgb.XGBClassifier(**default_params)

        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        self.models["xgboost"] = model
        return model

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ) -> lgb.LGBMClassifier:
        """Train LightGBM model"""
        print("\n" + "=" * 50)
        print("Training LightGBM...")
        print("=" * 50)

        default_params = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        default_params.update(kwargs)

        model = lgb.LGBMClassifier(**default_params)

        if X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="logloss",
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
        else:
            model.fit(X_train, y_train)

        self.models["lightgbm"] = model
        return model

    def train_logistic_regression(
        self, X_train: np.ndarray, y_train: np.ndarray, **kwargs
    ) -> LogisticRegression:
        """Train Logistic Regression model"""
        print("\n" + "=" * 50)
        print("Training Logistic Regression...")
        print("=" * 50)

        default_params = {
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        default_params.update(kwargs)

        model = LogisticRegression(**default_params)
        model.fit(X_train, y_train)

        self.models["logistic_regression"] = model
        return model

    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> SVC:
        """Train SVM model"""
        print("\n" + "=" * 50)
        print("Training SVM...")
        print("=" * 50)

        default_params = {
            "kernel": "rbf",
            "probability": True,
            "random_state": 42,
            "class_weight": "balanced",
        }
        default_params.update(kwargs)

        model = SVC(**default_params)
        model.fit(X_train, y_train)

        self.models["svm"] = model
        return model

    def train_neural_network(
        self, X_train: np.ndarray, y_train: np.ndarray, **kwargs
    ) -> MLPClassifier:
        """Train Neural Network model"""
        print("\n" + "=" * 50)
        print("Training Neural Network...")
        print("=" * 50)

        default_params = {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 500,
            "random_state": 42,
            "early_stopping": True,
        }
        default_params.update(kwargs)

        model = MLPClassifier(**default_params)
        model.fit(X_train, y_train)

        self.models["neural_network"] = model
        return model

    def train_ensemble(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> VotingClassifier:
        """Train ensemble of best models"""
        print("\n" + "=" * 50)
        print("Training Ensemble Model...")
        print("=" * 50)

        # Use top performing models
        estimators = []

        if "random_forest" in self.models:
            estimators.append(("rf", self.models["random_forest"]))
        if "xgboost" in self.models:
            estimators.append(("xgb", self.models["xgboost"]))
        if "lightgbm" in self.models:
            estimators.append(("lgb", self.models["lightgbm"]))

        if not estimators:
            # Train base models first
            self.train_random_forest(X_train, y_train)
            self.train_xgboost(X_train, y_train)
            self.train_lightgbm(X_train, y_train)

            estimators = [
                ("rf", self.models["random_forest"]),
                ("xgb", self.models["xgboost"]),
                ("lgb", self.models["lightgbm"]),
            ]

        ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        ensemble.fit(X_train, y_train)

        self.models["ensemble"] = ensemble
        return ensemble

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        models_to_train: List[str] = None,
    ) -> Dict[str, Any]:
        """Train all specified models"""
        if models_to_train is None:
            models_to_train = [
                "logistic_regression",
                "random_forest",
                "xgboost",
                "lightgbm",
                "neural_network",
                "ensemble",
            ]

        print("\n" + "=" * 70)
        print(f"Training {len(models_to_train)} models...")
        print("=" * 70)

        for model_name in models_to_train:
            try:
                if model_name == "random_forest":
                    self.train_random_forest(X_train, y_train)
                elif model_name == "xgboost":
                    self.train_xgboost(X_train, y_train, X_val, y_val)
                elif model_name == "lightgbm":
                    self.train_lightgbm(X_train, y_train, X_val, y_val)
                elif model_name == "logistic_regression":
                    self.train_logistic_regression(X_train, y_train)
                elif model_name == "svm":
                    self.train_svm(X_train, y_train)
                elif model_name == "neural_network":
                    self.train_neural_network(X_train, y_train)
                elif model_name == "ensemble":
                    self.train_ensemble(X_train, y_train)
            except Exception as e:
                print(f"Error training {model_name}: {e}")

        return self.models

    def evaluate_model(
        self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model"
    ) -> Dict[str, float]:
        """Evaluate a model and return metrics"""
        print(f"\nEvaluating {model_name}...")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        }

        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Calculate false positive and false negative rates
        tn, fp, fn, tp = cm.ravel()
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Print metrics
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        if "roc_auc" in metrics:
            print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {tn:5d}  FP: {fp:5d}")
        print(f"    FN: {fn:5d}  TP: {tp:5d}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")

        return metrics

    def evaluate_all_models(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models"""
        print("\n" + "=" * 70)
        print("Evaluating All Models")
        print("=" * 70)

        all_metrics = {}

        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            all_metrics[model_name] = metrics

        self.metrics = all_metrics

        # Find best model based on F1 score
        best_model_name = max(
            all_metrics.keys(), key=lambda k: all_metrics[k]["f1_score"]
        )
        self.best_model = self.models[best_model_name]

        print("\n" + "=" * 70)
        print(f"Best Model: {best_model_name}")
        print(f"F1 Score: {all_metrics[best_model_name]['f1_score']:.4f}")
        print("=" * 70)

        return all_metrics

    def get_feature_importance(
        self, model_name: str = None, top_n: int = 20
    ) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        if model_name is None:
            # Use best model
            model_name = max(
                self.metrics.keys(), key=lambda k: self.metrics[k]["f1_score"]
            )

        model = self.models.get(model_name)

        if model is None:
            print(f"Model {model_name} not found")
            return None

        # Extract feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            print(f"Model {model_name} does not support feature importance")
            return None

        # Create DataFrame
        feature_importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        self.feature_importance = feature_importance_df

        print(f"\nTop {top_n} Most Important Features ({model_name}):")
        print("=" * 60)
        for idx, row in feature_importance_df.head(top_n).iterrows():
            print(f"{row['feature']:40s}: {row['importance']:.6f}")

        return feature_importance_df

    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = "random_forest",
        param_grid: Dict = None,
        cv: int = 5,
    ) -> Tuple[Any, Dict]:
        """Perform hyperparameter tuning using GridSearchCV"""
        print(f"\n{'=' * 70}")
        print(f"Hyperparameter Tuning for {model_type}")
        print(f"{'=' * 70}")

        # Default parameter grids
        default_param_grids = {
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "xgboost": {
                "n_estimators": [100, 200],
                "max_depth": [6, 8, 10],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 1.0],
            },
            "lightgbm": {
                "n_estimators": [100, 200],
                "max_depth": [6, 8, 10],
                "learning_rate": [0.01, 0.1, 0.3],
                "num_leaves": [31, 50, 70],
            },
        }

        if param_grid is None:
            param_grid = default_param_grids.get(model_type, {})

        # Create base model
        if model_type == "random_forest":
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_type == "xgboost":
            base_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        elif model_type == "lightgbm":
            base_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
        else:
            print(f"Model type {model_type} not supported for tuning")
            return None, {}

        # Perform grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best F1 Score: {grid_search.best_score_:.4f}")

        # Update model
        self.models[f"{model_type}_tuned"] = grid_search.best_estimator_

        return grid_search.best_estimator_, grid_search.best_params_

    def save_model(self, model_name: str = None, filename: str = None):
        """Save model to disk"""
        if model_name is None:
            # Save best model
            model_name = max(
                self.metrics.keys(), key=lambda k: self.metrics[k]["f1_score"]
            )

        model = self.models.get(model_name)

        if model is None:
            print(f"Model {model_name} not found")
            return

        if filename is None:
            filename = f"{model_name}_model.pkl"

        filepath = self.models_dir / filename

        # Save model
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")

        # Save scaler
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

        # Save feature names
        feature_path = self.models_dir / f"{model_name}_features.json"
        with open(feature_path, "w") as f:
            json.dump(self.feature_names, f)
        print(f"Feature names saved to {feature_path}")

        # Save metrics
        metrics_path = self.models_dir / f"{model_name}_metrics.json"
        with open(metrics_path, "w") as f:
            # Convert numpy types to Python types
            metrics_serializable = {}
            for k, v in self.metrics[model_name].items():
                if k == "confusion_matrix":
                    metrics_serializable[k] = v
                else:
                    metrics_serializable[k] = (
                        float(v) if isinstance(v, (np.floating, np.integer)) else v
                    )
            json.dump(metrics_serializable, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    def load_model(self, model_name: str, filename: str = None):
        """Load model from disk"""
        if filename is None:
            filename = f"{model_name}_model.pkl"

        filepath = self.models_dir / filename

        if not filepath.exists():
            print(f"Model file {filepath} not found")
            return None

        # Load model
        model = joblib.load(filepath)
        self.models[model_name] = model
        print(f"Model loaded from {filepath}")

        # Load scaler
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")

        # Load feature names
        feature_path = self.models_dir / f"{model_name}_features.json"
        if feature_path.exists():
            with open(feature_path, "r") as f:
                self.feature_names = json.load(f)
            print(f"Feature names loaded from {feature_path}")

        return model

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "random_forest",
        cv: int = 5,
    ) -> Dict[str, float]:
        """Perform cross-validation"""
        print(f"\n{'=' * 70}")
        print(f"Cross-Validation for {model_name} (CV={cv})")
        print(f"{'=' * 70}")

        model = self.models.get(model_name)

        if model is None:
            print(f"Model {model_name} not trained yet")
            return {}

        # Perform cross-validation
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        results = {}

        for score in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=score, n_jobs=-1)
            results[f"{score}_mean"] = scores.mean()
            results[f"{score}_std"] = scores.std()
            print(f"{score:12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

        return results


if __name__ == "__main__":
    # Example usage
    print("Phishing Model Trainer")
    print("=" * 70)

    # This would typically load real data
    # For demonstration, create synthetic data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=10000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42,
    )

    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["is_phishing"] = y

    # Initialize trainer
    trainer = PhishingModelTrainer()

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)

    # Train models
    trainer.train_all_models(X_train, y_train, X_val, y_val)

    # Evaluate models
    metrics = trainer.evaluate_all_models(X_test, y_test)

    # Get feature importance
    trainer.get_feature_importance()

    # Save best model
    trainer.save_model()

    print("\nTraining complete!")
