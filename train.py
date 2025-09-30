"""
Complete Training Pipeline for Phishing Detection Models
Handles data collection, feature extraction, model training, and evaluation
"""

import sys
from pathlib import Path
import pandas as pd
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collector import PhishingDataCollector
from model_trainer import PhishingModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train phishing detection models")

    parser.add_argument(
        "--collect-data",
        action="store_true",
        help="Collect and prepare new training data",
    )

    parser.add_argument(
        "--data-file",
        type=str,
        default="data/processed/training_data_balanced.csv",
        help="Path to training data CSV file",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["random_forest", "xgboost", "lightgbm", "ensemble"],
        choices=[
            "random_forest",
            "xgboost",
            "lightgbm",
            "logistic_regression",
            "svm",
            "neural_network",
            "ensemble",
        ],
        help="Models to train",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning",
    )

    parser.add_argument(
        "--tune-model",
        type=str,
        choices=["random_forest", "xgboost", "lightgbm"],
        default="random_forest",
        help="Model to tune",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (0-1)",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set size (0-1)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained models",
    )

    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Perform cross-validation",
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )

    return parser.parse_args()


def collect_and_prepare_data():
    """Collect and prepare training data"""
    print("\n" + "=" * 70)
    print("DATA COLLECTION AND PREPARATION")
    print("=" * 70)

    collector = PhishingDataCollector()

    # Collect raw URLs
    print("\n[1/4] Collecting URLs from sources...")
    raw_df = collector.combine_datasets()

    if raw_df.empty:
        logger.error("Failed to collect data")
        return None

    # Extract features
    print("\n[2/4] Extracting features from URLs...")
    processed_df = collector.prepare_training_data(raw_df, max_workers=10)

    if processed_df.empty:
        logger.error("Failed to extract features")
        return None

    # Balance dataset
    print("\n[3/4] Balancing dataset...")
    balanced_df = collector.balance_dataset(processed_df, method="undersample")

    # Save balanced dataset
    balanced_file = collector.processed_dir / "training_data_balanced.csv"
    balanced_df.to_csv(balanced_file, index=False)
    print(f"✓ Saved balanced dataset to {balanced_file}")

    # Get and save statistics
    print("\n[4/4] Computing dataset statistics...")
    stats = collector.get_data_statistics(balanced_df)

    stats_file = collector.processed_dir / "dataset_stats.json"
    import json

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset Summary:")
    print(f"  Total Samples: {stats['total_samples']}")
    print(f"  Features: {stats['num_features']}")
    print(f"  Phishing: {stats['phishing_samples']}")
    print(f"  Legitimate: {stats['legitimate_samples']}")
    print(
        f"  Balance Ratio: {stats['phishing_samples'] / stats['legitimate_samples']:.2f}"
    )

    return balanced_df


def train_models(args):
    """Train phishing detection models"""
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    # Load or prepare data
    if args.collect_data:
        df = collect_and_prepare_data()
        if df is None:
            logger.error("Data preparation failed")
            return
    else:
        data_file = Path(args.data_file)
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            logger.info("Run with --collect-data to prepare training data")
            return

        print(f"\nLoading training data from {data_file}...")
        df = pd.read_csv(data_file)
        print(f"✓ Loaded {len(df)} samples with {df.shape[1]} features")

    # Initialize trainer
    trainer = PhishingModelTrainer(models_dir=args.output_dir)

    # Prepare data
    print("\n" + "=" * 70)
    print("PREPARING DATA FOR TRAINING")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        df,
        target_col="is_phishing",
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=42,
    )

    # Hyperparameter tuning (optional)
    if args.tune:
        print("\n" + "=" * 70)
        print("HYPERPARAMETER TUNING")
        print("=" * 70)

        print(f"\nTuning {args.tune_model}...")
        best_model, best_params = trainer.hyperparameter_tuning(
            X_train,
            y_train,
            model_type=args.tune_model,
            cv=args.cv_folds,
        )

        print(f"\n✓ Best parameters found:")
        for param, value in best_params.items():
            print(f"    {param}: {value}")

    # Train models
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    trainer.train_all_models(
        X_train,
        y_train,
        X_val,
        y_val,
        models_to_train=args.models,
    )

    # Evaluate models
    print("\n" + "=" * 70)
    print("EVALUATING MODELS")
    print("=" * 70)

    metrics = trainer.evaluate_all_models(X_test, y_test)

    # Print summary table
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC AUC':<10}"
    )
    print("-" * 70)

    for model_name, model_metrics in metrics.items():
        print(
            f"{model_name:<20} "
            f"{model_metrics['accuracy']:<10.4f} "
            f"{model_metrics['precision']:<10.4f} "
            f"{model_metrics['recall']:<10.4f} "
            f"{model_metrics['f1_score']:<10.4f} "
            f"{model_metrics.get('roc_auc', 0):<10.4f}"
        )

    # Get feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    trainer.get_feature_importance(top_n=20)

    # Cross-validation (optional)
    if args.cross_validate:
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION")
        print("=" * 70)

        # Combine train and val for CV
        X_combined = pd.concat(
            [
                pd.DataFrame(X_train, columns=trainer.feature_names),
                pd.DataFrame(X_val, columns=trainer.feature_names),
            ]
        )
        y_combined = pd.concat([pd.Series(y_train), pd.Series(y_val)])

        best_model_name = max(metrics.keys(), key=lambda k: metrics[k]["f1_score"])

        cv_results = trainer.cross_validate(
            X_combined.values,
            y_combined.values,
            model_name=best_model_name,
            cv=args.cv_folds,
        )

    # Save best model
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)

    # Find best model
    best_model_name = max(metrics.keys(), key=lambda k: metrics[k]["f1_score"])
    print(f"\nBest Model: {best_model_name}")
    print(f"F1 Score: {metrics[best_model_name]['f1_score']:.4f}")

    # Save best model
    trainer.save_model(model_name=best_model_name, filename="best_model.pkl")

    # Also save all individual models
    print("\nSaving all trained models...")
    for model_name in trainer.models.keys():
        trainer.save_model(model_name=model_name)

    # Generate training report
    print("\n" + "=" * 70)
    print("GENERATING TRAINING REPORT")
    print("=" * 70)

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "num_features": X_train.shape[1],
        },
        "models_trained": list(trainer.models.keys()),
        "best_model": best_model_name,
        "metrics": {
            model_name: {
                k: float(v) if not isinstance(v, list) else v
                for k, v in model_metrics.items()
            }
            for model_name, model_metrics in metrics.items()
        },
        "hyperparameter_tuning": args.tune,
        "cross_validation": args.cross_validate,
    }

    # Save report
    report_file = Path(args.output_dir) / "training_report.json"
    import json

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Training report saved to {report_file}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    print(f"\nModel files saved in: {args.output_dir}/")
    print(f"Best model: {args.output_dir}/best_model.pkl")
    print(f"Scaler: {args.output_dir}/best_model_scaler.pkl")
    print(f"Features: {args.output_dir}/best_model_features.json")
    print(f"\nTo use the model:")
    print(
        f"  1. Load in Python: detector.load_model('{args.output_dir}/best_model.pkl')"
    )
    print(f"  2. Start API: python api/main.py")
    print(
        f'  3. Test: curl -X POST http://localhost:8000/api/v1/analyze/url -d \'{{"url":"https://example.com"}}\''
    )


def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print("PHISHING DETECTION MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    args = parse_arguments()

    print(f"\nConfiguration:")
    print(f"  Data collection: {args.collect_data}")
    print(f"  Data file: {args.data_file}")
    print(f"  Models to train: {', '.join(args.models)}")
    print(f"  Hyperparameter tuning: {args.tune}")
    print(f"  Cross-validation: {args.cross_validate}")
    print(f"  Test size: {args.test_size}")
    print(f"  Validation size: {args.val_size}")
    print(f"  Output directory: {args.output_dir}")

    try:
        train_models(args)

        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print("\n" + "=" * 70)
        print("TRAINING FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
