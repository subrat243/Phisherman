"""
Enhanced Training Script with Advanced Features
- Automated data collection from free sources
- Dataset import support
- Reinforcement learning
- Online training
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collector import PhishingDataCollector
from dataset_loader import DatasetLoader
from model_trainer import PhishingModelTrainer
from reinforcement_trainer import ReinforcementTrainer
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Phisherman - Advanced ML Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data from free sources
  python train.py --collect-data --source all --max-samples 1000
  
  # Import existing dataset
  python train.py --dataset phishing_data.csv --dataset-format csv
  
  # Train with reinforcement learning
  python train.py --enable-rl --feedback-file data/feedback/feedback_log.json
  
  # Full training pipeline
  python train.py --collect-data --source all --models ensemble --enable-rl
        """
    )
    
    # Data Collection
    data_group = parser.add_argument_group('Data Collection')
    data_group.add_argument(
        '--collect-data',
        action='store_true',
        help='Collect data from free sources'
    )
    data_group.add_argument(
        '--source',
        choices=['phishtank', 'openphish', 'legitimate', 'all'],
        default='all',
        help='Data source to collect from'
    )
    data_group.add_argument(
        '--max-samples',
        type=int,
        default=1000,
        help='Maximum samples per source'
    )
    
    # Dataset Import
    dataset_group = parser.add_argument_group('Dataset Import')
    dataset_group.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset file'
    )
    dataset_group.add_argument(
        '--dataset-format',
        choices=['auto', 'csv', 'json', 'uci'],
        default='auto',
        help='Dataset format'
    )
    dataset_group.add_argument(
        '--url-column',
        type=str,
        default='url',
        help='Name of URL column in dataset'
    )
    dataset_group.add_argument(
        '--label-column',
        type=str,
        default='label',
        help='Name of label column in dataset'
    )
    
    # Model Training
    training_group = parser.add_argument_group('Model Training')
    training_group.add_argument(
        '--models',
        nargs='+',
        choices=['random_forest', 'xgboost', 'lightgbm', 'ensemble', 'all'],
        default=['ensemble'],
        help='Models to train'
    )
    training_group.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning'
    )
    training_group.add_argument(
        '--cross-validate',
        action='store_true',
        help='Perform cross-validation'
    )
    training_group.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    # Reinforcement Learning
    rl_group = parser.add_argument_group('Reinforcement Learning')
    rl_group.add_argument(
        '--enable-rl',
        action='store_true',
        help='Enable reinforcement learning'
    )
    rl_group.add_argument(
        '--feedback-file',
        type=str,
        default='data/feedback/feedback_log.json',
        help='Path to feedback file'
    )
    rl_group.add_argument(
        '--online-learning',
        action='store_true',
        help='Enable online learning mode'
    )
    rl_group.add_argument(
        '--update-interval',
        type=int,
        default=100,
        help='Number of samples before model update'
    )
    
    # General Options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for models'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def collect_data_from_sources(args):
    """Collect data from free sources"""
    logger.info("="*70)
    logger.info("Collecting Data from Free Sources")
    logger.info("="*70)
    
    collector = PhishingDataCollector()
    datasets = []
    
    if args.source in ['phishtank', 'all']:
        logger.info("Collecting from PhishTank...")
        df = collector.download_phishtank_data(limit=args.max_samples)
        if not df.empty:
            datasets.append(df)
    
    if args.source in ['openphish', 'all']:
        logger.info("Collecting from OpenPhish...")
        df = collector.download_openphish_data(limit=args.max_samples)
        if not df.empty:
            datasets.append(df)
    
    if args.source in ['legitimate', 'all']:
        logger.info("Collecting legitimate URLs...")
        df = collector.get_legitimate_urls(limit=args.max_samples)
        if not df.empty:
            datasets.append(df)
    
    if datasets:
        combined_df = pd.concat(datasets, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['url'])
        logger.info(f"Collected {len(combined_df)} total URLs")
        logger.info(f"  Phishing: {sum(combined_df['label'] == 1)}")
        logger.info(f"  Legitimate: {sum(combined_df['label'] == 0)}")
        return combined_df
    
    return None


def import_dataset(args):
    """Import dataset from file"""
    logger.info("="*70)
    logger.info("Importing Dataset")
    logger.info("="*70)
    
    loader = DatasetLoader()
    
    try:
        df = loader.load_dataset(
            path=args.dataset,
            format=args.dataset_format,
            url_column=args.url_column,
            label_column=args.label_column
        )
        
        logger.info(f"Imported {len(df)} samples")
        logger.info(f"  Phishing: {sum(df['is_phishing'] == 1)}")
        logger.info(f"  Legitimate: {sum(df['is_phishing'] == 0)}")
        
        # Rename column to match expected format
        df = df.rename(columns={'is_phishing': 'label'})
        
        return df
        
    except Exception as e:
        logger.error(f"Error importing dataset: {e}")
        return None


def train_models(df, args):
    """Train ML models"""
    logger.info("="*70)
    logger.info("Training ML Models")
    logger.info("="*70)
    
    # Prepare data
    collector = PhishingDataCollector()
    processed_df = collector.prepare_training_data(df, max_workers=5)
    
    if processed_df.empty:
        logger.error("No data available for training")
        return None
    
    # Balance dataset
    balanced_df = collector.balance_dataset(processed_df, method='undersample')
    
    # Initialize trainer
    trainer = PhishingModelTrainer()
    
    # Prepare train/val/test splits
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        balanced_df,
        test_size=0.2,
        val_size=0.1
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Determine models to train
    models_to_train = args.models
    if 'all' in models_to_train:
        models_to_train = ['random_forest', 'xgboost', 'lightgbm', 'ensemble']
    
    # Train models
    logger.info(f"Training models: {', '.join(models_to_train)}")
    trainer.train_all_models(
        X_train, y_train,
        X_val, y_val,
        models_to_train=models_to_train
    )
    
    # Evaluate
    logger.info("\nEvaluating models...")
    results = trainer.evaluate_all_models(X_test, y_test)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"\n🏆 Best Model: {best_model[0]}")
    logger.info(f"   Accuracy: {best_model[1]['accuracy']:.2%}")
    
    # Save best model
    trainer.save_model(best_model[0], filename='best_model.pkl')
    
    return trainer, best_model[0]


def setup_reinforcement_learning(args, model_path, scaler_path):
    """Setup reinforcement learning"""
    logger.info("="*70)
    logger.info("Setting up Reinforcement Learning")
    logger.info("="*70)
    
    try:
        rl_trainer = ReinforcementTrainer(
            model_path=model_path,
            scaler_path=scaler_path,
            feedback_dir='data/feedback',
            checkpoint_dir='models/checkpoints'
        )
        
        stats = rl_trainer.get_statistics()
        logger.info(f"Total feedback entries: {stats['total_feedback']}")
        logger.info(f"Labeled feedback: {stats['labeled_feedback']}")
        logger.info(f"Average reward: {stats['average_reward']:.3f}")
        
        if args.online_learning and stats['labeled_feedback'] >= args.update_interval:
            logger.info(f"\nUpdating model with feedback...")
            success = rl_trainer.update_model(batch_size=args.update_interval)
            if success:
                logger.info("✓ Model updated successfully")
            else:
                logger.warning("⚠ Model update failed or no improvement")
        
        return rl_trainer
        
    except Exception as e:
        logger.error(f"Error setting up RL: {e}")
        return None


def main():
    """Main training pipeline"""
    args = parse_arguments()
    
    print("="*70)
    print("🛡️  Phisherman - Advanced ML Training")
    print("="*70)
    print()
    
    # Step 1: Get training data
    df = None
    
    if args.collect_data:
        df = collect_data_from_sources(args)
    elif args.dataset:
        df = import_dataset(args)
    else:
        logger.error("No data source specified. Use --collect-data or --dataset")
        return
    
    if df is None or df.empty:
        logger.error("No data available for training")
        return
    
    # Step 2: Train models
    result = train_models(df, args)
    if result is None:
        logger.error("Training failed")
        return
    
    trainer, best_model_name = result
    
    # Step 3: Reinforcement Learning (if enabled)
    if args.enable_rl:
        model_path = f"models/best_model.pkl"
        scaler_path = f"models/{best_model_name}_scaler.pkl"
        
        rl_trainer = setup_reinforcement_learning(args, model_path, scaler_path)
    
    # Summary
    print()
    print("="*70)
    print("✅ Training Complete!")
    print("="*70)
    print()
    print("Next Steps:")
    print("  1. Test the model:")
    print('     python detect.py -m "models/best_model.pkl" \\')
    print(f'                      -s "models/{best_model_name}_scaler.pkl" \\')
    print(f'                      --features "models/{best_model_name}_features.json" \\')
    print('                      -u "https://example.com"')
    print()
    print("  2. Start API server:")
    print("     python api/main.py")
    print()
    if args.enable_rl:
        print("  3. Reinforcement learning is enabled!")
        print("     Feedback will be collected automatically")
        print()
    print("="*70)


if __name__ == "__main__":
    main()
