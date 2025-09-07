import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import joblib
import logging
from datetime import datetime

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# Model selection and evaluation
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, make_scorer, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
N_JOBS = -1
N_FOLDS = 5
TOP_FEATURES = 30

class ModelComparator:
    """A class to compare multiple machine learning models for dropout prediction."""
    
    def __init__(self, data_path: str, output_dir: str = 'model_comparison'):
        """Initialize the ModelComparator.
        
        Args:
            data_path: Path to the CSV file containing student data
            output_dir: Directory to save outputs (models, plots, etc.)
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.preprocessor = None
        
        # Results storage
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
        self.best_model_name = ""
        
        # Set up visualization style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = [10, 6]
        
        logger.info("ModelComparator initialized")
    
    def load_and_preprocess_data(self) -> None:
        """Load and preprocess the student data."""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Basic data validation
        if df.empty:
            raise ValueError("The dataset is empty")
            
        logger.info(f"Original dataset shape: {df.shape}")
        
        # Create target variable (dropout)
        attendance = df['Attendance'].str.extract(r'(\d+)').astype(float).squeeze()
        df['dropout'] = ((attendance < 60) | (df['Overall'] < 2.0)).astype(int)
        
        # Remove columns used in target definition
        columns_to_remove = ['Attendance', 'Overall', 'Last', 'Semester']
        df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
        
        # Split into features and target
        X = df.drop('dropout', axis=1)
        y = df['dropout']
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        
        logger.info(f"Training set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
        logger.info(f"Class distribution (train): {self.y_train.value_counts(normalize=True).to_dict()}")
        
        # Create preprocessing pipeline
        self._create_preprocessing_pipeline()
    
    def _create_preprocessing_pipeline(self) -> None:
        """Create the preprocessing pipeline for the data."""
        # Define categorical and numerical features
        categorical_features = [
            'Department', 'Gender', 'Hometown', 'Computer',
            'Preparation', 'Gaming', 'Job', 'English', 'Extra', 'Income'
        ]
        
        numerical_features = [
            'HSC', 'SSC'
        ]
        
        # Only keep features that exist in the dataframe
        categorical_features = [col for col in categorical_features if col in self.X_train.columns]
        numerical_features = [col for col in numerical_features if col in self.X_train.columns]
        
        # Create transformers
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        numerical_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
    
    def get_models(self) -> Dict[str, Any]:
        """Define and return a dictionary of models to compare."""
        # Calculate class weights for imbalanced data
        class_weights = len(self.y_train) / (2 * np.bincount(self.y_train))
        weight_ratio = class_weights[1] / class_weights[0]  # minority / majority
        
        models = {
            'Logistic Regression': Pipeline([
                ('preprocessor', self.preprocessor),
                ('feature_selector', SelectKBest(score_func=mutual_info_classif, k=TOP_FEATURES)),
                ('classifier', LogisticRegression(
                    class_weight='balanced',
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                    n_jobs=N_JOBS
                ))
            ]),
            
            'Random Forest': ImbPipeline([
                ('preprocessor', self.preprocessor),
                ('feature_selector', SelectKBest(score_func=mutual_info_classif, k=TOP_FEATURES)),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    class_weight='balanced_subsample',
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS
                ))
            ]),
            
            'XGBoost': ImbPipeline([
                ('preprocessor', self.preprocessor),
                ('feature_selector', SelectKBest(score_func=mutual_info_classif, k=TOP_FEATURES)),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('classifier', XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=weight_ratio,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                    eval_metric='logloss',
                    enable_categorical=True
                ))
            ]),
            
            'LightGBM': ImbPipeline([
                ('preprocessor', self.preprocessor),
                ('feature_selector', SelectKBest(score_func=mutual_info_classif, k=TOP_FEATURES)),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('classifier', LGBMClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=weight_ratio,
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                    verbose=-1
                ))
            ]),
            
            'CatBoost': ImbPipeline([
                ('preprocessor', self.preprocessor),
                ('feature_selector', SelectKBest(score_func=mutual_info_classif, k=TOP_FEATURES)),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('classifier', CatBoostClassifier(
                    iterations=200,
                    depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bylevel=0.8,
                    scale_pos_weight=weight_ratio,
                    random_seed=RANDOM_STATE,
                    thread_count=N_JOBS,
                    verbose=0
                ))
            ])
        }
        
        return models
    
    def evaluate_models(self) -> None:
        """Evaluate multiple models using cross-validation."""
        models = self.get_models()
        
        # Define metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision'
        }
        
        # Evaluate each model
        for name, model in models.items():
            logger.info(f"\nEvaluating {name}...")
            
            try:
                # Perform cross-validation
                cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                cv_results = cross_validate(
                    model, self.X_train, self.y_train,
                    cv=cv, scoring=scoring, n_jobs=1,  # n_jobs=1 to avoid memory issues
                    return_train_score=False, verbose=1
                )
                
                # Calculate mean and std of metrics
                metrics = {}
                for metric in scoring.keys():
                    scores = cv_results[f'test_{metric}']
                    metrics[f'{metric}_mean'] = np.mean(scores)
                    metrics[f'{metric}_std'] = np.std(scores)
                
                # Store results
                self.results[name] = metrics
                
                # Update best model
                if metrics['f1_mean'] > self.best_score:
                    self.best_score = metrics['f1_mean']
                    self.best_model_name = name
                    
                    # Fit the best model on full training data
                    logger.info(f"Fitting {name} on full training data...")
                    model.fit(self.X_train, self.y_train)
                    self.best_model = model
                
                logger.info(f"{name} - F1: {metrics['f1_mean']:.4f} (±{metrics['f1_std']:.4f})")
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
        
        # Save results to CSV
        self._save_results()
        
        # Plot comparison
        self._plot_model_comparison()
    
    def _save_results(self) -> None:
        """Save model comparison results to a CSV file."""
        if not self.results:
            logger.warning("No results to save.")
            return
        
        # Convert results to DataFrame
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Save to CSV
        results_file = self.output_dir / 'model_comparison_results.csv'
        results_df.to_csv(results_file)
        logger.info(f"Model comparison results saved to {results_file}")
    
    def _plot_model_comparison(self) -> None:
        """Create a bar plot comparing model performance."""
        if not self.results:
            logger.warning("No results to plot.")
            return
        
        # Prepare data for plotting
        metrics = ['f1_mean', 'precision_mean', 'recall_mean', 'roc_auc_mean']
        metric_names = ['F1 Score', 'Precision', 'Recall', 'ROC AUC']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            # Sort models by metric value
            sorted_results = sorted(
                [(name, values[metric]) for name, values in self.results.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            names = [x[0] for x in sorted_results]
            scores = [x[1] for x in sorted_results]
            
            # Plot
            ax = axes[i]
            y_pos = np.arange(len(names))
            bars = ax.barh(y_pos, scores, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel(metric_name)
            ax.set_title(f'Model Comparison - {metric_name}')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                       f'{width:.3f}',
                       va='center', ha='left')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.output_dir / 'model_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {plot_file}")
    
    def evaluate_best_model(self) -> Dict[str, float]:
        """Evaluate the best model on the test set."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet.")
        
        logger.info(f"\nEvaluating best model ({self.best_model_name}) on test set...")
        
        # Make predictions
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = None
        
        # Get predicted probabilities if available
        if hasattr(self.best_model, 'predict_proba'):
            y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
        }
        
        # Add probability-based metrics if available
        if y_pred_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'average_precision': average_precision_score(self.y_test, y_pred_proba)
            })
        
        # Log metrics
        logger.info("\nTest set evaluation:")
        for metric, score in metrics.items():
            logger.info(f"{metric}: {score:.4f}")
        
        # Save the best model
        self._save_best_model()
        
        return metrics
    
    def _save_best_model(self) -> None:
        """Save the best model to disk."""
        if self.best_model is None:
            logger.warning("No model to save.")
            return
        
        model_file = self.output_dir / 'best_model.pkl'
        joblib.dump(self.best_model, model_file)
        logger.info(f"Best model ({self.best_model_name}) saved to {model_file}")

def main():
    """Main function to run the model comparison."""
    try:
        # Initialize the model comparator
        comparator = ModelComparator(
            data_path='students.csv',
            output_dir='model_comparison'
        )
        
        # Load and preprocess data
        comparator.load_and_preprocess_data()
        
        # Evaluate models
        comparator.evaluate_models()
        
        # Evaluate the best model on the test set
        test_metrics = comparator.evaluate_best_model()
        
        logger.info("\nModel comparison completed successfully!")
        logger.info(f"Best model: {comparator.best_model_name}")
        logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
