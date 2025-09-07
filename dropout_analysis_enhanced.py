import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional

# Model and preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel, RFE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from skopt.space import Real, Integer, Categorical
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dropout_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
N_JOBS = -1  # Use all available cores
N_FOLDS = 5
TOP_FEATURES = 30
OPTIMIZE_THRESHOLD = True  # Whether to optimize decision threshold
TUNE_HYPERPARAMETERS = True  # Whether to perform hyperparameter tuning

class DropoutPredictor:
    """A class for predicting student dropout risk with enhanced features and evaluation."""
    
    def __init__(self, data_path: str, output_dir: str = 'output'):
        """Initialize the DropoutPredictor.
        
        Args:
            data_path: Path to the CSV file containing student data
            output_dir: Directory to save outputs (models, plots, etc.)
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store data and models
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.model = None
        self.preprocessor = None
        
        # Set up visualization style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = [10, 6]
        
        logger.info("DropoutPredictor initialized")
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the student data."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # Basic data validation
        if self.df.empty:
            raise ValueError("The dataset is empty")
            
        logger.info(f"Original dataset shape: {self.df.shape}")
        
        # Create target variable (dropout)
        self._create_target()
        
        # Remove columns that could cause data leakage
        self._remove_leaky_columns()
        
        # Basic feature engineering
        self._engineer_features()
        
        logger.info("Data preprocessing completed")
        return self.df
    
    def _create_target(self) -> None:
        """Create the target variable for dropout prediction."""
        # Define dropout based on attendance and overall performance
        attendance = self.df['Attendance'].str.extract(r'(\d+)').astype(float).squeeze()
        self.df['dropout'] = ((attendance < 60) | (self.df['Overall'] < 2.0)).astype(int)
        
        # Log class distribution
        class_dist = self.df['dropout'].value_counts(normalize=True)
        logger.info(f"Class distribution:\n{class_dist}")
    
    def _remove_leaky_columns(self) -> None:
        """Remove columns that could cause data leakage."""
        # Columns to remove (either directly used in target or not available at prediction time)
        columns_to_remove = [
            'Attendance',  # Used in target definition
            'Overall',     # Used in target definition
            'Last',        # Future information
            'Semester'     # Could be correlated with time to dropout
        ]
        
        self.df = self.df.drop(columns=[col for col in columns_to_remove if col in self.df.columns])
        logger.info(f"Removed potentially leaky columns: {columns_to_remove}")
    
    def _engineer_features(self) -> None:
        """Create new features that might be predictive of dropout."""
        # Convert categorical columns to appropriate types
        categorical_cols = [
            'Department', 'Gender', 'Hometown', 'Computer', 
            'Preparation', 'Gaming', 'Job', 'English', 'Extra', 'Income'
        ]
        
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
        
        # Create interaction terms or other engineered features
        if 'HSC' in self.df.columns and 'SSC' in self.df.columns:
            self.df['HSC_SSC_ratio'] = self.df['HSC'] / self.df['SSC']
            
        logger.info("Feature engineering completed")
    
    def split_data(self, test_size: float = 0.2) -> None:
        """Split data into training and testing sets."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        X = self.df.drop('dropout', axis=1)
        y = self.df['dropout']
        
        # Stratified split to maintain class distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
        )
        
        logger.info(f"Data split into training ({len(self.X_train)}) and test ({len(self.X_test)}) sets")
    
    def _create_pipeline(self, use_smote: bool = True) -> Pipeline:
        """Create the enhanced model pipeline with improved preprocessing and feature engineering."""
        # Define preprocessing for numerical and categorical features
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Enhanced numeric transformations
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False))
        ])
        
        # Enhanced categorical encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('target', TargetEncoder(min_samples_leaf=20)),  # Using category_encoders.TargetEncoder
            ('scaler', StandardScaler())
        ])
        
        # Create feature union for numeric and categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop',
            n_jobs=-1
        )
        
        # Enhanced model with feature selection and class balancing
        if use_smote:
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('feature_selection', SelectFromModel(
                    estimator=RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        class_weight='balanced'
                    ),
                    threshold='1.25*mean'  # Slightly more features
                )),
                ('classifier', RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced_subsample',
                    bootstrap=True,
                    oob_score=True
                ))
            ])
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selection', SelectFromModel(
                    estimator=RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        class_weight='balanced_subsample'
                    ),
                    threshold='1.25*mean'
                )),
                ('classifier', RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced_subsample',
                    bootstrap=True,
                    oob_score=True
                ))
            ])
            
        return pipeline
    
    def train_model(self, use_smote: bool = True) -> Dict[str, Any]:
        """Train the enhanced model with improved cross-validation and hyperparameter tuning."""
        logger.info("Training enhanced model with improved feature engineering and tuning...")
        
        # Create enhanced pipeline
        pipeline = self._create_pipeline(use_smote)
        
        # Define search spaces for BayesSearchCV
        search_spaces = [
            {
                'classifier__n_estimators': (100, 300),  # Integer range
                'classifier__max_depth': (3, 15),  # Integer range
                'classifier__min_samples_split': (2, 20),  # Integer range
                'classifier__min_samples_leaf': (1, 10),  # Integer range
                'classifier__max_features': Categorical(['sqrt', 'log2', 0.7]),  # Categorical
                'classifier__max_leaf_nodes': (10, 100),  # Integer range
                'classifier__min_impurity_decrease': (0.0, 0.2),  # Float range
                'classifier__ccp_alpha': (0.0, 0.1),  # Float range
                'smote__sampling_strategy': Categorical([0.5, 0.6, 0.7]),  # Fixed values
                'smote__k_neighbors': Categorical([3, 5, 7])  # Fixed values
            }
        ]
        
        # Use Bayesian Optimization with BayesSearchCV for better hyperparameter search
        search = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_spaces,
            n_iter=30,  # Number of parameter settings to sample
            cv=5,  # Number of cross-validation folds
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            random_state=42,
            n_points=1,  # Number of parameter settings to sample in parallel
            return_train_score=True
        )
        
        # Set class weight to balanced_subsample (removed from search space)
        pipeline.set_params(classifier__class_weight='balanced_subsample')
        
        # Perform hyperparameter tuning
        # Fit the search to find best parameters
        logger.info("Starting hyperparameter tuning with Bayesian optimization...")
        search.fit(self.X_train, self.y_train)
        
        # Get the best parameters and score
        best_params = search.best_params_
        best_score = search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best F1 score: {best_score:.4f}")
        
        # Update the pipeline with the best parameters
        pipeline = self._create_pipeline(use_smote)
        pipeline.set_params(**best_params)
        
        logger.info("\nTraining final model with best parameters...")
        
        # Perform cross-validation with best parameters
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        # Define metrics for cross-validation
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision'
        }
        
        # Get cross-validation scores
        cv_results = cross_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=N_JOBS,
            return_train_score=False,
            verbose=1
        )
        
        # Log cross-validation results
        logger.info("\nCross-validation results:")
        for metric in scoring.keys():
            scores = cv_results[f'test_{metric}']
            logger.info(f"{metric}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        # Fit the final model on the full training set with best parameters
        pipeline.fit(self.X_train, self.y_train)
        
        # Store the trained model and extract feature names
        self.model = pipeline
        self._extract_feature_names()
        
        # Get predicted probabilities for threshold optimization
        y_pred_proba = pipeline.predict_proba(self.X_train)[:, 1]
        
        # Optimize threshold
        self.optimal_threshold = self._optimize_threshold(self.y_train, y_pred_proba)
        logger.info(f"Optimal threshold: {self.optimal_threshold:.4f}")
        
        return cv_results
    
    def _optimize_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Find the optimal decision threshold that maximizes the F1 score.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for the positive class
            
        Returns:
            Optimal threshold value
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find the threshold that maximizes F1 score
        best_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
        
        # Store results for visualization
        self.threshold_optimization_results = {
            'thresholds': thresholds,
            'f1_scores': f1_scores[:-1] if len(f1_scores) > 1 else [0.5],  # Exclude last threshold as it's added by precision_recall_curve
            'precision': precision[:-1] if len(precision) > 1 else [0.5],
            'recall': recall[:-1] if len(recall) > 1 else [0.5]
        }
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores[:-1], 'b-', label='F1 Score')
        plt.plot(thresholds, precision[:-1], 'g-', label='Precision')
        plt.plot(thresholds, recall[:-1], 'r-', label='Recall')
        
        # Mark the optimal threshold
        plt.axvline(x=self.optimal_threshold, color='k', linestyle='--', 
                   label=f'Optimal Threshold ({self.optimal_threshold:.2f})')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Tuning for F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        output_file = self.output_dir / 'threshold_optimization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved threshold optimization plot to {output_file}")
        return self.optimal_threshold
        
    def _extract_feature_names(self) -> None:
        """Extract and store feature names after preprocessing."""
        try:
            # Get feature names after one-hot encoding
            ohe = self.model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            cat_features = ohe.get_feature_names_out(
                self.model.named_steps['preprocessor'].named_transformers_['cat'].feature_names_in_
            )
            
            # Get numerical features
            num_features = self.model.named_steps['preprocessor'].named_transformers_['num'].feature_names_in_
            
            # Combine and select top features
            all_features = list(num_features) + list(cat_features)
            selected_indices = self.model.named_steps['feature_selector'].get_support(indices=True)
            self.feature_names = [all_features[i] for i in selected_indices]
            
            logger.info(f"Selected {len(self.feature_names)} features")
            
        except Exception as e:
            logger.warning(f"Could not extract feature names: {str(e)}")
            self.feature_names = None
    
    def create_comprehensive_visualizations(self, y_pred_proba: np.ndarray, y_pred: np.ndarray) -> None:
        """Create comprehensive visualizations for model evaluation.
        
        Args:
            y_pred_proba: Predicted probabilities for the positive class
            y_pred: Predicted class labels
        """
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(exist_ok=True)
            
            # 1. ROC Curve
            logger.info("Generating ROC curve...")
            self._plot_roc_curve(y_pred_proba, 'roc_curve.png')
            
            # 2. Precision-Recall Curve with optimal threshold
            logger.info("Generating Precision-Recall curve...")
            self._plot_precision_recall_curve(
                y_pred_proba, 
                'precision_recall_curve.png',
                threshold=self.optimal_threshold if hasattr(self, 'optimal_threshold') else None
            )
            
            # 3. Confusion Matrix
            logger.info("Generating Confusion Matrix...")
            self._plot_confusion_matrix(y_pred, 'confusion_matrix.png')
            
            # 4. Feature Importance
            logger.info("Generating Feature Importance plot...")
            self._plot_feature_importance('feature_importance.png')
            
            # 5. SHAP Summary Plot (if available)
            try:
                logger.info("Generating SHAP summary plot...")
                self.explain_predictions(n_samples=100)
            except Exception as e:
                logger.warning(f"SHAP visualization failed: {str(e)}")
            
            # 6. Threshold Optimization Plot (if optimization was performed)
            if hasattr(self, 'threshold_optimization_results') and self.threshold_optimization_results is not None:
                logger.info("Generating Threshold Optimization plot...")
                try:
                    self._plot_threshold_optimization('threshold_optimization.png')
                except Exception as e:
                    logger.warning(f"Threshold optimization plot failed: {str(e)}")
            
            # 7. Calibration Plot
            logger.info("Generating Calibration plot...")
            try:
                self._plot_calibration_curve(y_pred_proba, 'calibration_curve.png')
            except Exception as e:
                logger.warning(f"Calibration plot failed: {str(e)}")
            
            # 8. Class Distribution
            logger.info("Generating Class Distribution plot...")
            try:
                self._plot_class_distribution('class_distribution.png')
            except Exception as e:
                logger.warning(f"Class distribution plot failed: {str(e)}")
            
            # 9. Correlation Heatmap
            logger.info("Generating Correlation Heatmap...")
            try:
                self._plot_correlation_heatmap('correlation_heatmap.png')
            except Exception as e:
                logger.warning(f"Correlation heatmap failed: {str(e)}")
            
            logger.info("All visualizations generated successfully!")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise
    
    def _plot_threshold_optimization(self, filename: str) -> None:
        """Plot threshold optimization results."""
        if not hasattr(self, 'threshold_optimization_results'):
            return
            
        plt.figure(figsize=(10, 6))
        results = self.threshold_optimization_results
        
        plt.plot(results['thresholds'], results['f1_scores'], label='F1 Score', color='blue')
        plt.axvline(x=self.optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal Threshold: {self.optimal_threshold:.2f}')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved threshold optimization plot to {output_path}")
    
    def _plot_calibration_curve(self, y_pred_proba: np.ndarray, filename: str) -> None:
        """Plot calibration curve for the model."""
        from sklearn.calibration import calibration_curve
        
        prob_true, prob_pred = calibration_curve(
            self.y_test, y_pred_proba, n_bins=10, strategy='quantile'
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.plot(prob_pred, prob_true, 's-', label='Model')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved calibration curve to {output_path}")
    
    def _plot_class_distribution(self, filename: str) -> None:
        """Plot class distribution in the dataset."""
        plt.figure(figsize=(10, 6))
        
        # Plot training and test set distribution
        train_counts = pd.Series(self.y_train).value_counts().sort_index()
        test_counts = pd.Series(self.y_test).value_counts().sort_index()
        
        x = np.arange(2)
        width = 0.35
        
        plt.bar(x - width/2, train_counts.values, width, label='Training Set')
        plt.bar(x + width/2, test_counts.values, width, label='Test Set')
        
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(x, ['Not At Risk', 'At Risk'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved class distribution plot to {output_path}")
    
    def _plot_correlation_heatmap(self, filename: str, top_n: int = 15) -> None:
        """Plot correlation heatmap of features."""
        if not hasattr(self, 'X_train') or not hasattr(self, 'feature_names'):
            return
            
        try:
            # Get feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                top_indices = np.argsort(importances)[-top_n:]
                top_features = [self.feature_names[i] for i in top_indices]
            else:
                top_features = self.feature_names[:min(top_n, len(self.feature_names))]
            
            # Get correlation matrix for top features
            X_df = pd.DataFrame(self.X_train, columns=self.feature_names)
            corr = X_df[top_features].corr()
            
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                       center=0, vmin=-1, vmax=1, cbar_kws={"shrink": .8})
            
            plt.title(f'Top {len(top_features)} Features Correlation Heatmap')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved correlation heatmap to {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not create correlation heatmap: {str(e)}")
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model on the test set with optional threshold optimization."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Get predicted probabilities
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            
            # Make predictions using optimized threshold if available
            if hasattr(self, 'optimal_threshold'):
                y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)
                logger.info(f"Using optimized threshold: {self.optimal_threshold:.4f}")
            else:
                y_pred = self.model.predict(self.X_test)
        else:
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = None
        
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
            
            # Generate comprehensive visualizations
            try:
                self.create_comprehensive_visualizations(y_pred_proba, y_pred)
            except Exception as e:
                logger.error(f"Error during visualization: {str(e)}")
        
        # Log metrics
        logger.info("\nTest set evaluation:")
        for metric, score in metrics.items():
            logger.info(f"{metric}: {score:.4f}")
        
        # Generate and save visualizations
        self._plot_confusion_matrix(y_pred, 'confusion_matrix.png')
        self._plot_roc_curve(y_pred_proba, 'roc_curve.png')
        self._plot_precision_recall_curve(y_pred_proba, 'precision_recall_curve.png')
        
        if self.feature_names is not None:
            self._plot_feature_importance('feature_importance.png')
        
        return metrics
    
    def _plot_confusion_matrix(self, y_pred: np.ndarray, filename: str) -> None:
        """Plot and save confusion matrix with enhanced visualization."""
        try:
            cm = confusion_matrix(self.y_test, y_pred)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
            
            plt.figure(figsize=(10, 8))
            
            # Create custom colormap
            cmap = sns.color_palette("Blues", as_cmap=True)
            
            # Plot confusion matrix
            ax = sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap=cmap, 
                           cbar=True, linewidths=0.5, linecolor='lightgray',
                           annot_kws={'size': 12, 'weight': 'bold'})
            
            # Add count annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j+0.5, i+0.3, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                           ha='center', va='center', color='black', fontsize=10)
            
            # Customize labels
            class_names = ['Not At Risk', 'At Risk']
            tick_marks = np.arange(len(class_names)) + 0.5
            
            plt.xticks(tick_marks, class_names, rotation=0, fontsize=11)
            plt.yticks(tick_marks, class_names, rotation=90, va='center', fontsize=11)
            
            plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
            plt.ylabel('True Label', fontsize=12, labelpad=10)
            plt.title('Confusion Matrix', pad=15, fontsize=14, fontweight='bold')
            
            # Add border
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_color('gray')
            
            output_path = self.output_dir / filename
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Saved confusion matrix to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {str(e)}")
            raise
    
    def _plot_roc_curve(self, y_pred_proba: np.ndarray, filename: str) -> None:
        """Plot and save ROC curve."""
        try:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='#1f77b4', lw=3, alpha=0.8, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.6)
            plt.fill_between(fpr, tpr, alpha=0.2, color='#1f77b4')
            
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.grid(True, alpha=0.3)
            
            plt.xlabel('False Positive Rate', fontsize=12, labelpad=10)
            plt.ylabel('True Positive Rate', fontsize=12, labelpad=10)
            plt.title('Receiver Operating Characteristic (ROC) Curve', pad=15, fontsize=14, fontweight='bold')
            
            # Add text annotation for AUC score
            plt.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
                    fontsize=12)
            
            plt.legend(loc="lower right", fontsize=10, frameon=True, fancybox=True, framealpha=0.9)
            
            output_path = self.output_dir / filename
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            logger.info(f"Saved ROC curve to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating ROC curve: {str(e)}")
            raise
    
    def _plot_precision_recall_curve(self, y_pred_proba: np.ndarray, filename: str, threshold: float = None) -> None:
        """Plot and save precision-recall curve with enhanced visualization and optional threshold marker."""
        try:
            precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
            avg_precision = average_precision_score(self.y_test, y_pred_proba)
            
            plt.figure(figsize=(10, 8))
            
            # Plot PR curve with fill
            plt.step(recall, precision, where='post', 
                    color='#2ca02c', linewidth=3, alpha=0.8,
                    label=f'PR Curve (AP = {avg_precision:.3f})')
            plt.fill_between(recall, precision, step='post', alpha=0.2, color='#2ca02c')
            
            # Mark the optimal threshold if provided
            if threshold is not None:
                # Find the closest threshold in the curve
                idx = np.argmin(np.abs(thresholds - threshold))
                if idx < len(recall) and idx < len(precision):  # Ensure indices are within bounds
                    plt.scatter(recall[idx], precision[idx], marker='o', 
                              color='red', s=150, zorder=10,
                              label=f'Optimal Threshold ({threshold:.2f})',
                              edgecolor='black', linewidth=1.5)
            
            # Plot no-skill line
            no_skill = len(self.y_test[self.y_test == 1]) / len(self.y_test)
            plt.axhline(y=no_skill, color='navy', linestyle='--', 
                       linewidth=2, alpha=0.7,
                       label=f'No Skill (AP = {no_skill:.3f})')
            
            # Customize axes and grid
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xticks(np.arange(0, 1.1, 0.1), fontsize=10)
            plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Add labels and title
            plt.xlabel('Recall (Sensitivity)', fontsize=12, labelpad=10)
            plt.ylabel('Precision (Positive Predictive Value)', fontsize=12, labelpad=10)
            plt.title('Precision-Recall Curve', pad=15, fontsize=14, fontweight='bold')
            
            # Add AP score annotation
            plt.text(0.02, 0.98, f'Average Precision = {avg_precision:.3f}',
                    ha='left', va='top', transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
                    fontsize=11)
            
            # Add legend
            plt.legend(loc='lower left', fontsize=10, frameon=True, 
                      fancybox=True, framealpha=0.9, bbox_to_anchor=(0, 0.02))
            
            # Save the figure
            output_path = self.output_dir / filename
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Saved precision-recall curve to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating precision-recall curve: {str(e)}")
            raise
    
    def _plot_feature_importance(self, filename: str, top_n: int = 15) -> None:
        """Plot and save feature importance with enhanced visualization."""
        try:
            if not hasattr(self.model, 'feature_importances_') or not hasattr(self, 'feature_names'):
                logger.warning("Feature importance not available for this model")
                return
                
            importances = self.model.feature_importances_
            
            # Handle different model types
            if hasattr(self.model, 'estimators_'):  # For ensemble models
                std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
            else:  # For non-ensemble models
                std = np.zeros_like(importances)
                
            indices = np.argsort(importances)[::-1]
            
            # Limit to top_n features
            top_n = min(top_n, len(importances))
            indices = indices[:top_n]
            
            # Create a horizontal bar plot
            plt.figure(figsize=(12, 0.5 * top_n + 2))
            
            # Get feature names, handling potential None or empty values
            feature_names = np.array(self.feature_names)[indices]
            
            # Create horizontal bars
            y_pos = np.arange(len(feature_names))
            bars = plt.barh(y_pos, importances[indices], xerr=std[indices],
                          align='center', color='#1f77b4', ecolor='#ff7f0e',
                          capsize=4, alpha=0.8)
            
            # Add value labels on the bars
            for i, (v, err) in enumerate(zip(importances[indices], std[indices])):
                plt.text(v + 0.01, i, f'{v:.3f} ± {err:.3f}', 
                        va='center', fontsize=10, color='black')
            
            # Customize the plot
            plt.yticks(y_pos, feature_names, fontsize=11)
            plt.xticks(fontsize=10)
            plt.xlabel('Feature Importance Score', fontsize=12, labelpad=10)
            plt.title(f'Top {top_n} Most Important Features', 
                     pad=15, fontsize=14, fontweight='bold')
            
            # Add grid lines
            plt.grid(True, axis='x', linestyle='--', alpha=0.3)
            
            # Invert y-axis for better visualization
            plt.gca().invert_yaxis()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Save feature importance to CSV
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances[indices],
                'std': std[indices]
            }).sort_values('importance', ascending=False)
            
            csv_path = self.output_dir / 'feature_importance.csv'
            importance_df.to_csv(csv_path, index=False)
            
            logger.info(f"Saved feature importance plot to {output_path}")
            logger.info(f"Saved feature importance data to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error generating feature importance plot: {str(e)}")
            raise
    
    def explain_predictions(self, n_samples: int = 100) -> None:
        """Generate SHAP values to explain model predictions."""
        try:
            # Sample data for SHAP (can be slow on large datasets)
            X_sample = self.X_test.sample(min(n_samples, len(self.X_test)), random_state=RANDOM_STATE)
            
            # Get the preprocessed data
            preprocessed_data = self.model.named_steps['preprocessor'].transform(X_sample)
            
            # Get the feature selector
            selected_features = self.model.named_steps['feature_selector'].get_support(indices=True)
            preprocessed_data = preprocessed_data[:, selected_features]
            
            # Get the model
            model = self.model.named_steps['classifier']
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(preprocessed_data)
            
            # Plot SHAP summary
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, preprocessed_data, 
                             feature_names=self.feature_names,
                             show=False, plot_size=(10, 8))
            plt.tight_layout()
            plt.savefig(self.output_dir / 'shap_summary.png')
            plt.close()
            
            logger.info("SHAP analysis completed")
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {str(e)}")
    
    def save_model(self, filename: str = 'dropout_model.pkl') -> None:
        """Save the trained model and related artifacts."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Save the model
        model_path = self.output_dir / filename
        joblib.dump(self.model, model_path)
        
        # Save feature names if available
        if self.feature_names is not None:
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.named_steps['classifier'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv(self.output_dir / 'feature_importance.csv', index=False)
        
        logger.info(f"Model and artifacts saved to {self.output_dir}")
    
    def predict_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Make a copy to avoid modifying the input
        data = data.copy()
        
        # Make predictions
        data['Predicted_Dropout'] = self.model.predict(data)
        data['Dropout_Probability'] = self.model.predict_proba(data)[:, 1]
        
        return data

def main():
    """Main function to run the dropout analysis pipeline."""
    try:
        # Initialize the predictor
        predictor = DropoutPredictor(
            data_path='students.csv',
            output_dir='output'
        )
        
        # 1. Load and preprocess data
        df = predictor.load_and_preprocess_data()
        
        # 2. Split data
        predictor.split_data(test_size=0.2)
        
        # 3. Train model with cross-validation
        cv_results = predictor.train_model(use_smote=True)
        
        # 4. Evaluate on test set
        test_metrics = predictor.evaluate_model()
        
        # 5. Explain predictions with SHAP (optional, can be slow)
        predictor.explain_predictions(n_samples=100)
        
        # 6. Save the model and artifacts
        predictor.save_model()
        
        # 7. Generate predictions for all data (without target)
        predictions = predictor.predict_new_data(df.drop('dropout', axis=1, errors='ignore'))
        predictions.to_csv(predictor.output_dir / 'all_predictions.csv', index=False)
        
        logger.info("\nProcess completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
