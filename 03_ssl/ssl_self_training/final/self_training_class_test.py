import sys
sys.path.append('../../') 
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, log_loss, top_k_accuracy_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from joblib import dump
from sklearn.utils import resample
from ssl_data_transformers_robust import PreprocessingPipeline
import json
from scipy.stats import randint, uniform
from plotting_utils import save_figure
import os
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert NumPy scalars to Python scalars
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(value) for value in obj]
    else:
        return obj

class SelfTrainingPipelineFinal:
    def __init__(self, threshold=0.75):
        # Set random seed
        self.random_seed = 42
        # Set threshold for pseudo-labeling
        self.threshold = threshold
        # Define classifiers
        self.classifiers = {
            'RandomForest': RandomForestClassifier(random_state=self.random_seed, class_weight='balanced',
                                                   n_jobs=2),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softprob',
                                         random_state=self.random_seed, class_weight='balanced'),
            'LightGBM': lgb.LGBMClassifier(random_state=self.random_seed, importance_type='gain', 
                                           verbose='-1', objective='multiclass', class_weight='balanced', num_class=5,
                                           metric='multi_logloss', n_jobs=2),
            # 'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(class_weight='balanced'), random_state=self.random_seed, n_jobs=4)
        }

        # Define hyperparams grid
        self.optimal_hyperparameters = {
            'RandomForest': {
                 'n_estimators': randint(10, 1000),  # Number of trees in the forest
                 'max_depth': randint(8, 100),
                'min_samples_leaf': randint(2, 20),
                'min_samples_split': randint(2, 20),
                'bootstrap': [True, False],
                'max_features': ['sqrt', 'log2', None],
                'criterion': ['gini', 'entropy']
            },
            'XGBoost': {
                'n_estimators': randint(50, 500),  # Number of trees in the ensemble
                'max_depth': randint(3, 12),  # Complexity of the trees
                'min_child_weight': randint(1, 10),  # Regularization to prevent overfitting
                'learning_rate': uniform(0.05, 0.25),  # Rate of learning of the model
                'gamma': uniform(0, 5),  # Regularization parameter (min_split_loss)
                'colsample_bytree': uniform(0.5, 0.5),  # Fraction of features to use for a tree
                'subsample': uniform(0.6, 0.4)  # Fraction of instances to use for a tree
            },
            'LightGBM': {
                'max_depth': randint(3, 12),  # Control the maximum depth of trees
                'num_leaves': randint(8, 60),  # Number of leaves in full tree
                'min_data_in_leaf': randint(100, 1000),  # Minimum number of data in one leaf
                'feature_fraction': uniform(0.6, 0.4),  # Fraction of features to be used in each iteration
                'bagging_fraction': uniform(0.6, 0.4),  # Fraction of data to be used for each iteration
                'learning_rate': uniform(0.01, 0.2)  # Speed of learning
            },
            # 'Bagging': {
            #     'n_estimators': randint(10, 100),  # Number of base estimators in the ensemble
            #     'max_samples': uniform(0.5, 0.5),  # Fraction of samples to draw from X to train each base estimator
            #     'max_features': uniform(0.5, 0.5), # Fraction of features to draw from X to train each base estimator
            #     'bootstrap': [True, False],
            #     'bootstrap_features': [True, False]
            # }
        }

        # Initialize other variables
        self.le = LabelEncoder()
        self.results = {
            'cv_metrics': [],
            'training': [],
            'test': [],
            'pseudo_labels_info': [],
            'labeled_samples_info': [],
            'training_time': []
        }
        self.best_hyperparameters = {}

    def set_data(self, df, label_column, categorical_columns, numerical_columns, text_column=None, include_text=True):
        self.data = df
        self.label_column = label_column
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_column = text_column
        self.include_text = include_text

    def _prepare_data(self, labeled_percentage):
        # Split data into labeled and unlabeled subsets
        labeled_data = self.data[self.data[self.label_column] != '-1']
        unlabeled_data = self.data[self.data[self.label_column] == '-1']

        # Split labeled data into training and test sets
        X_labeled = labeled_data.drop(columns=self.label_column)
        y_labeled = labeled_data[self.label_column] 
        X_train, X_test, y_train, y_test = train_test_split(
            X_labeled, y_labeled, test_size=0.2, stratify=y_labeled, random_state=self.random_seed
        )

        # Reduce the size of the labeled training data if needed
        if labeled_percentage < 1.0:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=labeled_percentage, stratify=y_train, random_state=self.random_seed
            )

        # Apply preprocessing pipeline
        self.preprocessing_pipeline = PreprocessingPipeline(
            self.categorical_columns, self.numerical_columns, self.text_column if self.include_text else None
        )
        X_train = self.preprocessing_pipeline.fit_transform(X_train)
        X_test = self.preprocessing_pipeline.transform(X_test)
        X_unlabeled = self.preprocessing_pipeline.transform(unlabeled_data.drop(columns=self.label_column))
        self.X_unlabeled = X_unlabeled

        # Encode labels
        self.le.fit(y_train)
        y_train_encoded = self.le.transform(y_train)
        y_test_encoded = self.le.transform(y_test)

        # Apply SMOTE only to the labeled data
        X_train_labeled = X_train[y_train_encoded != '-1']
        y_train_labeled = y_train_encoded[y_train_encoded != '-1']
        smote = SMOTE(random_state=self.random_seed)
        X_train_labeled, y_train_labeled = smote.fit_resample(X_train_labeled, y_train_labeled)
        # Store the number of labeled samples
        self.num_labeled_samples = len(y_train_labeled) 

        logger.info(f"Data shape after SMOTE: {X_train_labeled.shape} with labels shape: {y_train_labeled.shape}")

        # Combine labeled and unlabeled data
        X_train_final = np.vstack((X_train_labeled, X_unlabeled))
        y_train_final = np.concatenate((y_train_labeled, [-1] * len(X_unlabeled)))

        logger.info(f"Total training data shape after combining labeled and unlabeled data: {X_train_final.shape}")

        return X_train_final, y_train_final, X_test, y_test_encoded
        
    def _evaluate_on_bootstrap_samples(self, model, X_original_test, y_original_test, n_bootstrap_samples):
        """Evaluate the model on multiple bootstrap samples."""
        bootstrap_accuracies = []

        # Generate multiple bootstrap test sets of the same size as the original test set
        for _ in range(n_bootstrap_samples):
            X_test, y_test = resample(X_original_test, y_original_test, replace=True, n_samples=len(X_original_test))
            y_test_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)
            bootstrap_accuracies.append(test_acc)

        # Calculate mean and standard deviation of accuracies
        mean_acc = np.mean(bootstrap_accuracies)
        std_acc = np.std(bootstrap_accuracies)

        return mean_acc, std_acc
    
    def _train_and_evaluate(self, X_train, y_train, X_test, y_test, model_name, labeled_percentage):

        #feature_importances = {}

        logger.info(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

        start_time = time.time()

        try:
            # Initialize the classifier with optimal hyperparameters
            logger.info(f"Initializing {model_name} classifier.")
            classifier = self.classifiers[model_name]

            # Set up cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
            random_search = RandomizedSearchCV(
                estimator=classifier, 
                param_distributions=self.optimal_hyperparameters[model_name], 
                n_iter=20,  # Number of parameter settings sampled
                cv=skf, 
                verbose=1, 
                random_state=self.random_seed,
                n_jobs=2  # Use half of the cores
        )
            
            # Only the labeled data for hyperparameter tuning
            is_labeled = y_train != -1
            X_train_labeled = X_train[is_labeled]
            y_train_labeled = y_train[is_labeled]

            # Best hyperparameterss
            search_params = random_search.fit(X_train_labeled, y_train_labeled)
            best_params = search_params.best_params_
            cv_results = search_params.cv_results_
            best_index = search_params.best_index_
            best_mean_score = cv_results['mean_test_score'][best_index]
            best_std_score = cv_results['std_test_score'][best_index]
            best_estimator = search_params.best_estimator_

            # Create a DataFrame from the cross-validation results
            cv_results_df = pd.DataFrame(search_params.cv_results_)
            csv_filename = f'cv_results_{model_name}_{labeled_percentage}.csv'
            csv_file_path = os.path.join('cv_results', csv_filename) 
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            cv_results_df.to_csv(csv_file_path, index=False)
            logger.info(f"Cross-validation results saved to {csv_filename}")

            # Store best parameters in the dictionary
            self.best_hyperparameters[model_name] = best_params
            logger.info(f"Best parameters for {model_name}: {best_params}")

            # Initialize the classifier with the best hyperparameters
            best_classifier = classifier.set_params(**best_params)

            # Apply self-training
            logger.info(f"Starting self-training for {model_name}.")
            self_training_model = SelfTrainingClassifier(
                best_classifier, criterion='threshold', threshold=self.threshold, verbose=True)

            self_training_model.fit(X_train, y_train)

            # Evaluate on training set post self-training
            y_train_pred = self_training_model.predict(X_train_labeled)
            train_acc = accuracy_score(y_train_labeled, y_train_pred)
            train_recall = recall_score(y_train_labeled, y_train_pred, average='weighted')
            train_precision = precision_score(y_train_labeled, y_train_pred, average='weighted')
            train_f1 = f1_score(y_train_labeled, y_train_pred, average='weighted')
            train_balanced_acc = balanced_accuracy_score(y_train_labeled, y_train_pred)
            train_mcc = matthews_corrcoef(y_train_labeled, y_train_pred)
            y_train_proba = self_training_model.predict_proba(X_train_labeled)
            train_top_k_acc = top_k_accuracy_score(y_train_labeled, y_train_proba, k=3)
            train_error = 1 - train_acc  # Training error
            train_log_loss = log_loss(y_train_labeled, y_train_proba)
            train_kappa_score = cohen_kappa_score(y_train_labeled, y_train_pred)
            train_conf_matrix = confusion_matrix(y_train_labeled, y_train_pred).tolist()
            train_evaluation_report = classification_report(y_train_labeled, y_train_pred, output_dict=True, zero_division=0)

            # Evaluate on test set
            logger.info(f"Evaluating {model_name} on the test set.")
            y_test_pred = self_training_model.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred, average='weighted')
            test_precision = precision_score(y_test, y_test_pred, average='weighted')
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
            test_mcc = matthews_corrcoef(y_test, y_test_pred)
            y_test_proba = self_training_model.predict_proba(X_test)
            test_top_k_acc = top_k_accuracy_score(y_test, y_test_proba, k=3)
            test_error = 1 - test_acc  
            test_log_loss = log_loss(y_test, y_test_proba)  
            test_kappa_score= cohen_kappa_score(y_test, y_test_pred)
            test_conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()
            test_evaluation_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

            end_time = time.time()
            training_time = end_time - start_time
            logger.info(f"{model_name} training completed in {training_time:.2f} seconds.")
            
            # Count pseudo-labels
            pseudo_labels = self_training_model.transduction_[-len(self.X_unlabeled):]
            pseudo_label_count = np.sum(pseudo_labels != -1)

            # Logging the count and distribution of pseudo-labels
            pseudo_label_distribution = np.bincount(pseudo_labels[pseudo_labels != -1])
            logger.info(f"Pseudo-labels generated: {pseudo_label_count}")
            logger.info(f"Pseudo-label distribution: {pseudo_label_distribution}")

            # This step depends on your PreprocessingPipeline implementation
            feature_names = self.preprocessing_pipeline.get_feature_names()

            # Check and store feature importances with names
            if hasattr(best_estimator, 'feature_importances_'):
                importances = best_estimator.feature_importances_
                feature_importance_dict = {name: importance for name, importance in zip(feature_names, importances)}
            else:
                feature_importance_dict = None

            # Save training and test results
            # Add CV metrics to the results
            # Append feature importances with names to the results
            self.results['feature_importances'].setdefault(model_name, []).append({
                'labeled_percentage': labeled_percentage,
                'importances': feature_importance_dict
            })
            self.results['cv_metrics'].append({
                'model': model_name,
                'labeled_percentage': labeled_percentage,
                'best_params': best_params,
                'best_mean_cv_score': best_mean_score,
                'best_std_cv_score': best_std_score
            })
            self.results['training'].append({
                'model': model_name,
                'labeled_percentage': labeled_percentage,
                'train_accuracy': train_acc,
                'train_recall': train_recall,
                'train_precision': train_precision,
                'train_f1': train_f1,
                'train_balanced_accuracy': train_balanced_acc,
                'train_top_k_accuracy': train_top_k_acc,
                'train_mcc': train_mcc,
                'train_error': train_error,
                'train_log_loss': train_log_loss,
                'kappa_score_train': train_kappa_score,
                'confusion_matrix': train_conf_matrix,
                'train_report': train_evaluation_report
            })
            self.results['test'].append({
                'model': model_name,
                'labeled_percentage': labeled_percentage,
                'test_accuracy': test_acc,
                'test_recall': test_recall,
                'test_precision': test_precision,
                'test_f1': test_f1,
                'test_balanced_accuracy': test_balanced_acc,
                'test_top_k_accuracy': test_top_k_acc,
                'test_mcc': test_mcc,
                'test_error': test_error,
                'test_log_loss': test_log_loss,
                'kappa_score_test': test_kappa_score,
                'test_confusion_matrix': test_conf_matrix,
                'test_report': test_evaluation_report
            })
            self.results['pseudo_labels_info'].append({
                'model': model_name,
                'labeled_percentage': labeled_percentage,
                'pseudo_label_count': pseudo_label_count,
                'pseudo_label_distribution': pseudo_label_distribution.tolist()  # Convert to list for JSON serialization
            }),
            self.results['labeled_samples_info'].append({
                'model': model_name,
                'labeled_percentage': labeled_percentage,
                'labeled_samples_count': self.num_labeled_samples
            })
            self.results['training_time'].append({
                'model': model_name,
                'labeled_percentage': labeled_percentage,
                'training_time': training_time
        })
                        
            logger.info(f"{model_name}: Training and evaluation completed with {labeled_percentage * 100}% labeled data.")

        except Exception as e:
            logger.error(f"Error in training and evaluating {model_name}: {str(e)}")

        self.classifiers[model_name] = self_training_model

        # New method to get best hyperparameters
    def get_best_hyperparameters(self):
        return self.best_hyperparameters
    
    def run(self, labeled_percentages, n_bootstrap_samples=500):
        # Initialize dictionaries to store results
        self.results['training'] = []
        self.results['test'] = []
        self.results['bootstrap_test'] = []  # Added for bootstrap results
        self.results['pseudo_labels_info'] = []
        self.results['labeled_samples_info'] = []
        self.results['feature_importances'] = {}

        for labeled_percentage in labeled_percentages:
            logger.info(f"Running for {labeled_percentage * 100}% labeled data...")

            try:
                # Prepare data
                X_train, y_train, X_original_test, y_original_test = self._prepare_data(labeled_percentage)

                # Train and evaluate each model
                for model_name in self.classifiers:
                    logger.info(f"Training {model_name}...")
                    self._train_and_evaluate(X_train, y_train, X_original_test, y_original_test, model_name, labeled_percentage)

                    # Evaluate on bootstrap samples
                    model = self.classifiers[model_name]
                    mean_acc, std_acc = self._evaluate_on_bootstrap_samples(model, X_original_test, y_original_test, n_bootstrap_samples)
                    logger.info(f"{model_name}: Mean Accuracy on Bootstrap Samples: {mean_acc}, Standard Deviation: {std_acc}")

                    # Store bootstrap results
                    self.results['bootstrap_test'].append({
                        'model': model_name,
                        'labeled_percentage': labeled_percentage,
                        'mean_accuracy': mean_acc,
                        'std_accuracy': std_acc
                    })

            except Exception as e:
                logger.error(f"Error during pipeline run: {e}")

        # Convert results to JSON serializable format
        converted_results = convert_numpy(self.results)

        # Save or process the results as needed
        with open('self_training_results_test.json', 'w') as file:
            json.dump(converted_results, file, indent=4)

        return self.results
    
    


# # Example usage
# pipeline = SelfTrainingPipeline()
# pipeline.set_data(df, label_column='category_bertopic', include_text=True)

# # Define the percentages of labeled data to be used in the loop
# labeled_data_percentages = [1.0, 0.8, 0.6, 0.4, 0.2, 0]
# results = pipeline.run(labeled_data_percentages)
