import sys
sys.path.append('../../') 
import os
import time
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, log_loss, top_k_accuracy_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.utils import resample
from ssl_data_transformers_robust import PreprocessingPipeline
from scipy.stats import randint, uniform
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import label_binarize


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
    

# Define the class for the self-training pipeline
class SelfTrainingPipelineFinal:
    def __init__(self, initial_threshold=0.8, min_threshold=0.5, threshold_decrement=0.05, eval_metric='f1_score'):
        # Set initial threshold and other threshold parameters
        self.initial_threshold = initial_threshold
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.threshold_decrement = threshold_decrement
        self.eval_metric = eval_metric
        # Add a class attribute to store the best F1 score
        self.best_f1_score = 0
        self.improvement_threshold = 0.01 

        # Set random seed
        self.random_seed = 42

        # Define classifiers
        self.classifiers = {
            # class_weight = 'balanced_subsample' for Random Forest
            'RandomForest': RandomForestClassifier(random_state=self.random_seed, 
                                                   class_weight='balanced', n_jobs=2),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
                                            objective='multi:softprob',
                                            random_state=self.random_seed),
            'LightGBM': lgb.LGBMClassifier(random_state=self.random_seed, importance_type='gain', 
                                           verbose='-1', objective='multiclass', 
                                           class_weight='balanced', num_class=5,
                                           metric='multi_logloss'),
            # 'Bagging': BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight='balanced'), 
            #                                        sampling_strategy='auto', replacement=False,
            #                                        random_state=self.random_seed, warm_start=True, n_jobs=2)
        }

        # Define hyperparams grid
        self.optimal_hyperparameters = {
            'RandomForest': {
                'n_estimators': randint(10, 1000),  # Number of trees in the forest
                'max_depth': randint(20, 80),
                'min_samples_leaf': randint(5, 15),
                'min_samples_split': randint(5, 15),
                'bootstrap': [True, False],
                'max_features': ['sqrt', 'log2', None],
                'criterion': ['gini', 'entropy']            
            },
            'XGBoost': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(4, 10),
                'min_child_weight': randint(1, 10),
                'learning_rate': uniform(0.05, 0.25),
                'gamma': uniform(0, 5),
                'colsample_bytree': uniform(0.5, 0.5),
                'subsample': uniform(0.6, 0.4),
                'alpha': [0.001, 0.01, 0.1, 1],  # L1 regularization
                'lambda': [0.1, 1, 5, 10, 20]  # L2 regularization
            },
            'LightGBM': {
                'max_depth': randint(6, 10),
                'num_leaves': randint(20, 40),
                'min_data_in_leaf': randint(300, 700),
                'feature_fraction': uniform(0.8, 0.2),
                'bagging_fraction': uniform(0.5, 0.2),
                'learning_rate': uniform(0.1, 0.15),
                'lambda_l1': [0.01, 0.1, 1, 10, 100],  # L1 regularization
                'lambda_l2': [0.01, 0.1, 1, 10, 100]  # L2 regularization
            },
        #    'Bagging': {
        #         'n_estimators': randint(30, 85),  # Adjusted to range between the CV results
        #         'max_samples': uniform(0.75, 0.25),  # Shifted towards higher values found in CV
        #         'max_features': uniform(0.6, 0.4),  # Adjusted to encompass the range of CV results
        #         'bootstrap': [True, False],  # Keeping as is, since both values are used
        #         'bootstrap_features': [True, False]  # Keeping as is, since both values are used
        #     }
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

    # Function to calculate class weights
    def calculate_class_weights(self, y):
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        return {i: weight for i, weight in enumerate(class_weights)}
    
    # Function to set data and features
    def set_data(self, df, label_column, categorical_columns, numerical_columns, text_column=None, include_text=True):
        self.data = df
        self.label_column = label_column
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_column = text_column
        self.include_text = include_text

    # Function to prepare data for modeling
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

        logger.info(f"Shapes after initial train-test split: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")


        # Further split the labeled data into training and validation sets
        X_train_labeled, X_val_labeled, y_train_labeled, y_val_labeled = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=self.random_seed
        )

        logger.info(f"Shapes after train-validation split: X_train_labeled: {X_train_labeled.shape}, y_train_labeled: {y_train_labeled.shape}, X_val_labeled: {X_val_labeled.shape}, y_val_labeled: {y_val_labeled.shape}")

        # Reduce the size of the labeled training data if needed
        if labeled_percentage < 1.0:
            X_train_labeled, _, y_train_labeled, _ = train_test_split(
                X_train_labeled, y_train_labeled, train_size=labeled_percentage, stratify=y_train_labeled, random_state=self.random_seed
            )

        logger.info(f"Shapes after reducing labeled data: X_train_labeled: {X_train_labeled.shape}, y_train_labeled: {y_train_labeled.shape}")

        # Apply preprocessing pipeline
        self.preprocessing_pipeline = PreprocessingPipeline(
            self.categorical_columns, self.numerical_columns, self.text_column if self.include_text else None
            )
        logger.info(f"Feature names before preprocessing: {X_train_labeled.columns.tolist()}")
        X_train_labeled = self.preprocessing_pipeline.fit_transform(X_train_labeled)
        logger.info(f"Feature names after preprocessing: {self.preprocessing_pipeline.get_feature_names()}")        
        X_val_labeled = self.preprocessing_pipeline.transform(X_val_labeled)  # Preprocess the validation set
        X_test = self.preprocessing_pipeline.transform(X_test)
        X_unlabeled = self.preprocessing_pipeline.transform(unlabeled_data.drop(columns=self.label_column))
        self.X_unlabeled = X_unlabeled

        # Encode labels
        self.le.fit(y_train)
        y_train_encoded = self.le.transform(y_train_labeled)
        y_test_encoded = self.le.transform(y_test)
        y_val_labeled_encoded = self.le.transform(y_val_labeled)  # Encode y_val_labeled

        # Apply SMOTE only to the labeled data
        y_train_labeled = y_train_encoded[y_train_encoded != '-1']
        smote = SMOTE(random_state=self.random_seed)
        X_train_labeled, y_train_labeled = smote.fit_resample(X_train_labeled, y_train_labeled)
        # Store the number of labeled samples
        self.num_labeled_samples = len(y_train_labeled) 
 
        logger.info(f"Data shape after SMOTE: {X_train_labeled.shape} with labels shape: {y_train_labeled.shape}")

        # Combine labeled and unlabeled data
        X_train_final = np.vstack((X_train_labeled, X_unlabeled))
        y_train_final = np.concatenate((y_train_labeled, [-1] * len(X_unlabeled)))

        logger.info(f"Shapes after combining labeled and unlabeled data: X_train_final: {X_train_final.shape}, y_train_final: {y_train_final.shape}, X_unlabeled: {X_unlabeled.shape}")

        return X_train_final, y_train_final, X_val_labeled, y_val_labeled_encoded, X_test, y_test_encoded
        
    # Function to evaluate the model on multiple bootstrap samples   
    def _evaluate_on_bootstrap_samples(self, model, X_original_test, y_original_test, n_bootstrap_samples):
        """Evaluate the model on multiple bootstrap samples."""
        
        # Dictionaries to store metrics
        bootstrap_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
        }

        # Generate multiple bootstrap test sets and evaluate the model
        for _ in range(n_bootstrap_samples):
            X_test, y_test = resample(X_original_test, y_original_test, replace=True, n_samples=len(X_original_test))
            X_test_selected = self.feature_selector.transform(X_test)  # Apply feature selection
            y_test_pred = model.predict(X_test_selected)
            
            # Calculate and store each metric
            bootstrap_metrics['accuracy'].append(accuracy_score(y_test, y_test_pred))
            bootstrap_metrics['precision'].append(precision_score(y_test, y_test_pred, average='weighted', zero_division=0))
            bootstrap_metrics['recall'].append(recall_score(y_test, y_test_pred, average='weighted', zero_division=0))
            bootstrap_metrics['f1_score'].append(f1_score(y_test, y_test_pred, average='weighted', zero_division=0))

        # Calculate mean and standard deviation for each metric
        mean_std_metrics = {metric: (np.mean(scores), np.std(scores)) for metric, scores in bootstrap_metrics.items()}

        return mean_std_metrics
    
    # Function to calculate ROC AUC
    def calculate_roc_auc(self, y_true, y_scores, n_classes):
        y_true_binarized = label_binarize(y_true, classes=range(n_classes))
        fpr, tpr, roc_auc = dict(), dict(), dict()

        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return fpr, tpr, roc_auc

    # Function to save ROC data for plotting
    def save_roc_data(self, model_name, labeled_percentage, fpr, tpr, roc_auc):
        save_dir = 'roc_auc_models'
        os.makedirs(save_dir, exist_ok=True)

        # Recursive function to convert NumPy arrays in nested structures to lists
        def convert_to_list(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, dict):
                return {k: convert_to_list(v) for k, v in item.items()}
            else:
                return item

        # Apply conversion to fpr, tpr, and roc_auc
        fpr_json = convert_to_list(fpr)
        tpr_json = convert_to_list(tpr)
        roc_auc_json = convert_to_list(roc_auc)

        filename = os.path.join(save_dir, f'roc_data_{model_name}_{labeled_percentage}.json')
        roc_data = {'fpr': fpr_json, 'tpr': tpr_json, 'roc_auc': roc_auc_json}

        with open(filename, 'w') as file:
            json.dump(roc_data, file, indent=4)
        print(f'Saved ROC data for {model_name} with {labeled_percentage}% labeled data to {filename}')
    
    # Function to train and evaluate the model
    def _train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, model_name, labeled_percentage):

        #feature_importances = {}

        logger.info(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

        start_time = time.time()

        try:
            # Initialize the classifier with optimal hyperparameters
            logger.info(f"Initializing {model_name} classifier.")
            classifier = self.classifiers[model_name]

            # Only the labeled data for hyperparameter tuning and training
            is_labeled = y_train != -1
            X_train_labeled = X_train[is_labeled]
            y_train_labeled = y_train[is_labeled]

            # Set up cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
            random_search = RandomizedSearchCV(
                    estimator=classifier, 
                    param_distributions=self.optimal_hyperparameters[model_name], 
                    n_iter=20,  # Number of parameter settings sampled
                    cv=skf, 
                    verbose=1, 
                    random_state=self.random_seed,
                    n_jobs=4  # Use half of the cores
            )

            # Handle class imbalance for XGBoost
            if model_name == 'XGBoost':
                class_weights = self.calculate_class_weights(y_train_labeled)
                sample_weights = np.array([class_weights[i] for i in y_train_labeled])
                logger.info(f"Class weights for XGBoost: {class_weights}")

                # Best hyperparameterss
                logger.info("Starting hyperparameter tuning for XGBoost...")
                search_params = random_search.fit(X_train_labeled, y_train_labeled, sample_weight=
                                                  sample_weights)
            else:
                # Perform hyperparameter tuning without class weights
                logger.info(f"Starting hyperparameter tuning for {model_name}...")
                search_params = random_search.fit(X_train_labeled, y_train_labeled) 

            
            best_params = search_params.best_params_
            cv_results = search_params.cv_results_
            best_index = search_params.best_index_
            best_mean_score = cv_results['mean_test_score'][best_index]
            best_std_score = cv_results['std_test_score'][best_index]
            best_estimator = search_params.best_estimator_

            # Create a DataFrame from the cross-validation results
            cv_results_df = pd.DataFrame(search_params.cv_results_)
            csv_filename = f'cv_results_{model_name}_{labeled_percentage}_SMOTE.csv'
            csv_file_path = os.path.join('cv_results', csv_filename) 
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            cv_results_df.to_csv(csv_file_path, index=False)
            logger.info(f"Cross-validation results saved to {csv_filename}")

            # Store best parameters in the dictionary
            self.best_hyperparameters[model_name] = best_params
            logger.info(f"Best parameters for {model_name}: {best_params}")

            # Initialize the classifier with the best hyperparameters
            best_classifier = classifier.set_params(**best_params)

            # Select features based on feature importances
            # Feature Selection
            logger.info(f"Applying feature selection for {model_name}.")

            self.feature_selector = SelectFromModel(best_classifier, threshold='mean')
            X_train_labeled_selected = self.feature_selector.fit_transform(X_train_labeled, y_train_labeled)
            X_val_selected = self.feature_selector.transform(X_val)
            X_test_selected = self.feature_selector.transform(X_test)
            X_train_selected = self.feature_selector.transform(X_train)

            logger.info(f"Number of features after selection in X_train_labeled: {X_train_labeled_selected.shape}")
            logger.info(f"Number of features after selection in X_train_final: {X_train_selected.shape}")

            selected_features_indices = self.feature_selector.get_support(indices=True)
            feature_names = self.preprocessing_pipeline.get_feature_names()
            selected_feature_names = [feature_names[i] for i in selected_features_indices]    

            # Apply self-training with selected features
            logger.info(f"Starting self-training for {model_name} with selected features.")
            self_training_model = SelfTrainingClassifier(
                best_classifier, criterion='threshold', threshold=self.threshold, verbose=True
            )
            self_training_model.fit(X_train_selected, y_train)

            # Evaluate on the validation set
            y_val_pred = self_training_model.predict(X_val_selected)
            val_performance = f1_score(y_val, y_val_pred, average='weighted')

            # Adjust the threshold based on F1 score change
            if val_performance > self.best_f1_score * (1 + self.improvement_threshold):
                self.best_f1_score = val_performance  # Update the best F1 score
                self.threshold = max(self.threshold - self.threshold_decrement, self.min_threshold)
                logger.info(f"Threshold decreased to {self.threshold} based on improved F1 score.")
            elif val_performance < self.best_f1_score:
                self.threshold = min(self.threshold + self.threshold_decrement, self.initial_threshold)
                logger.info(f"Threshold increased to {self.threshold} due to decreased F1 score.")

            # Update the previous performance for next iteration comparison
            self.previous_val_performance = val_performance

            n_classes = np.unique(y_train_labeled)
            y_train_binarized = label_binarize(y_train_labeled, classes=n_classes)

            logger.info(f"Finished self-training for {model_name} with selected features.")

            # Evaluate on training set post self-training
            y_train_pred = self_training_model.predict(X_train_labeled_selected)
            logger.info(f"Shape of X_train_labeled_selected: {X_train_labeled_selected.shape}")
            logger.info(f"Shape of y_train_pred: {y_train_pred.shape}")

            train_acc = accuracy_score(y_train_labeled, y_train_pred)
            train_recall = recall_score(y_train_labeled, y_train_pred, average='weighted', zero_division=0)
            train_precision = precision_score(y_train_labeled, y_train_pred, average='weighted', zero_division=0)
            train_f1 = f1_score(y_train_labeled, y_train_pred, average='weighted', zero_division=0)
            train_balanced_acc = balanced_accuracy_score(y_train_labeled, y_train_pred)
            train_mcc = matthews_corrcoef(y_train_labeled, y_train_pred)
            y_train_proba = self_training_model.predict_proba(X_train_labeled_selected)
            train_roc_auc = roc_auc_score(y_train_binarized, y_train_proba, multi_class="ovr")
            train_top_k_acc = top_k_accuracy_score(y_train_labeled, y_train_proba, k=3)
            train_error = 1 - train_acc  # Training error
            train_log_loss = log_loss(y_train_labeled, y_train_proba)
            train_kappa_score = cohen_kappa_score(y_train_labeled, y_train_pred)
            train_conf_matrix = confusion_matrix(y_train_labeled, y_train_pred).tolist()
            train_evaluation_report = classification_report(y_train_labeled, y_train_pred, output_dict=True, zero_division=0)

            y_test_binarized = label_binarize(y_test, classes=n_classes)

            # Evaluate on test set
            logger.info(f"Evaluating {model_name} on the test set.")
            y_test_pred = self_training_model.predict(X_test_selected)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
            test_mcc = matthews_corrcoef(y_test, y_test_pred)
            y_test_proba = self_training_model.predict_proba(X_test_selected)
            test_roc_auc = roc_auc_score(y_test_binarized, y_test_proba, multi_class="ovr")
            test_top_k_acc = top_k_accuracy_score(y_test, y_test_proba, k=3)
            test_error = 1 - test_acc  
            test_log_loss = log_loss(y_test, y_test_proba)  
            test_kappa_score= cohen_kappa_score(y_test, y_test_pred)
            test_conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()
            test_evaluation_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

            end_time = time.time()
            training_time = end_time - start_time
            logger.info(f"{model_name} training completed in {training_time:.2f} seconds.")

            # Save roc auc data
            n_classes = len(np.unique(y_train_labeled))
            fpr_train, tpr_train, roc_auc_train = self.calculate_roc_auc(y_train_labeled, y_train_proba, n_classes)
            fpr_test, tpr_test, roc_auc_test = self.calculate_roc_auc(y_test, y_test_proba, n_classes)
            # Save the class-wise, micro, and macro ROC AUC
            self.save_roc_data(model_name, labeled_percentage, {'train': fpr_train, 'test': fpr_test}, 
                            {'train': tpr_train, 'test': tpr_test}, {'train': roc_auc_train, 'test': roc_auc_test})            
            
            # Count pseudo-labels
            pseudo_labels = self_training_model.transduction_[-len(self.X_unlabeled):]
            pseudo_label_count = np.sum(pseudo_labels != -1)
            not_pseudo_label_count = np.sum(pseudo_labels == -1) 

            # Logging the count and distribution of pseudo-labels
            pseudo_label_distribution = np.bincount(pseudo_labels[pseudo_labels != -1])
            logger.info(f"Pseudo-labels generated: {pseudo_label_count}")
            logger.info(f"Pseudo-label distribution: {pseudo_label_distribution}")

            # This step depends on your PreprocessingPipeline implementation
            feature_names = self.preprocessing_pipeline.get_feature_names()

            # Check and store feature importances with names for selected features
            if hasattr(best_estimator, 'feature_importances_'):
                importances = best_estimator.feature_importances_
                selected_feature_importance_dict = {name: importances[i] for i, name in zip(selected_features_indices, selected_feature_names)}
            else:
                selected_feature_importance_dict = None

            # Save training and test results
            self.results['feature_importances'].setdefault(model_name, []).append({
                'labeled_percentage': labeled_percentage,
                'importances': selected_feature_importance_dict
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
                'train_roc_auc': train_roc_auc,
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
                'test_roc_auc': test_roc_auc,
                'test_top_k_accuracy': test_top_k_acc,
                'test_mcc': test_mcc,
                'test_error': test_error,
                'test_log_loss': test_log_loss,
                'kappa_score_test': test_kappa_score,
                'confusion_matrix': test_conf_matrix,
                'test_report': test_evaluation_report
            })
            self.results['pseudo_labels_info'].append({
                'model': model_name,
                'labeled_percentage': labeled_percentage,
                'pseudo_label_count': pseudo_label_count,
                'not_pseudo_label_count': not_pseudo_label_count,
                'pseudo_label_distribution': pseudo_label_distribution.tolist(),  # Convert to list for JSON serialization
                'threshold': self.threshold, # Store the threshold used
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

    # Function to get best hyperparameters
    def get_best_hyperparameters(self):
        return self.best_hyperparameters
    
    # Function to run the pipeline
    def run(self, labeled_percentages, n_bootstrap_samples=200):
        # Initialize dictionaries to store results
        self.results['training'] = []
        self.results['test'] = []
        self.results['bootstrap_test'] = []  # Added for bootstrap results
        self.results['pseudo_labels_info'] = []
        self.results['labeled_samples_info'] = []
        self.results['feature_importances'] = {}
        self.previous_val_performance = -np.inf


        for labeled_percentage in labeled_percentages:
            logger.info(f"Running for {labeled_percentage * 100}% labeled data...")

            try:
                # Prepare data
                X_train_final, y_train_final, X_val_labeled, y_val_labeled, X_test, y_test_encoded = self._prepare_data(labeled_percentage)

                # Train and evaluate each model
                for model_name in self.classifiers:
                    logger.info(f"Training {model_name}...")
                    self._train_and_evaluate(X_train_final, y_train_final, X_val_labeled, y_val_labeled, X_test, y_test_encoded, model_name, labeled_percentage)

                    # Evaluate on bootstrap samples
                    model = self.classifiers[model_name]
                    mean_std_metrics = self._evaluate_on_bootstrap_samples(model, X_test, y_test_encoded, n_bootstrap_samples)
                    logger.info(f"{model_name}: Mean and Standard Deviation on Bootstrap Samples: {mean_std_metrics}")

                    # Store bootstrap results
                    self.results['bootstrap_test'].append({
                        'model': model_name,
                        'labeled_percentage': labeled_percentage,
                        'bootstrap_metrics': mean_std_metrics
                    })


            except Exception as e:
                logger.error(f"Error during pipeline run: {e}")

        # Convert results to JSON serializable format
        converted_results = convert_numpy(self.results)

        # Save or process the results as needed
        with open('self_training_results_100_SMOTE.json', 'w') as file:
            json.dump(converted_results, file, indent=4)

        return self.results

# # Example usage
# pipeline = SelfTrainingPipeline()
# pipeline.set_data(df, label_column='category_bertopic', include_text=True)

# # Define the percentages of labeled data to be used in the loop
# labeled_data_percentages = [1.0, 0.8, 0.6, 0.4, 0.2, 0]
# results = pipeline.run(labeled_data_percentages)
