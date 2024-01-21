# Import necessary libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from ssl.ssl_data_transformers_robust import PreprocessingPipeline
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_utils import save_figure
import json
from sklearn.utils import resample
import time
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define a class for Label Propagation and Spreading
class LabelPropagationAndSpreading:
    def __init__(self, df, categorical_cols, numerical_cols, target_col, text_col=None, use_text=False, 
                 output_file='unlabeled_predictions.csv', eval_dir='saved_models_final', balance=True, labeled_percentage=0.2, verbose=True,
                 random_state=42):
        """
            Constructor for the LabelPropagationAndSpreading class.

            Parameters:
            - df: DataFrame containing the dataset.
            - categorical_cols: List of names of categorical columns.
            - numerical_cols: List of names of numerical columns.
            - target_col: Name of the target column.
            - text_col: Name of the text column, if any.
            - use_text: Boolean indicating whether to use text data in the model.
            - output_file: Path to the file where predictions for unlabeled data will be saved.
            - eval_dir: Directory where evaluation results are saved.
            - balance: Boolean indicating whether to balance the dataset using SMOTE.
            - labeled_percentage: Percentage of data to be considered as labeled.
            - verbose: Boolean indicating whether to display verbose output.
            - random_state: Seed for random number generator.

            Initializes various properties used in the learning process.
        """
        # Initialize class attributes
        self.df = df
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.target_col = target_col
        self.text_col = text_col
        self.use_text = use_text
        self.output_file = output_file
        self.models_dir = eval_dir
        self.balance = balance
        self.verbose = verbose
        self.labeled_percentage = labeled_percentage
        self.random_state = random_state
        self.category_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}  # Add a mapping for categories.

    # Method to balance the data using BorderlineSMOTE (Synthetic Minority Over-sampling Technique)
    def balance_data(self, X, y):
        """
            Balances the data using BorderlineSMOTE.

            Parameters:
            - X: Features of the training data.
            - y: Labels of the training data.

            Returns:
            Balanced features and labels.

            This method applies BorderlineSMOTE to balance the class distribution in the training data.
        """
        smote = SMOTE(random_state=self.random_state, n_jobs=None)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        # Log if NaN values are found after applying SMOTE
        if np.isnan(X_balanced).any():
            logger.info("NaN values found after applying SMOTE.")
        else:
            logger.info("No NaN values found after applying SMOTE.")
        return X_balanced, y_balanced

    # Main method to perform semi-supervised learning using Label Propagation and Spreading
    
    def perform_learning(self):
        """
            Main method to perform the learning process. It involves data preprocessing,
            splitting, model training and evaluation, and saving the results.

            Returns:
            Dictionary of best models for each algorithm.

            This method performs the following steps:
            1. Prepares the data for training and evaluation.
            2. Balances the training data if specified.
            3. Trains and evaluates Label Propagation and Spreading models.
            4. Saves predictions for unlabeled data.
            5. Returns a dictionary containing the best models.
        """
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        # Split the dataset into labeled and unlabeled data based on the target column value
        labeled_data = self.df[self.df[self.target_col] != '-1']
        unlabeled_data = self.df[self.df[self.target_col] == '-1']

        # Split labeled data into features and target
        X_labeled = labeled_data.drop([self.target_col], axis=1)
        y_labeled = labeled_data[self.target_col]

        # Split labeled data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, stratify=y_labeled, random_state=self.random_state)

        # Reduce the size of the labeled training data if needed
        if self.labeled_percentage < 1.0:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=self.labeled_percentage, stratify=y_train, random_state=self.random_state
            )

        # Apply the preprocessing pipeline to the data
        pipeline = PreprocessingPipeline(categorical_columns=self.categorical_cols, 
                                        numerical_columns=self.numerical_cols, 
                                        text_column=self.text_col if self.use_text else None)
        X_train = pipeline.fit_transform(X_train, include_text=self.use_text)
        X_test = pipeline.transform(X_test, include_text=self.use_text)
        X_unlabeled = pipeline.transform(unlabeled_data.drop([self.target_col], axis=1), include_text=self.use_text)

        # Balance the training data if needed
        if self.balance:
            X_train, y_train = self.balance_data(X_train, y_train)

        # Configuration for models (LabelPropagation and LabelSpreading) with their parameters
        models_params = {
             'LabelPropagation': {
                 'model': LabelPropagation(max_iter=100),
                 'params': {'gamma': [10, 20, 100, 200, 500], 'kernel': ['rbf'], 'n_neighbors': [3, 5, 7, 9]}
             },
            'LabelSpreading': {
                'model': LabelSpreading(),
                'params': {'gamma': [20, 100, 200], 'kernel': ['rbf'], 'n_neighbors': [5, 7, 9], 'alpha': np.linspace(0.2, 0.8, 4)}
            }
        }

        # Train models and get predictions for unlabeled data
        best_models, unlabeled_data_with_predictions = self.train_and_evaluate_models(X_train, y_train, X_test, y_test, models_params, X_unlabeled, unlabeled_data)

        # Save predictions for unlabeled data
        unlabeled_data_with_predictions.to_csv(self.output_file, index=False)

        return best_models

    def train_and_evaluate_models(self, X_train, y_train, X_test, y_test, models_params, X_unlabeled, unlabeled_data):
        """
            Evaluates the model on multiple bootstrap samples.

            Parameters:
            - model: Trained model to be evaluated.
            - X_original_test: Test features.
            - y_original_test: Test labels.
            - n_bootstrap_samples: Number of bootstrap samples to be used for evaluation.

            Returns:
            Mean and standard deviation of accuracy, precision, recall, and f1-score.

            This method evaluates the model on multiple bootstrap samples and calculates
            mean and standard deviation for accuracy, precision, recall, and f1-score.
        """
        
        best_models = {}
        labeled_percentage_str = f"{int(self.labeled_percentage * 100)}%"  # Format the labeled percentage for display

        for model_name, model_details in models_params.items():
            if self.verbose:
                logger.info(f"Training and evaluating {model_name}...")

            # Training the model
            start_time_training = time.time()
            trained_model_info = self.train_model(model_name, model_details['model'], model_details['params'], X_train, y_train, X_test, y_test)
            end_time_training = time.time()
            training_time = end_time_training - start_time_training
            logger.info(f"Training time for {model_name}: {training_time} seconds")

            trained_model_info['training_time'] = training_time

            # Perform bootstrap test
            bootstrap_mean_metrics, bootstrap_std_metrics = self._evaluate_on_bootstrap_samples(trained_model_info['model'], X_test, y_test, n_bootstrap_samples=200)

            # Prepare the dictionary to store bootstrap results
            bootstrap_results = {}
            for metric in bootstrap_mean_metrics.keys():
                bootstrap_results[f'mean_{metric}'] = bootstrap_mean_metrics[metric]
                bootstrap_results[f'std_{metric}'] = bootstrap_std_metrics[metric]

            trained_model_info['bootstrap_results'] = bootstrap_results

            # Retrieve the best model object
            best_model = trained_model_info['model']

            # Predict on unlabeled data
            predictions = best_model.predict(X_unlabeled)
            unlabeled_data.loc[:, f'{model_name}_predictions'] = predictions

            # Predict probabilities for unlabeled data (if available)
            predicted_probabilities = best_model.predict_proba(X_unlabeled)
            max_probs = np.max(predicted_probabilities, axis=1)

            # Plot and save the distribution of prediction confidence
            fig, ax = plt.subplots()
            sns.histplot(max_probs, bins=20, edgecolor='black', ax=ax)
            title = f'Confidence Distribution for {model_name} ({labeled_percentage_str} Labeled Data)'
            ax.set_title(title)
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Frequency')
            save_figure(fig, f'confidence_distribution_{model_name}_{labeled_percentage_str}')
            plt.close(fig)

            # Analyze and plot the distribution of predicted classes
            predicted_classes = np.argmax(predicted_probabilities, axis=1)
            unique, counts = np.unique(predicted_classes, return_counts=True)
            # Mapping the numeric classes to letters
            class_labels = [self.category_mapping.get(x, x) for x in unique]  # Map numbers to letters  
            class_distribution_df = pd.DataFrame({'Class': class_labels, 'Frequency': counts})
            fig, ax = plt.subplots()
            sns.barplot(x='Class', y='Frequency', data=class_distribution_df, ax=ax)
            title = f'Predicted Category Distribution for {model_name} ({labeled_percentage_str} Labeled Data)'
            ax.set_title(title)
            ax.set_xlabel('Category')
            ax.set_ylabel('Frequency')
            save_figure(fig, f'predicted_class_distribution_{model_name}_{labeled_percentage_str}')
            plt.close(fig)

            # Save model and results
            self.save_model_and_results(trained_model_info, model_name, self.labeled_percentage)

            # Store the trained model along with its information
            best_models[model_name] = trained_model_info

        return best_models, unlabeled_data
    
        # Method to evaluate model performance on bootstrap samples
    def _evaluate_on_bootstrap_samples(self, model, X_original_test, y_original_test, n_bootstrap_samples):
        """
            Evaluates the model on multiple bootstrap samples.

            Parameters:
            - model: Trained model to be evaluated.
            - X_original_test: Test features.
            - y_original_test: Test labels.
            - n_bootstrap_samples: Number of bootstrap samples to be used for evaluation.

            Returns:
            Mean and standard deviation of accuracy, precision, recall, and f1-score.

            This method evaluates the model on multiple bootstrap samples and calculates
            mean and standard deviation for accuracy, precision, recall, and f1-score.
        """
        bootstrap_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        # Perform evaluations on bootstrap samples
        for _ in range(n_bootstrap_samples):
            X_test, y_test = resample(X_original_test, y_original_test, replace=True, n_samples=len(X_original_test))
            y_test_pred = model.predict(X_test)
            bootstrap_metrics['accuracy'].append(accuracy_score(y_test, y_test_pred))
            bootstrap_metrics['precision'].append(precision_score(y_test, y_test_pred, average='weighted', zero_division=0))
            bootstrap_metrics['recall'].append(recall_score(y_test, y_test_pred, average='weighted', zero_division=0))
            bootstrap_metrics['f1'].append(f1_score(y_test, y_test_pred, average='weighted', zero_division=0))

        # Calculate mean and standard deviation for each metric
        mean_metrics = {metric: np.mean(values) for metric, values in bootstrap_metrics.items()}
        std_metrics = {metric: np.std(values) for metric, values in bootstrap_metrics.items()}

        return mean_metrics, std_metrics

    def train_model(self, model_name, model, params, X_train, y_train, X_test, y_test):
        """
            Trains a machine learning model with hyperparameter tuning using RandomizedSearchCV.

            Parameters:
            - model_name: Name of the machine learning model being trained.
            - model: The base machine learning model object to be tuned.
            - params: Hyperparameter search space for RandomizedSearchCV.
            - X_train: Features of the training data.
            - y_train: Labels of the training data.
            - X_test: Features of the test data.
            - y_test: Labels of the test data.

            Returns:
            A dictionary containing the trained model, training and evaluation metrics, and best hyperparameters.

            This function performs the following steps:
            1. Initializes cross-validation with stratified k-fold.
            2. Uses RandomizedSearchCV to perform hyperparameter tuning.
            3. Retrieves the best model with optimized hyperparameters.
            4. Evaluates the best model on the test data and computes various metrics.
            5. Returns a dictionary containing model information and evaluation results.
        """
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        random_search = RandomizedSearchCV(model, params, cv=cv, scoring='accuracy', n_iter=1, n_jobs=1, verbose=1)
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        metrics = {
            'cv_best_score': random_search.best_score_,
            'cv_best_mean_test_score': random_search.cv_results_['mean_test_score'].max(),  
            'cv_best_std_test_score': random_search.cv_results_['std_test_score'][random_search.cv_results_['mean_test_score'].argmax()],
            'test_score': best_model.score(X_test, y_test),
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'kappa_score': cohen_kappa_score(y_test, y_pred),
            'mcc_score': matthews_corrcoef(y_test, y_pred),
            'error_rate': 1 - accuracy_score(y_test, y_pred),
            'best_params': random_search.best_params_,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }

        return {'model_name': model_name, 'model': best_model, **metrics}

    def save_model_and_results(self, trained_model_info, model_name, labeled_percentage):
        """
            Saves trained models and evaluation results.

            Parameters:
            - trained_model_info: Dictionary containing trained model information and results.
            - model_name: Name of the machine learning model.
            - labeled_percentage: Percentage of data considered as labeled.

            This method saves the trained model (if available) and evaluation results to files.
        """
        # Check if the 'model' key exists before proceeding
        model = trained_model_info.pop('model', None)

        # Format the labeled percentage for filename
        labeled_percentage_str = f"{int(labeled_percentage * 100)}"

        # Save the model separately if it exists
            # if model is not None:
            #     model_filename = os.path.join(self.models_dir, f'{model_name}_model_{labeled_percentage_str}.joblib')
            #     joblib.dump(model, model_filename)

        # Prepare results dictionary
        results_with_bootstrap = {
                "training_results": trained_model_info,
                "training_time": trained_model_info.get('training_time', 'Not recorded'),
                "bootstrap_results": trained_model_info.get('bootstrap_results', {})
            }

        # Save results to JSON file
        json_filename = os.path.join(self.models_dir, f'{model_name}_results_{labeled_percentage_str}.json')
        with open(json_filename, 'w') as json_file:
            json.dump(results_with_bootstrap, json_file, indent=4)

# Example usage:
# learner = LabelPropagationAndSpreading(df, categorical_cols, numerical_cols, target_col, text_col, use_text, 'predictions.csv', 'saved_models')
# best_models = learner.perform_learning()
