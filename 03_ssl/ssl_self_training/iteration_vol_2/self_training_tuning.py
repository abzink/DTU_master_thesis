# tune hyperparameters
# save predictions for each classifier

import sys
sys.path.append('../../') 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from ssl.ssl_data_transformers_robust import PreprocessingPipeline
from scipy.stats import uniform, randint
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from joblib import dump
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold


class SelfTrainingPipeline:

    def __init__(self, labeled_percentage=0.8, threshold = 0.75):
        self.random_seed = 42
        self.threshold = threshold
        self.labeled_percentage = labeled_percentage

        # Define classifiers and their parameter grids for grid search - run 1
#         self.classifiers = [
#             ('Random Forest', RandomForestClassifier(random_state=self.random_seed), {
#                 'n_estimators': randint(50, 500),
#                 'max_depth': [None] + list(randint(10, 30).rvs(5)),
#                 'min_samples_split': randint(4, 50),
#                 'min_samples_leaf': randint(2, 50),
#                 'bootstrap': [True, False]
#             }),
#             ('XGBoost', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=self.random_seed), {
#                 'learning_rate': uniform(0.01, 0.2),
#                 'n_estimators': randint(50, 300),
#                 'max_depth': randint(1, 10),
#                 'subsample': uniform(0.5, 0.4),
#                 'colsample_bytree': uniform(0.5, 0.4)
#             }),
#             ('LightGBM', lgb.LGBMClassifier(random_state=self.random_seed, importance_type='gain', verbose='-1'), {
#                 'learning_rate': uniform(0.01, 0.1),
#                 'n_estimators': randint(100, 300),
#                 'max_depth': randint(3, 20),
#                 'subsample': uniform(0.6, 0.3),
#                 'colsample_bytree': uniform(0.6, 0.3),
#                 'num_leaves': randint(20, 60)
#             }),
#             ('SVM', SVC(probability=True, random_state=self.random_seed), {
#                 'C': [0.01, 0.1, 1, 10, 100],
#                 'kernel': ['linear', 'rbf'],
#                 'gamma': ['scale', 'auto']
#             }),
#             ('KNN', KNeighborsClassifier(), {
#                 'n_neighbors': randint(5, 30),
#                 'weights': ['uniform', 'distance'],
#                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#                 'leaf_size': randint(20, 40),
#                 'p': [1, 2]
#             }),
#             ('Decision Tree', DecisionTreeClassifier(random_state=self.random_seed), {
#                 'criterion': ['gini', 'entropy'],
#                 'splitter': ['best', 'random'],
#                 'max_depth': [None] + list(randint(3, 20).rvs(5)),
#                 'min_samples_split': randint(5, 20),
#                 'min_samples_leaf': randint(2, 20),
#                 'max_features': ['sqrt', 'log2', None]
#             }),
#             ('Neural Network', MLPClassifier(max_iter=1000), {
#                 'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Different configurations of layers
#                 'activation': ['tanh', 'relu'],  # Activation function for the hidden layer
#                 'solver': ['sgd', 'adam'],  # Solver for weight optimization
#                 'alpha': 10.0 ** -np.arange(1, 7),  # L2 penalty (regularization term) parameter
#                 'learning_rate': ['constant','adaptive'],  # Learning rate schedule for weight updates
#                 'learning_rate_init': uniform(0.001, 0.01),  # The initial learning rate
# })
#         ]

        self.classifiers = [
            ('RandomForest', RandomForestClassifier(random_state=self.random_seed), {
                'n_estimators': randint(400, 500),
                'max_depth': randint(10, 20),
                'min_samples_split': randint(10, 20),
                'min_samples_leaf': randint(10, 15),
                'bootstrap': [True, False]
            }),
            ('XGBoost', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=self.random_seed), {
                'learning_rate': uniform(0.1, 0.2),
                'n_estimators': randint(200, 300),
                'max_depth': randint(5, 10),
                'subsample': uniform(0.6, 0.4), # Generates values in the range [0.6, 1.0]
                'colsample_bytree': uniform(0.5, 0.4)
            }),
            ('LightGBM', lgb.LGBMClassifier(random_state=self.random_seed, importance_type='gain', verbose='-1'), {
                'learning_rate': uniform(0.05, 0.1),
                'n_estimators': randint(200, 300),
                'max_depth': randint(5, 10),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.3),
                'num_leaves': randint(30, 50)
            }),
            ('SVM', SVC(probability=True, random_state=self.random_seed), {
                'C': [50, 100, 200, 300],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }),
            # ('KNN', KNeighborsClassifier(), {
            #     'n_neighbors': randint(15, 25),
            #     'weights': ['uniform', 'distance'],
            #     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            #     'leaf_size': randint(20, 40),
            #     'p': [1, 2]
            # }),
            ('DecisionTree', DecisionTreeClassifier(random_state=self.random_seed), {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': randint(10, 15),
                'min_samples_split': randint(10, 20),
                'min_samples_leaf': randint(10, 20),
                'max_features': ['sqrt', 'log2', None]
            }),
            ('Neural Network', MLPClassifier(max_iter=1000), {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Different configurations of layers
                'activation': ['tanh', 'relu'],  # Activation function for the hidden layer
                'solver': ['sgd', 'adam'],  # Solver for weight optimization
                'alpha': 10.0 ** -np.arange(1, 7),  # L2 penalty (regularization term) parameter
                'learning_rate': ['constant','adaptive'],  # Learning rate schedule for weight updates
                'learning_rate_init': uniform(0.005, 0.01),  # The initial learning rate
            })
        ]


    def set_data(self, df, categorical_columns, numerical_columns, text_column=None, include_text=True):
        self.data = df
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_column = text_column
        self.include_text = include_text

    def _create_balanced_subsample(self, X, y, labeled_percentage):
        """
        Create a balanced subsample of the labeled data with the given percentage.
        """
        from sklearn.model_selection import StratifiedShuffleSplit
        
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=labeled_percentage, random_state=self.random_seed)
        for train_index, _ in splitter.split(X, y):
            return X.iloc[train_index], y.iloc[train_index]
        
    def _prepare_data(self):
        # Split the labeled data into labeled and unlabeled datasets
        labeled_data = self.data[self.data['category_bertopic'] != '-1']
        unlabeled_data = self.data[self.data['category_bertopic'] == '-1']

        X_labeled = labeled_data.drop(columns='category_bertopic')
        y_labeled = labeled_data['category_bertopic']
        X_unlabeled = unlabeled_data.drop(columns='category_bertopic')

        # Split labeled data into training, validation, and test datasets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_labeled, y_labeled, test_size=0.2, stratify=y_labeled, random_state=self.random_seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=self.random_seed)  # 0.25 x 0.8 = 0.2
        
        # Create a balanced subsample of labeled data
        X_train, y_train = self._create_balanced_subsample(X_train, y_train, self.labeled_percentage)

        # Add the unlabeled data to the training set
        X_train_print = pd.concat([X_train, X_unlabeled])
        y_train_print = pd.concat([y_train, pd.Series([-1]*len(X_unlabeled))])

        # Print the number of training samples and unlabeled samples
        print("Number of training samples:", len(X_train_print))
        print("Unlabeled samples in training set:", sum(1 for x in y_train_print if x == -1))

        # Initialize the preprocessing pipeline
        self.pipeline = PreprocessingPipeline(self.categorical_columns, self.numerical_columns, text_column=self.text_column)

        # Process the datasets
        X_train_processed = self.pipeline.fit_transform(X_train, include_text=self.include_text)
        X_val_processed = self.pipeline.transform(X_val, include_text=self.include_text)
        X_test_processed = self.pipeline.transform(X_test, include_text=self.include_text)
        X_unlabeled_processed = self.pipeline.transform(X_unlabeled, include_text=self.include_text)

        # Fit the LabelEncoder on all available labels (excluding the unlabeled)
        self.le = LabelEncoder()
        self.le.fit(y_labeled)

        # Encode the labels
        y_train_encoded = self.le.transform(y_train)
        y_val_encoded = self.le.transform(y_val)
        y_test_encoded = self.le.transform(y_test)

        # Apply SMOTE to balance the training data
        smote = SMOTE(random_state=self.random_seed)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train_encoded)

        # Set the instance variables for use in training and evaluation
        self.X_train_processed = X_train_balanced
        self.y_train_encoded = y_train_balanced
        self.X_val_processed = X_val_processed
        self.y_val_encoded = y_val_encoded
        self.X_test_processed = X_test_processed
        self.y_test_encoded = y_test_encoded
        self.X_unlabeled_processed = X_unlabeled_processed


    # def _grid_search(self, classifier, param_grid):
    #     grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
    #     grid_search.fit(self.X_train_processed, self.y_train_encoded)
    #     return grid_search.best_estimator_
    
    def _random_search(self, classifier, param_distributions):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rand_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_distributions, 
                                         n_iter=5, cv=cv, n_jobs=2, random_state=self.random_seed)
        rand_search.fit(self.X_train_processed, self.y_train_encoded)
        return rand_search.best_estimator_
    

    def _self_training(self, classifier):
        """This method takes a classifier and applies self-training to it using the combined labeled and unlabeled data."""
        # Initialize self-training model
        self_training_model = SelfTrainingClassifier(classifier, criterion='threshold', threshold=self.threshold, verbose=True)

        # Combine the data
        combined_X = np.vstack((self.X_train_processed, self.X_unlabeled_processed))
        combined_y = np.concatenate([self.y_train_encoded, [-1] * len(self.X_unlabeled_processed)])

        # Fit and pseudo-label
        self_training_model.fit(combined_X, combined_y)
        
        return self_training_model
    
    def _evaluate_model(self, model, name):
        """This function evaluates the trained model on the validation and test datasets and returns the evaluation metrics."""
        # Evaluate on validation set
        y_val_pred = model.predict(self.X_val_processed)
        val_acc = accuracy_score(self.y_val_encoded, y_val_pred)
        val_report = classification_report(self.y_val_encoded, y_val_pred, output_dict=True, zero_division=0)
        val_conf_matrix = confusion_matrix(self.y_val_encoded, y_val_pred).tolist()

        # Evaluate on test set
        y_test_pred = model.predict(self.X_test_processed)
        test_acc = accuracy_score(self.y_test_encoded, y_test_pred)
        test_report = classification_report(self.y_test_encoded, y_test_pred, output_dict=True, zero_division=0)
        test_conf_matrix = confusion_matrix(self.y_test_encoded, y_test_pred).tolist()

        # [TBC and added in run func] Correct the count for pseudo-labeled instances
        pseudo_labels = model.transduction_[-len(self.X_unlabeled_processed):]

        return {
            'validation': {
                "Accuracy": val_acc,
                "Report": val_report,
                "Confusion Matrix": val_conf_matrix
            },
            'test': {
                "Accuracy": test_acc,
                "Report": test_report,
                "Confusion Matrix": test_conf_matrix
            },
            'predictions': {
                'val': y_val_pred,
                'test': y_test_pred
            },
            'pseudo_labeled_count': pseudo_labels
        }

    def _store_results(self, validation_results, test_results, feature_importances):
        """This function saves the evaluation results and feature importances to CSV files."""
        # Save evaluation results to CSV - validation
        validation_results.sort(key=lambda x: x["Accuracy"], reverse=True)
        pd.DataFrame(validation_results).to_csv('self_training_val_results.csv', index=False)

        # Save evaluation results to CSV - test
        test_results.sort(key=lambda x: x['Test Accuracy'], reverse=True)
        pd.DataFrame(test_results).to_csv('self_training_test_results.csv', index=False)

        # Save feature importances to CSV if available
        if feature_importances:
            feature_names = self.pipeline.get_transformed_df().columns
            pd.DataFrame(feature_importances, index=feature_names).to_csv('feature_importances.csv', index=True)

    def _save_hyperparameters(self, best_hyperparameters):
        """ This function saves the best hyperparameters using joblib."""
        dump(best_hyperparameters, 'best_hyperparameters.joblib')


    def _save_predictions(self, predictions):
        for name, pred in predictions.items():
            np.save(f"{name}_predictions.n py", pred)


    def run(self):
        # Splitting data into labeled and unlabeled samples
        self._prepare_data()

        # Initialize storage structures
        feature_importances = {}
        best_hyperparameters = {}
        validation_results = []
        test_results = []
        predictions = {}

        # Iterate over classifiers
        for name, base_classifier, param_distributions in tqdm(self.classifiers, desc="Training Classifiers"):
            
            # Perform hyperparameter optimization
            best_classifier = self._random_search(base_classifier, param_distributions)
            best_hyperparameters[name] = best_classifier.get_params()

            # Check and collect feature importances if available
            if hasattr(best_classifier, 'feature_importances_'):
                feature_importances[name] = best_classifier.feature_importances_

            # Proceed only if the classifier supports probability prediction
            if hasattr(best_classifier, 'predict_proba'):
                # Apply self-training
                self_training_model = self._self_training(best_classifier)

                # Evaluate the model on both validation and test sets
                evaluation_results = self._evaluate_model(self_training_model, name)
                
                # Store validation results
                val_results_dict = {
                    "Classifier": name,
                    "Accuracy": evaluation_results['validation']["Accuracy"],
                    "Evaluation Report": evaluation_results['validation']["Report"],
                    "Confusion Matrix": evaluation_results['validation']["Confusion Matrix"],
                    "Best Parameters": best_hyperparameters[name]
                }
                validation_results.append(val_results_dict)
                
                # Store test results
                test_results_dict = {
                    "Classifier": name,
                    "Test Accuracy": evaluation_results['test']["Accuracy"],
                    "Test Report": evaluation_results['test']["Report"],
                    "Test Confusion Matrix": evaluation_results['test']["Confusion Matrix"]
                }
                test_results.append(test_results_dict)
                
                # Save model predictions
                predictions[name] = evaluation_results['predictions']

                # Extract pseudo-labels from the results
                pseudo_labels = evaluation_results['pseudo_labeled_count']

                # Filter out the -1 labels (unassigned by the classifier)
                valid_indices = pseudo_labels != -1
                valid_pseudo_labels = pseudo_labels[valid_indices]

                # Map valid pseudo-labels to original unlabeled data
                unlabeled_data_indices = self.data[self.data['category_bertopic'] == '-1'].index[valid_indices]
                unlabeled_data_with_pseudo = self.data.loc[unlabeled_data_indices].copy()
                unlabeled_data_with_pseudo['pseudo_label'] = self.le.inverse_transform(valid_pseudo_labels)

                # Select original columns to be stored along with the pseudo-labels
                original_columns = self.data.columns.tolist()    # Include your desired columns here
                final_data_to_save = unlabeled_data_with_pseudo[original_columns + ['pseudo_label']]

                # Save to file pseudo-labels for each classifier
                final_data_to_save.to_csv(f'{name}_pseudo_labels.csv', index=False)

      
        # Save the best hyperparameters
        self._save_hyperparameters(best_hyperparameters)

        # Save predictions
        # self._save_predictions(predictions)

        # Store evaluation results as CSV files
        self._store_results(validation_results, test_results, feature_importances)

        return validation_results, test_results



""""labeled_data_percentage determines how much of the labeled data is available for the whole pipeline, 
while labeled_percentage is used within a method to further control the portion of 
this available labeled data to create your training subset."""


# without text column

# pipeline = SelfTrainingPipeline()
# pipeline.set_data(df, categorical_columns, numerical_columns, include_text=False)
# results = pipeline.run()

# with text column

# pipeline = SelfTrainingPipeline()
# pipeline.set_data(df, categorical_columns, numerical_columns, text_column='your_text_column', include_text=True)
# results = pipeline.run()