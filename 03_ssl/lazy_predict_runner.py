# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from lazypredict.Supervised import LazyClassifier
import sys
# Add the folder path to sys.path
# Assuming my_notebook.ipynb and mymodule are in the same parent directory (project)
sys.path.append("../01_data_preparation")
from num_cat_data_transformers import PreprocessingPipeline  # Make sure this is the updated version of PreprocessingPipeline

# Define the LazyPredictPipeline class
class LazyPredictPipeline:
    # Initialize the pipeline with data, target column, columns types, log-transformed columns, test size, and number of cross-validation folds
    def __init__(self, df, target_col, categorical_columns, numerical_columns, log_transform_columns, test_size=0.2, n_folds=5):
        self.df = df
        self.target_col = target_col
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.log_transform_columns = log_transform_columns
        self.test_size = test_size
        self.n_folds = n_folds

    # Define the run method to execute the pipeline
    def run(self):
        # Split data into features and target
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

        # Initialize the preprocessing pipeline and preprocess the data
        preprocessor = PreprocessingPipeline(self.categorical_columns, self.numerical_columns, self.log_transform_columns)
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        # Run LazyPredict on the preprocessed data
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models, predictions = clf.fit(X_train_preprocessed, X_test_preprocessed, y_train, y_test)

        # Initialize a dictionary to store the mean cross-validation scores for each classifier
        cv_scores = {}
        all_classifiers = clf.classifiers

        # For each classifier in LazyPredict, compute its cross-validation score
        for model_tuple in all_classifiers:
            model_name, model_class = model_tuple

            # Skip the StackingClassifier since it requires additional parameters
            if model_name == "StackingClassifier":
                continue

            # Instantiate the current classifier
            model_instance = model_class()

            # Try computing its cross-validation score; if an error occurs, store the error message
            try:
                cv_score = cross_val_score(model_instance, X_train_preprocessed, y_train, cv=self.n_folds).mean()
                cv_scores[model_name] = cv_score
            except Exception as e:
                cv_scores[model_name] = str(e)

        # Add the computed cross-validation scores to the models dataframe
        models['Cross_Val_Score'] = models.index.map(cv_scores)

        # Return the models dataframe and predictions dictionary
        return models, predictions
