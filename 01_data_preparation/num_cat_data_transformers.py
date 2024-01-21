import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

class LogTransformer(BaseEstimator, TransformerMixin):
    """Transformer to apply log1p transformation to numerical data."""
    
    def fit(self, X, y=None):
        """Fit method (does nothing as this transformer doesn't need to learn any parameters)."""
        return self
    
    def transform(self, X):
        """Applies the log1p transformation and returns the transformed data."""
        return np.log1p(X)

class PreprocessingPipeline:
    """
    A preprocessing pipeline that handles categorical and numerical features.
    For numerical features, it provides an option to apply a log transformation before scaling.
    The transformed data can be returned as a numpy array or as a DataFrame with headers.
    """
    
    def __init__(self, categorical_columns, numerical_columns, log_transform_columns=[]):
        """
        Initialize the preprocessing pipeline.
        
        Parameters:
        - categorical_columns (list): List of column names that are categorical.
        - numerical_columns (list): List of column names that are numerical.
        - log_transform_columns (list): List of numerical column names to undergo log transformation.
        """
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.log_transform_columns = log_transform_columns
        self.non_log_transform_columns = list(set(numerical_columns) - set(log_transform_columns)) # columns not being log-transformed
        
        # Pipeline for one-hot encoding of categorical columns.
        self.categorical_pipe = Pipeline([
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])
        
        # Pipeline for numerical columns that will undergo log transformation and then be standardized.
        self.log_pipe = Pipeline([
            ('logtransform', LogTransformer()),
            ('scaler', RobustScaler())
        ])
        
        # Pipeline for numerical columns that will only be standardized.
        self.non_log_pipe = Pipeline([
            ('scaler', RobustScaler())
        ])
        
        # Combines the above pipelines into one column transformer.
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', self.categorical_pipe, self.categorical_columns),
                ('log', self.log_pipe, self.log_transform_columns),
                ('num', self.non_log_pipe, self.non_log_transform_columns)
            ],
            sparse_threshold=0 # ensures the output is a dense array
        )
        
    def fit(self, df):
        """
        Fits the preprocessing pipeline to the provided dataframe.
        """
        self.preprocessor.fit(df)
        
    def transform(self, df):
        """
        Transforms the provided dataframe using the fitted preprocessing pipeline.
        The transformed data is stored as both a numpy array and a DataFrame.
        
        Returns:
        - Processed data as a numpy array.
        """
        
        transformed_data = self.preprocessor.transform(df)
        column_names = (list(self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_columns)) +
                        self.log_transform_columns + self.non_log_transform_columns)
        
        self.transformed_df_ = pd.DataFrame(transformed_data, columns=column_names)
        return transformed_data
        
    def get_transformed_df(self):
        """
        Retrieve the transformed dataframe with headers.
        """
        return self.transformed_df_

    def fit_transform(self, df):
        """
        Fits the preprocessing pipeline to the provided dataframe and then transforms it.
        """
        self.fit(df)
        return self.transform(df)

# Usage:
# pipeline = PreprocessingPipeline(categorical_columns, numerical_columns, log_transform_columns)
# transformed_data = pipeline.fit_transform(df)
# transformed_df_with_headers = pipeline.get_transformed_df()
