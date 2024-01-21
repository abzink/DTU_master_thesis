import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer

class LogTransformer(BaseEstimator, TransformerMixin):
    """Transformer to apply log1p transformation to numerical data."""
    
    def fit(self, X, y=None):
        """Nothing to do as there are no parameters to fit."""
        return self
    
    def transform(self, X):
        """Apply the log1p transformation."""
        return np.log1p(X)

class PreprocessingPipeline:
    """
    A preprocessing pipeline that handles categorical, numerical, and text features.
    All numerical features will undergo a log transformation and be scaled.
    """
    
    def __init__(self, categorical_columns, numerical_columns, text_column=None):
        """
        Initialize the pipeline.
        
        Parameters:
        - categorical_columns (list): Names of columns that are categorical.
        - numerical_columns (list): Names of columns that are numerical and will undergo log transformation and robust scaling.
        - text_column (str, optional): Name of the column that contains text data.
        """
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_column = text_column
        
        # Pipeline for one-hot encoding of categorical columns.
        cat_pipe = Pipeline([
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Pipeline for log transforming and scaling numerical columns.
        num_pipe = Pipeline([
            ('logtransform', LogTransformer()),
            ('scaler', RobustScaler())
        ])
        
        transformers = [
            ('cat', cat_pipe, self.categorical_columns),
            ('num', num_pipe, self.numerical_columns)
        ]
        
        # If a text column exists, it gets priority in only_text mode.
        if self.text_column:
            transformers.append(('text', TfidfVectorizer(min_df=0.005, max_df=0.99), self.text_column))
        
        # Combining the pipelines into one column transformer.
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            sparse_threshold=0 # ensures the output is a dense array
        )
        
    def fit(self, df, only_text=False):
        """
        Fits the preprocessing pipeline to the provided dataframe.
        If only_text is True, only the text column is processed.
        """
        if only_text and self.text_column:
            self.preprocessor.fit(df[[self.text_column]])
        else:
            self.preprocessor.fit(df)
        return self
        
    def transform(self, df, include_text=True, only_text=False):
        """
        Transforms the provided dataframe.
        If only_text is True, only the text column is processed.
        """
        if only_text and self.text_column:
            transformed_data = self.preprocessor.transform(df[[self.text_column]])
            column_names = [f"{self.text_column}_{i}" for i in range(transformed_data.shape[1])]
        else:
            transformed_data = self.preprocessor.transform(df)
            
            column_names = (list(self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_columns)) +
                            self.numerical_columns)
        
            if self.text_column and include_text:
                column_names.extend([f"{self.text_column}_{i}" for i in range(transformed_data.shape[1] - len(column_names))])
        
        self.transformed_df_ = pd.DataFrame(transformed_data, columns=column_names)
        return transformed_data

    def get_transformed_df(self):
        """Retrieve the transformed dataframe with headers."""
        return self.transformed_df_
    
    def get_feature_names(self):
        """
        Retrieve the names of the features after preprocessing.
        """
        feature_names = []

        # Get feature names for categorical columns
        if self.categorical_columns:
            cat_features = list(self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_columns))
            feature_names.extend(cat_features)

        # Add numerical column names as they are
        if self.numerical_columns:
            feature_names.extend(self.numerical_columns)

        # Get feature names for text column
        if self.text_column:
            text_features = [f"{self.text_column}_{i}" for i in range(self.preprocessor.named_transformers_['text'].get_feature_names().shape[0])]
            feature_names.extend(text_features)

        return feature_names

    def fit_transform(self, df, include_text=True, only_text=False):
        """
        Fit to the data, then transform it.
        If only_text is True, only the text column is processed.
        """
        self.fit(df, only_text=only_text)
        return self.transform(df, include_text=include_text, only_text=only_text)
