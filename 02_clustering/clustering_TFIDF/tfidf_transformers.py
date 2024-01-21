from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class TfidfTransformer:
    """
    Class for transforming text data into TF-IDF vectors.

    Parameters to modify:
    max_df: Ignore terms that have a document frequency strictly higher than the given threshold.
    min_df: Ignore terms that have a document frequency strictly lower than the given threshold.
    """
    def __init__(self, max_features=None, min_df=0.005, max_df=0.99):
        self.vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)
        
    def fit_transform(self, df: pd.DataFrame, column: str):
        descriptions = df[column].dropna()
        tfidf_matrix = self.vectorizer.fit_transform(descriptions)
        result = tfidf_matrix.toarray()
        feature_names = self.vectorizer.get_feature_names_out()

        print("Result Type: ", type(result))  # should be <class 'numpy.ndarray'>
        print("Feature Names Type: ", type(feature_names))  # should be <class 'list'>
        
        return result, feature_names
        