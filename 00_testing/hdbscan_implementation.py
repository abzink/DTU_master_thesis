import umap
import hdbscan
import numpy as np
from tqdm import trange
import random
import pandas as pd
from sentence_transformers import SentenceTransformer


# Sentence Embeddings

def generate_embeddings(df, column_name):
    """
    Generate sentence embeddings for a specified column in a DataFrame using different models.
    
    Parameters:
    - df: DataFrame containing the data.
    - column_name: Name of the column from which sentences are extracted.
    
    Returns:
    - Dictionary containing embeddings for each model.
    """
    
    # Extract sentences from the specified column
    sentences = df[column_name].tolist()
    
    # Define the models
    models = {
        'st1': SentenceTransformer('all-mpnet-base-v2'),
        'st2': SentenceTransformer('all-MiniLM-L6-v2'),
        'st3': SentenceTransformer('paraphrase-mpnet-base-v2')
    }
    
    # Generate embeddings for each model
    embeddings = {}
    for model_name, model in models.items():
        embeddings[model_name] = model.encode(sentences)
    
    return embeddings

# Usage
# data_sample = pd.read_csv('../data/processed/data_sample.csv')
# embeddings = generate_embeddings(data_sample, 'description')

# Accessing embeddings for a specific model, e.g., 'st1'
# embeddings_st1 = embeddings['st1']

# Cluster

def generate_clusters(sentence_embeddings,
                      n_neighbors,
                      n_components, 
                      min_cluster_size,
                      random_state = None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP.
    
    Parameters:
    - sentence_embeddings : ndarray
        Precomputed sentence embeddings to cluster.
    - n_neighbors : int
        The size of local neighborhood used for manifold approximation.
    - n_components : int
        The dimension of the space to embed into.
    - min_cluster_size : int
        The minimum size of clusters.
    - random_state : int, RandomState instance, default=None
        Determines the random number generation for reproducibility.
        
    Returns:
    - clusters : HDBSCAN
        HDBSCAN clustering object.
    """
    
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                metric='cosine', 
                                random_state=random_state)
                            .fit_transform(sentence_embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               metric='euclidean', 
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters

# Evaluate

def score_clusters(clusters, prob_threshold = 0.05):
    """
    Returns the label count and cost of a given cluster supplied from running hdbscan
    """
    
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num)
    
    return label_count, cost
 
# Hyperparameter Tuning

def random_search(embeddings, space, num_evals):
    """
    Randomly search hyperparameter space and limited number of times 
    and return a summary of the results
    """
    
    results = []
    
    for i in trange(num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])
        
        clusters = generate_clusters(embeddings, 
                                     n_neighbors = n_neighbors, 
                                     n_components = n_components, 
                                     min_cluster_size = min_cluster_size, 
                                     random_state = 42)
    
        label_count, cost = score_clusters(clusters, prob_threshold = 0.05)
                
        results.append([i, n_neighbors, n_components, min_cluster_size, 
                        label_count, cost])
    
    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components', 
                                               'min_cluster_size', 'label_count', 'cost'])
    
    return result_df.sort_values(by='cost')