# python bertopic_tuner.py data/processed_data_all.csv

import os
import sys
import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import multiprocessing
import hdbscan
from umap import UMAP

# from multiprocessing import util
# util.log_to_stderr(util.DEBUG)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def process_batch(param_batch, df, embeddings):
    """
    Process a batch of hyperparameters to fit the BERTopic model and generate results.
    """
    results = []
    for params in param_batch:
        n_neighbors, min_cluster_size, min_samples, min_topic_size = params

        # Set up UMAP and HDBSCAN models with given parameters
        umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, metric='cosine', random_state=42)
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=True)
        
        # Initialize BERTopic with given hyperparameters
        model = BERTopic(language="english", calculate_probabilities=False, verbose=True, nr_topics="auto", 
                         min_topic_size=min_topic_size, top_n_words=10, umap_model=umap_model, hdbscan_model=hdbscan_model)

        # Fit the BERTopic model and get topics for each document
        topics, _ = model.fit_transform(df['description'].tolist(), embeddings=embeddings)
        df['topic'] = topics

        # Extract topic representations and topic-to-document mappings
        topic_representation = model.get_topic_info()[['Topic', 'Count', 'Name', 'Representation']]
        topic_documents = model.get_document_info(df['description'])[['Document', 'Topic']]
        
        results.append((n_neighbors, min_cluster_size, min_samples, min_topic_size, topic_representation, topic_documents))

    return results

if __name__ == "__main__":
    # Check and retrieve input file from command-line argument
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <data_filepath>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    df = pd.read_csv(file_path)
    
    # Generate embeddings for the descriptions using SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(df['description'].tolist(), show_progress_bar=True)

    # Hyperparameter choices
    umap_neighbors = [15, 50, 100, 200]
    hdbscan_cluster_size = [10, 50, 100, 200]
    hdbscan_samples = [5, 10, 25, 50]
    bertopic_min_topic_size = [50, 100, 200, 500]

    # Generate all possible combinations of hyperparameters
    all_combinations = [(n, cs, ms, ts) for n in umap_neighbors for cs in hdbscan_cluster_size for ms in hdbscan_samples for ts in bertopic_min_topic_size]

    # Sample a random subset of hyperparameter combinations
    num_samples = 50
    sampled_combinations = random.sample(all_combinations, num_samples)

    # Split the sampled combinations into batches for parallel processing
    num_cores = multiprocessing.cpu_count()
    batch_size = len(sampled_combinations) // (num_cores * 2)
    param_batches = [sampled_combinations[i:i + batch_size] for i in range(0, len(sampled_combinations), batch_size)]

    # Use multiprocessing to fit BERTopic for each hyperparameter combination
    with multiprocessing.Pool(num_cores) as pool:
        all_results = pool.starmap(process_batch, [(batch, df, embeddings) for batch in param_batches])

    # Create an output directory if it doesn't exist
    output_directory = "data/bert_topics_40_vol_2"
    os.makedirs(output_directory, exist_ok=True)

    # Save the results to files
    for batch_results in all_results:
        for n_neighbors, min_cluster_size, min_samples, min_topic_size, topic_representation, topic_documents in batch_results:
            topic_repr_file = f"{output_directory}/topic_repr_neighbors_{n_neighbors}_cluster_{min_cluster_size}_samples_{min_samples}_topic_size_{min_topic_size}.csv"
            topic_doc_file = f"{output_directory}/topic_doc_neighbors_{n_neighbors}_cluster_{min_cluster_size}_samples_{min_samples}_topic_size_{min_topic_size}.csv"
        
            topic_representation.to_csv(topic_repr_file, index=False)
            topic_documents.to_csv(topic_doc_file, index=False)

    try:
        with multiprocessing.Pool(num_cores) as pool:
            all_results = pool.starmap(process_batch, [(batch, df, embeddings) for batch in param_batches])
            pool.close()  # Close the pool
            pool.join()   # Wait for the worker processes to terminate
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # If there are any cleanup actions, they can be performed here
        pass

