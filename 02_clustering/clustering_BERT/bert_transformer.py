# Importing necessary libraries and modules
import torch  # Deep learning library, primarily used here for its GPU support
import numpy as np  # Library for numerical operations with support for large, multi-dimensional arrays and matrices
import pandas as pd  # Data manipulation and analysis library, particularly useful for handling tabular data
from transformers import SentenceTransformer  # Classes from the Hugging Face Transformers library for working with BERT models
from torch.utils.data import DataLoader, Dataset  # PyTorch utilities for handling and batching datasets
from sklearn.preprocessing import normalize  # Function for normalizing vectors (e.g., embeddings)

# Definition of a custom dataset class to handle text data
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts  # Storing the input texts
    
    def __len__(self):
        return len(self.texts)  # Returns the number of texts
    
    def __getitem__(self, idx):
        return self.texts[idx]  # Returns the text at the specified index

# Class for generating embeddings using a MiniLM model
class MiniLMEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load the pre-trained MiniLM model from the Sentence Transformers library
        self.model = SentenceTransformer(model_name)
        
        # Determine the device to use (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move the model to the chosen device (GPU or CPU)
        self.model.to(self.device)
    
    def get_embeddings(self, texts):
        # Create a DataLoader to handle batching of texts
        data_loader = DataLoader(TextDataset(texts), batch_size=32)
        
        # List to store computed embeddings
        embeddings = []
        
        # Process each batch of texts without computing gradients (to save memory and improve speed)
        with torch.no_grad():
            for texts in data_loader:
                # Compute embeddings for the batch of texts
                batch_embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
                
                # Move embeddings to CPU and convert to NumPy format for easier handling
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batch embeddings into a single array
        return np.concatenate(embeddings)

# Function to generate normalized MiniLM embeddings from a DataFrame
def generate_minilm_embeddings(df, column_name='description'):
    # Initialize the MiniLMEmbedder
    embedder = MiniLMEmbedder()
    
    # Compute embeddings for the specified column in the DataFrame
    embeddings = embedder.get_embeddings(df[column_name].tolist())
    
    # Normalize the embeddings using the L2 norm
    normalized_embeddings = normalize(embeddings, norm='l2')
    
    return normalized_embeddings
