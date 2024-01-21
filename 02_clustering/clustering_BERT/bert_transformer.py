# Importing necessary libraries and modules
import torch  # Main library for deep learning tasks
import numpy as np  # Library for handling multidimensional arrays and matrices
import pandas as pd  # Library for data manipulation and analysis
from transformers import BertModel, BertTokenizer, BertConfig  # Importing necessary classes from transformers library
from torch.utils.data import DataLoader, Dataset  # Utilities for handling datasets and creating batches of data
from sklearn.preprocessing import normalize # For normalizing vectors

# Definition of a custom Dataset class to hold the text data
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import normalize

# Custom Dataset class remains the same
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

# Updated class for MiniLM Embedding
class MiniLMEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Loading pre-trained MiniLM model
        self.model = SentenceTransformer(model_name)
        
        # Setting device to GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Moving the model to the chosen device
        self.model.to(self.device)
    
    def get_embeddings(self, texts):
        # Creating a DataLoader for the input texts
        data_loader = DataLoader(TextDataset(texts), batch_size=32)
        
        # List to hold the computed embeddings
        embeddings = []
        
        # Looping over batches of data and computing embeddings
        with torch.no_grad():
            for texts in data_loader:
                # Computing embeddings directly
                batch_embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
                
                # Moving to CPU and converting to NumPy format
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenating embeddings from all batches and returning
        return np.concatenate(embeddings)

# Function to generate and return MiniLM embeddings
def generate_minilm_embeddings(df, column_name='description'):
    # Initialize the MiniLMEmbedder
    embedder = MiniLMEmbedder()
    
    # Getting embeddings for the specified column of the DataFrame
    embeddings = embedder.get_embeddings(df[column_name].tolist())
    
    # Normalize the embeddings using L2 norm
    normalized_embeddings = normalize(embeddings, norm='l2')
    
    return normalized_embeddings

# class TextDataset(Dataset):
#     def __init__(self, texts):
#         # Storing the texts passed to this object
#         self.texts = texts
    
#     def __len__(self):
#         # Returning the number of texts in the dataset
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         # Returning the text at the given index
#         return self.texts[idx]

# # Definition of a class to perform BERT embedding
# class BertEmbedder:
#     def __init__(self, model_name='bert-base-uncased', max_length=128):
#         # Loading pre-trained BERT model and tokenizer
#         self.config = BertConfig.from_pretrained(model_name)
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertModel.from_pretrained(model_name, config=self.config)
        
#         # Setting device to GPU if available, otherwise CPU
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Moving the model to the chosen device
#         self.model.to(self.device)
        
#         # Setting the model to evaluation mode
#         self.model.eval()
        
#         # Setting maximum sequence length for tokenization
#         self.max_length = max_length
    
#     def encode_texts(self, texts):
#         # Tokenizing and encoding the batch of texts and moving the tensors to the correct device
#         inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
#         return inputs.to(self.device)
        
#     def get_embeddings(self, texts):
#         # Creating a DataLoader for the input texts
#         data_loader = DataLoader(TextDataset(texts), batch_size=32)
        
#         # List to hold the computed embeddings
#         embeddings = []
        
#         # Looping over batches of data and computing embeddings
#         with torch.no_grad():
#             for texts in data_loader:
#                 inputs = self.encode_texts(texts)
#                 outputs = self.model(**inputs)
                
#                 # Extracting [CLS] embeddings, moving to CPU and converting to NumPy format
#                 embeddings.append(outputs['last_hidden_state'][:, 0, :].cpu().numpy())
        
#         # Concatenating embeddings from all batches and returning
#         return np.concatenate(embeddings)

# # Function to generate and return BERT embeddings
# def generate_bert_embeddings(df, column_name='description'):
#     # Initializing the BertTokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
#     # Finding the maximum length of the text entries in terms of the number of tokens
#     max_length = df[column_name].apply(lambda x: len(tokenizer.tokenize(x))).max()
    
#     # Initialize the BertEmbedder with the computed max_length
#     embedder = BertEmbedder(max_length=max_length)
    
#     # Getting embeddings for the specified column of the DataFrame
#     embeddings = embedder.get_embeddings(df[column_name].tolist())
    
#     # Normalize the embeddings using L2 norm
#     normalized_embeddings = normalize(embeddings, norm='l2')
    
#     return normalized_embeddings
