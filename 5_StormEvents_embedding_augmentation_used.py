# # Data Augmentation with Embedding Model

# Text data encoded in embeddings can be used as additional features in tabular dataset as data augmentation.
# 
# Sentence transformers are employed to produce embeddings, and then a dimensionality reduction is applied to obtain a desired number of features.

# # Prepare Workspace

# %%
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ["HF_TOKEN"] = ""

# # Load Model and dataset

# Load a transformer model suitable for sentence embedding, on the GPU
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device='cuda')

df = pd.read_csv('./work/StormEvents_narratives_filled_fin.csv', low_memory=False)
print(f"Loaded df: {df.shape}")


# # Build Embeddings features from episode

# Extract the text data from the last column
text_data_ep = df['EPISODE_NARRATIVE'].tolist()

# Generate embeddings for the text data (larger batch size to keep the GPU busy)
embeddings1 = model.encode(text_data_ep, batch_size=128, show_progress_bar=True)

# Convert embeddings to numpy arrays if needed
embeddings1 = np.array(embeddings1)

# Print the shape of the embeddings to verify the dimension
print(embeddings1.shape)

# # Apply Dimensionality Reduction to the Embeddings

# Create a TruncatedSVD instance
n_components = 10  # Desired number of dimensions
svd = TruncatedSVD(n_components=n_components, random_state=0)

# Fit and transform the embeddings
reduced_embeddings1 = svd.fit_transform(embeddings1)

# Check the shape to ensure reduction worked as expected
print(reduced_embeddings1.shape)

# # Attach Reduced Embeddings to the Dataset

# make a copy
df1 = df.copy()

# Generate feature names for the reduced dimensions
reduced_feature_names1 = [f"ep_embedding_{i+1}" for i in range(n_components)]

# Add columns to the DataFrame
for i, feature_name in enumerate(reduced_feature_names1):
    df1.loc[:, feature_name] = reduced_embeddings1[:, i]



# Verify the DataFrame to ensure columns are added
df1.head()

df1.to_csv('./work/StormEvents_episode_embedding.csv', index=False)

# # Build Embeddings features from event

# Extract the text data from the last column
text_data_ev = df1['EVENT_NARRATIVE'].tolist()

# Generate embeddings for the text data (larger batch size to keep the GPU busy)
embeddings2 = model.encode(text_data_ev, batch_size=128, show_progress_bar=True)

# Convert embeddings to numpy arrays if needed
embeddings2 = np.array(embeddings2)

# Print the shape of the embeddings to verify the dimension
print(embeddings2.shape)

# # Apply Dimensionality Reduction to the Embeddings

# Create a TruncatedSVD instance
n_components = 10  # Desired number of dimensions
svd = TruncatedSVD(n_components=n_components, random_state=0)

# Fit and transform the embeddings
reduced_embeddings2 = svd.fit_transform(embeddings2)

# Check the shape to ensure reduction worked as expected
print(reduced_embeddings2.shape)

# # Attach Reduced Embeddings to the Dataset

# make a copy
df2 = df1.copy()

# Generate feature names for the reduced dimensions
reduced_feature_names2 = [f"ev_embedding_{i+1}" for i in range(n_components)]

# Add columns to the DataFrame
for i, feature_name in enumerate(reduced_feature_names2):
    df2.loc[:, feature_name] = reduced_embeddings2[:, i]



# Verify the DataFrame to ensure columns are added
df2.head()

df2.to_csv('./work/StormEvents_fe_embedding_fin.csv', index=False)


