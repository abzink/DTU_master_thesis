# dimensionality_reduction.py

# Import necessary libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
from plotting_utils import save_figure
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib as mpl
import matplotlib.font_manager as font_manager

# Font settings
mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.formatter.use_mathtext'] = True

# Determine the directory of the current script for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))

def pca_reduction(features_matrix, feature_type, save_path=None):
    """
    Reduces the dimensionality of features using PCA.
    Visualizes and saves the explained variance for different numbers of components.
    """
    pca = PCA(n_components=min(features_matrix.shape))
    pca.fit(features_matrix)
    
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    threshold = 0.95
    optimal_components = np.argmax(explained_variance_ratio >= threshold) + 1
    pca_final = PCA(n_components=optimal_components)
    X_pca = pca_final.fit_transform(features_matrix)

    # Choose a color from the viridis palette
    viridis_color = sns.color_palette("viridis", 10)  # 10 to get a range of colors from the palette

    # Visualization using Seaborn with color from viridis palette
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=np.arange(1, len(explained_variance_ratio) + 1), y=explained_variance_ratio, color=viridis_color[2])  # Using the third color in the palette
    line = plt.axvline(optimal_components, color=viridis_color[7], linestyle='--')

    # Creating a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=viridis_color[7], linestyle='--', label=f'95% variance at {optimal_components} components')]
    plt.legend(handles=legend_elements, loc='best')

    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'PCA on {feature_type} Embeddings - Explained Variance')
    plt.grid(True)

    plt.tight_layout()

    # Save figure
    figure_path = f"{feature_type}_PCA_explained_variance"
    save_figure(plt.gcf(), figure_path)

    # Show the plot
    plt.show()

    return X_pca

def tsne_reduction(features_matrix, feature_type, perplexity=30, metric=None):
    """
    Reduces the dimensionality of features using t-SNE.
    Visualizes and saves the 2D representation of features.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, metric=metric, random_state=42)
    X_tsne = tsne.fit_transform(features_matrix)

    # Visualization
    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
    ax.set_title(f'2D t-SNE on {feature_type} features (Perplexity: {perplexity})')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    plt.show()

    # Save figure
    figure_path = f"{feature_type}_TSNE_perplexity_{perplexity}"
    save_figure(fig, figure_path)
    
    # Save features
    # filename = os.path.join(script_dir, f"data/dim_red/{feature_type}_TSNE_perplexity_{perplexity}.npy")
    # np.save(filename, X_tsne)
    
    return X_tsne

def umap_reduction(features_matrix, feature_type, n_neighbors=15, min_dist=0.1, metric=None):
    """
    Reduces the dimensionality of features using UMAP.
    Visualizes and saves the 2D representation of features.
    """
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    X_umap = umap_model.fit_transform(features_matrix)
    
    # Visualization
    fig, ax = plt.subplots()
    ax.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.5)
    ax.set_title(f'UMAP on {feature_type} features (n_neighbors={n_neighbors}, min_dist={min_dist})')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    plt.show()

    # Save figure
    figure_path = f"{feature_type}_UMAP_neighbors_{n_neighbors}_minDist_{min_dist}"
    save_figure(fig, figure_path)
    
    # Save features
    # filename = os.path.join(script_dir, f"data/dim_red/{feature_type}_UMAP_neighbors_{n_neighbors}_minDist_{min_dist}.npy")
    # np.save(filename, X_umap)
    
    return X_umap

# Create necessary directories if they don't exist
data_path = os.path.join(script_dir, 'data/dim_red')
if not os.path.exists(data_path):
    os.makedirs(data_path)
