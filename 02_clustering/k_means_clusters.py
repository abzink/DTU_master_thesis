import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from plotting_utils import save_figure
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.cm as cm

import matplotlib as mpl
import matplotlib.font_manager as font_manager

# Font settings
mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.formatter.use_mathtext'] = True

def find_optimal_clusters(data, max_k, reduction_method_name=None, feature_type=None):
    """Find the optimal number of clusters using the elbow method and silhouette score."""
    wcss = []
    silhouette_scores = []
    K = range(2, max_k + 1)

    # Compute Within-Cluster-Sum-of-Squares (WCSS) and Silhouette scores for various k values
    for k in K:
        kmean = KMeans(n_clusters=k, n_init=10, random_state=42).fit(data)
        wcss.append(kmean.inertia_)
        silhouette_avg = silhouette_score(data, kmean.labels_)
        silhouette_scores.append(silhouette_avg)

    # Plot for the Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(K, wcss, 'o-', color=cm.viridis(0.3))  # Color from Viridis palette
    plt.title(f'Elbow Method for Optimal Clusters in {feature_type} via {reduction_method_name}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.tight_layout()
    filename = f"{feature_type}_via_{reduction_method_name}_elbow_{max_k}_clusters"
    # save_figure(plt.gcf(), filename)
    plt.show()

    # Plot for the Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(K, silhouette_scores, 'o-', color=cm.viridis(0.6))  # Another color from Viridis palette
    plt.title(f'Silhouette Score for Optimal Clusters in {feature_type} via {reduction_method_name}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    filename = f"{feature_type}_via_{reduction_method_name}_silhouette_{max_k}_clusters"
    # save_figure(plt.gcf(), filename)
    plt.show()


def cluster_and_visualize(data, main_df, reduction_method_name=None, feature_type=None):
    """Cluster the data and visualize the results."""
    find_optimal_clusters(data, max_k=10, reduction_method_name=reduction_method_name, feature_type=feature_type)
    optimal_k = int(input("Enter the optimal k from the plot: "))
    kmeans = KMeans(n_clusters=optimal_k, n_init=5, random_state=42).fit(data)
    
    # Assign cluster labels to the main dataframe
    cluster_labels = kmeans.labels_
    column_suffix = f'{feature_type}_{reduction_method_name}' if feature_type and reduction_method_name else feature_type or reduction_method_name
    main_df[f'{column_suffix}_Cluster_Labels'] = cluster_labels
    
    # Compute and assign distance to nearest centroid to the main dataframe
    pairwise_dist = pairwise_distances(data, kmeans.cluster_centers_)
    closest_centroid_distance = np.min(pairwise_dist, axis=1)
    main_df[f'{column_suffix}_Distance_to_Centroid'] = closest_centroid_distance

    # Define colors for the unique categories using Seaborn's vibrant palette
    unique_cats = ['A', 'B', 'C', 'D', 'E']
    cat_colors = sns.color_palette("deep", n_colors=len(unique_cats))
    cat_color_dict = {cat: cat_colors[i] for i, cat in enumerate(unique_cats)}

    # Define pastel colors for clusters with a low alpha value for transparency
    cluster_colors = sns.color_palette("pastel", n_colors=optimal_k)
    alpha_value = 0.2  # Setting a low alpha value for high transparency

    # Adjusting the alpha value for each color in the palette
    cluster_colors_with_alpha = [(r, g, b, alpha_value) for r, g, b in cluster_colors]
    cluster_color_dict = {label: cluster_colors_with_alpha[label] for label in range(optimal_k)}

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(data[:, :2], columns=['Dim1', 'Dim2'])
    plot_df['Cluster'] = cluster_labels
    plot_df['Category'] = main_df['category']

    if reduction_method_name:
        plt.figure(figsize=(8, 6))
        
        # Plot clusters using the pastel color palette with low alpha
        sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', palette=cluster_colors_with_alpha, data=plot_df, s=50, legend=False)

        # Overlay category scatter plot using the pastel palette with low alpha
        for cat in unique_cats:
            cat_data = plot_df[plot_df['Category'] == cat]
            sns.scatterplot(x='Dim1', y='Dim2', color=cat_color_dict[cat], data=cat_data, s=50, label=cat, edgecolor='black')

        plt.title(f'{feature_type} Embeddings Clusters via {reduction_method_name}', fontsize= 18)
        plt.xlabel(f'{reduction_method_name} Component 1', fontsize = 16)
        plt.ylabel(f'{reduction_method_name} Component 2', fontsize = 16)
        # Increase the size of the x-tick and y-tick labels
        plt.tick_params(axis='x', labelsize=12)  # Increase x-tick label size
        plt.tick_params(axis='y', labelsize=12)  # Increase y-tick label size
        plt.legend(title="Categories", fontsize = 14, title_fontsize = 14)

        plt.tight_layout()

        # Save the figure before displaying it
        save_figure(plt.gcf(), f"{feature_type}_{reduction_method_name}_clusters")
        plt.show()

    return main_df


def cluster_from_file(reduction_file_path):
    """Cluster and visualize data from a given file."""
    # Extract method names from the filename
    filename_parts = reduction_file_path.split('/')[-1].replace('.npy', '').split('_')
    reduction_method = filename_parts[0]
    feature_type = filename_parts[1]

    # Load the main dataframe and reduced data
    main_df = pd.read_csv('data/processed_data_all.csv')
    X_reduced = np.load(reduction_file_path)

    # Cluster and visualize the data
    cluster_and_visualize(X_reduced, main_df, reduction_method, feature_type)

if __name__ == "__main__":
    file_path = input("Enter the path to the file containing the dimensionally reduced data: ")
    cluster_from_file(file_path)
