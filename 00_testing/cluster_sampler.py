import pandas as pd

class ClusterSampler:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the ClusterSampler object with a DataFrame.
        :param df: The DataFrame containing the data points, cluster labels, and distances to cluster centers.
        """
        self.df = df
        
    def sample_clusters(self, cluster_column: str, n_samples: int, category_filter='-1'):
        """
        Perform Random Sampling from each cluster.
        :param cluster_column: The column containing cluster labels.
        :param n_samples: The number of samples to be drawn from each cluster.
        :param category_filter: Exclude rows where category is equal to category_filter.
        :return: A DataFrame containing sampled data.
        """
        filtered_df = self.df[self.df['category'] != category_filter]  # Filter out specified category
        sampled_dfs = []
        
        # Iterate over each unique cluster label and perform random sampling
        for cluster_label in filtered_df[cluster_column].unique():
            cluster_df = filtered_df[filtered_df[cluster_column] == cluster_label]
            sampled_df = cluster_df.sample(n=min(n_samples, len(cluster_df)), random_state=42)
            sampled_dfs.append(sampled_df)
        
        final_sampled_df = pd.concat(sampled_dfs, axis=0)
        return final_sampled_df
    
    def sample_based_on_distance(self, cluster_column: str, n_samples: int, sampling_type: str = 'centroid', category_filter='-1'):
        """
        Sample points based on their distance to the cluster center.
        :param cluster_column: The column containing cluster labels.
        :param n_samples: The number of samples to be drawn from each cluster.
        :param sampling_type: The type of sampling ('centroid' for centroid-based and 'edge' for edge cases).
        :param category_filter: Exclude rows where category is equal to category_filter.
        :return: A DataFrame containing sampled data points.
        """
        filtered_df = self.df[self.df['category'] != category_filter]  # Filter out specified category
        sampled_dfs = []
        
        # Iterate over each unique cluster label and perform sampling based on distance to the cluster center
        for cluster_label in filtered_df[cluster_column].unique():
            cluster_df = filtered_df[filtered_df[cluster_column] == cluster_label]
            
            # Sample points closest to the centroid for centroid-based sampling
            if sampling_type == 'centroid':
                sampled_df = cluster_df.nsmallest(n=min(n_samples, len(cluster_df)), columns='distance_to_center')
                
            # Sample points farthest from the centroid for edge case sampling
            elif sampling_type == 'edge':
                sampled_df = cluster_df.nlargest(n=min(n_samples, len(cluster_df)), columns='distance_to_center')
                
            else:
                raise ValueError(f"Invalid sampling_type: {sampling_type}. Choose 'centroid' or 'edge'.")
            
            sampled_dfs.append(sampled_df)
        
        final_sampled_df = pd.concat(sampled_dfs, axis=0)
        return final_sampled_df


# # Initialize the object with your DataFrame
# sampler = ClusterSampler(df)

# # To perform random sampling, call the method sample_clusters on the sampler object
# random_samples_df = sampler.sample_clusters(cluster_column='cluster_label', n_samples=10)

# # To sample points closest to the centroid, call the method sample_based_on_distance with sampling_type='centroid'
# centroid_based_samples = sampler.sample_based_on_distance(cluster_column='cluster_label', n_samples=10, sampling_type='centroid')

# # To sample edge cases, call the method sample_based_on_distance with sampling_type='edge'
# edge_based_samples = sampler.sample_based_on_distance(cluster_column='cluster_label', n_samples=10, sampling_type='edge')
