import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import matplotlib as mpl
import matplotlib.font_manager as font_manager

# Font settings
mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.formatter.use_mathtext'] = True

sys.path.append('../')
from plotting_utils import save_figure

class GraphResultsProcessor:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.data_frames = []
    
    def load_data(self):
        data_list = []
        for file_path in self.file_paths:
            with open(file_path, 'r') as file:
                data = json.load(file)
                training_results = data['training_results']
                labeled_percentage = self.extract_labeled_percentage(file_path)
                
                # Flatten the 'training_results' into the top-level dictionary
                flat_data = {**training_results, 'labeled_percentage': labeled_percentage}
                flat_data['model_name'] = flat_data.pop('model_name')  # Ensure 'model_name' is at the top level

                # Flatten 'bootstrap_results' into the top-level dictionary
                for key, value in training_results['bootstrap_results'].items():
                    flat_data[key] = value
                
                data_list.append(flat_data)
        
        self.data_frames = pd.DataFrame(data_list)
        print(f"Loaded data into DataFrame with shape: {self.data_frames.shape}")



    def plot_metric_comparison_across_models(self, metrics, save=True):
        # Mapping of metric abbreviations to full names
        metric_name_mapping = {
            'accuracy': 'Accuracy',
            'balanced_accuracy': 'Balanced Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1 Score',
            'kappa_score': 'Cohen Kappa Score',
            'mcc_score': 'MCC Score'
        }

        if isinstance(metrics, str):
            metrics = [metrics]

        for metric in metrics:
            if metric not in self.data_frames.columns:
                print(f"Metric '{metric}' not found in DataFrame.")
                continue

            full_metric_name = metric_name_mapping.get(metric, metric.capitalize())

            plt.figure(figsize=(10, 7))
            for model in self.data_frames['model_name'].unique():
                model_data = self.data_frames[self.data_frames['model_name'] == model]
                sns.lineplot(
                    data=model_data,
                    x='labeled_percentage',
                    y=metric,
                    label=model,
                    marker=self.markers.get(model, 'o'),
                    linestyle=self.line_styles.get(model, '-'),
                    color=self.colors.get(model, 'blue')
                )
            plt.title(f'Semi-Supervised Graph Based Methods Performance: {full_metric_name}')
            plt.xlabel('Labeled Fraction (%)')
            plt.ylabel(full_metric_name)

            plt.xticks([20, 40, 60, 80, 100])
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()

            if save:
                filename = f'metric_comparison_{model}_{metric}'
                save_figure(plt.gcf(), filename)
                print(f"Saved plot as '{filename}'")

            plt.show()

    def plot_confusion_matrix(self, save=True):
        for index, row in self.data_frames.iterrows():
            model_name = row['model_name']
            labeled_percentage = row['labeled_percentage']
            if 'confusion_matrix' not in row or row['confusion_matrix'] is None:
                print(f"Confusion matrix not available for {model_name} with {labeled_percentage}% labeled data.")
                continue

            cm = np.array(row['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xticklabels(['A', 'B', 'C', 'D', 'E'])
            ax.set_yticklabels(['A', 'B', 'C', 'D', 'E'])
            ax.set_title(f'{model_name} ({labeled_percentage}% Labeled Data)')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            plt.show()

            if save:
                filename = f'confusion_matrix_{model_name}_{labeled_percentage}'
                save_figure(fig, filename)

    def extract_labeled_percentage(self, file_path):
            base_name = os.path.basename(file_path)
            percentage = base_name.split('_')[-1].split('.')[0]
            return int(percentage)
        
    def plot_bootstrap_metrics(self, metric, save=False):
        if not isinstance(metric, str):
            print("Please provide a single metric as a string.")
            return

        unique_models = self.data_frames['model_name'].unique()
        unique_fractions = sorted(self.data_frames['labeled_percentage'].unique())

        # Prepare the plot
        plt.figure(figsize=(12, 8))
        n_models = len(unique_models)
        n_fractions = len(unique_fractions)
        width = 0.35  # width of the bars
        bar_width = width / n_models  # width of one bar

        # Calculate the total width for all bars together
        total_width = n_models * bar_width

        for i, fraction in enumerate(unique_fractions):
            # Calculate the base position for the group of bars
            for j, model in enumerate(unique_models):
                # Filter data for each model and fraction
                model_data = self.data_frames[(self.data_frames['model_name'] == model) &
                                              (self.data_frames['labeled_percentage'] == fraction)]

                mean_metric = f'mean_{metric}'
                std_metric = f'std_{metric}'

                if mean_metric in model_data and std_metric in model_data:
                    # Calculate position for each bar
                    bar_position = i - (total_width - bar_width) / 2 + j * bar_width
                    plt.bar(bar_position, model_data[mean_metric].iloc[0], width=bar_width,
                            yerr=model_data[std_metric].iloc[0], capsize=5,
                            color=self.colors[model], label=model if i == 0 else "")

        plt.xlabel('Labeled Fraction (%)')
        plt.ylabel(f'Mean {metric.capitalize()}')
        plt.title(f'Bootstrap Test Across Labeled Fractions for {metric.capitalize()}')

        # Setting the position of the xticks
        plt.xticks(range(n_fractions), unique_fractions)

        # Removing grid lines
        plt.grid(False)

        # Adding the legend below the plot
        if n_models > 1:
            plt.legend(title='Model', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_models)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rectangle in which to place the plot

        if save:
            filename = f'bootstrap_metrics_{metric}'
            # Implement save_figure function or replace with plt.savefig
            # save_figure(plt.gcf(), filename)
            print(f"Saved plot as '{filename}'")

        plt.show()

    def create_hyperparams_table(self):
        # Initialize a dictionary to store the hyperparameters
        hyperparams_dict = {}

        # Loop over each unique model in the DataFrame
        for model in self.data_frames['model_name'].unique():
            # Filter the DataFrame for the current model
            model_data = self.data_frames[self.data_frames['model_name'] == model]
            # Sort the data by labeled percentage
            model_data = model_data.sort_values(by='labeled_percentage')
            
            # Extract the hyperparameters and labeled percentages
            hyperparams = model_data['best_params'].tolist()
            labeled_percentages = model_data['labeled_percentage'].tolist()
            
            # Create a DataFrame from the hyperparameters
            hyperparams_df = pd.DataFrame(hyperparams, index=labeled_percentages)
            
            # Store the DataFrame in the dictionary with the model as the key
            hyperparams_dict[model] = hyperparams_df
        
        # Return the dictionary of DataFrames
        return hyperparams_dict

    



    # Define custom settings for each model
    palette = sns.color_palette()
    colors = {
        'LabelPropagation': palette[7],  # Color for LabelPropagation
        'LabelSpreading': palette[9],  # Color for LabelSpreading
        # Add more models and colors as needed
    }

    markers = {
        'LabelPropagation': 'o',  # Marker for LabelPropagation
        'LabelSpreading': 'D',  # Marker for LabelSpreading
        # Add more models and markers as needed
    }

    line_styles = {
        'LabelPropagation': ':',  # Line style for LabelPropagation
        'LabelSpreading': '--',  # Line style for LabelSpreading
        # Add more models and line styles as needed
    }
    

        # sns palette -1, -3 for models
        # palette = sns.color_palette()
        # colors = {
        #     'RandomForest': palette[-4],
        #     'XGBoost': palette[-5],
        #     'LightGBM': palette[-2],
        #     #'Bagging': palette[-1],
        #     # Add more models and colors as needed
        # }
