import sys
sys.path.append('../../') 
import os
import json
import pandas as pd
from plotting_utils import save_figure
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as font_manager

# Font settings
mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.formatter.use_mathtext'] = True


class SelfTrainingResultsProcessor:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def map_labels_to_alphabetical(self, label):
        label_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        return label_mapping.get(label, label)  # Returns original label if not in mapping

    def read_and_integrate_data(self):
        # Initialize DataFrames for each section
        data_frames = {
            'cv_metrics': pd.DataFrame(),
            'training': pd.DataFrame(),
            'test': pd.DataFrame(),
            'feature_importances': pd.DataFrame(),
            'bootstrap_test': pd.DataFrame(),
            'pseudo_labels_info': pd.DataFrame(),
            'labeled_samples_info': pd.DataFrame(),
            'training_time': pd.DataFrame() 
        }

        # Read and integrate data from each file
        for file_path in self.file_paths:
            with open(file_path, 'r') as file:
                data = json.load(file)

                # Process cv_metrics, training, test, and bootstrap_test sections
                for section in ['cv_metrics', 'training', 'test', 'bootstrap_test', 'pseudo_labels_info', 'labeled_samples_info', 'training_time']:  
                    if section in data:
                        df = pd.DataFrame(data[section])
                        data_frames[section] = pd.concat([data_frames[section], df], ignore_index=True)

                # Special processing for feature_importances
                if 'feature_importances' in data:
                    fi_data = data['feature_importances']
                    for model, model_data in fi_data.items():
                        for entry in model_data:
                            labeled_percentage = entry['labeled_percentage']
                            importances = entry['importances']
                            if importances is not None:
                                fi_df = pd.DataFrame.from_dict(importances, orient='index', columns=[f'{model}_{labeled_percentage}'])
                                data_frames['feature_importances'] = pd.concat([data_frames['feature_importances'], fi_df], axis=1)
                            else:
                                pass

        # Handling missing values in feature_importances after concatenation
        data_frames['feature_importances'] = data_frames['feature_importances'].fillna(0)

        return data_frames

    def create_and_save_visualizations(self, data_frames):
        # Create the directory if it doesn't exist
        output_dir = 'figures/Self_Training'
        os.makedirs(output_dir, exist_ok=True)

        # Visualization for cv_metrics
        self.plot_cv_metrics(data_frames['cv_metrics'], os.path.join(output_dir, 'cv_metrics.png'))

        # Visualization for training data
        training_figs = self.plot_training_data(data_frames['training'], os.path.join(output_dir, 'training_data.png'))
        for i, fig in enumerate(training_figs):
            save_figure(fig, os.path.join(output_dir, f'training_metric_{i}'))

        # Visualization for test data
        test_figs = self.plot_test_data(data_frames['test'], os.path.join(output_dir, 'test_data.png'))

        # Visualization for feature_importances
        self.plot_feature_importances(data_frames['feature_importances'], os.path.join(output_dir, 'feature_importances.png'))

       
    """Cross Validation Metrics"""

    """Train Data Plots"""

    def plot_training_data(self, df, base_filename = 'sf_performance'):
        # Mapping of original metric names to more descriptive titles
        metric_titles = {
            'train_accuracy': 'Accuracy',
            'train_balanced_accuracy': 'Balanced Accuracy',
            'train_top_k_accuracy': 'Top 3-Accuracy',
            'train_precision': 'Precision',
            'train_recall': 'Recall',
            'train_f1': 'F1 Score',
            'train_mcc': 'MCC Score',
            'train_error': 'Error',
            'train_log_loss': 'Log Loss',
            'kappa_score_train': 'Cohen Kappa Score'
        }

        figs = []
                
        # Define dictionaries for markers, line styles, and colors
        markers = {
            'RandomForest': 'o',
            'XGBoost': 's',
            'LightGBM': 'D',
            #'Bagging': 'X',
            # Add more models and markers as needed
        }
        
        line_styles = {
            'RandomForest': '--',
            'XGBoost': ':',
            'LightGBM': '-.',
            #'Bagging': '--',
            # Add more models and line styles as needed
        }
        
        palette = sns.color_palette()
        colors = {
            'RandomForest': palette[-4],
            'XGBoost': palette[-5],
            'LightGBM': palette[-2],
            #'Bagging': palette[-1],
            # Add more models and colors as needed
        }


        figs = []
        models = df['model'].unique()

        for metric, title in metric_titles.items():
            fig, ax = plt.subplots()

            for model in models:
                model_data = df[df['model'] == model]
                sns.lineplot(data=model_data, x='labeled_percentage', y=metric, label=model,
                             ax=ax, marker=markers.get(model, 'o'),
                             linestyle=line_styles.get(model, '-'), color=colors.get(model, 'blue'))


            ax.set_title(f'Self-Training Performance')
            ax.set_xlabel('Labeled Fraction (%)')
            ax.set_ylabel(title)

            # Set x-axis ticks
            ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1])
            ax.set_xticklabels(['20', '40', '60', '80', '100'])

            ax.grid(True)

            # Remove legend title
            legend = ax.get_legend()
            if legend:
                legend.set_title(None)

            figs.append(fig)

                        # Save the figure if a base filename is provided
            if base_filename:
                full_filename = f"{base_filename}_{metric}"
                save_figure(fig, full_filename)
            plt.close(fig)  # Close the figure after saving


        return figs
    
    """Test Data Plots"""
    
    def plot_test_data(self, df, base_filename = 'sf_performance'):
        # Mapping of original metric names to more descriptive titles
        metric_titles = {
            'test_accuracy': 'Accuracy',
            'test_balanced_accuracy': 'Balanced Accuracy',
            'test_top_k_accuracy': 'Top 3-Accuracy',
            'test_precision': 'Precision',
            'test_recall': 'Recall',
            'test_f1': 'F1 Score',
            'test_mcc': 'MCC Score',
            'test_error': 'Error',
            'test_log_loss': 'Log Loss',
            'kappa_score_test': 'Cohen Kappa Score'
        }

        figs = []
                
        # Define dictionaries for markers, line styles, and colors
        markers = {
            'RandomForest': 'o',
            'XGBoost': 's',
            'LightGBM': 'D',
            #'Bagging': 'X',
            # Add more models and markers as needed
        }
        
        line_styles = {
            'RandomForest': '-',
            'XGBoost': ':',
            'LightGBM': '-.',
            #'Bagging': '--',
            # Add more models and line styles as needed
        }
        
        palette = sns.color_palette()
        colors = {
            'RandomForest': palette[-4],
            'XGBoost': palette[-5],
            'LightGBM': palette[-2],
            #'Bagging': palette[-1],
            # Add more models and colors as needed
        }


        figs = []
        models = df['model'].unique()

        for metric, title in metric_titles.items():
            fig, ax = plt.subplots()

            for model in models:
                model_data = df[df['model'] == model]
                sns.lineplot(data=model_data, x='labeled_percentage', y=metric, label=model,
                             ax=ax, marker=markers.get(model, 'o'),
                             linestyle=line_styles.get(model, '-'), color=colors.get(model, 'blue'))


            ax.set_title(f'Self-Training Performance')
            ax.set_xlabel('Labeled Fraction (%)')
            ax.set_ylabel(title)

            # Set x-axis ticks
            ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1])
            ax.set_xticklabels(['20', '40', '60', '80', '100'])

            ax.grid(True)

            # Remove legend title
            legend = ax.get_legend()
            if legend:
                legend.set_title(None)

            figs.append(fig)

            if base_filename:
                full_filename = f"{base_filename}_{metric}"
                save_figure(fig, full_filename)
            plt.close(fig)  # Close the figure after saving


        return figs


    def plot_accuracy_bar_chart(self, df, base_filename='accuracy_barplots'):
        # Filtering DataFrame for accuracy metrics
        accuracy_df = df[['model', 'labeled_percentage', 'test_accuracy']]  # Replace 'train_accuracy' with 'test_accuracy' if needed

        # Reshaping the DataFrame for plotting
        accuracy_df = accuracy_df.melt(id_vars=['model', 'labeled_percentage'], var_name='metric', value_name='accuracy')

        # Define the color palette for the models
        palette = sns.color_palette()
        colors = {
            'RandomForest': palette[-4],
            'XGBoost': palette[-5],
            'LightGBM': palette[-2],
            # 'Bagging': palette[-1],
            # Add more models and colors as needed
        }

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        barplot = sns.barplot(data=accuracy_df, x='labeled_percentage', y='accuracy', hue='model', palette=colors, ax=ax)
        
        ax.set_xticklabels([20, 40, 60, 80, 100])

        ax.set_title('Accuracies Across Models for Different Labeled Fractions')
        ax.set_xlabel('Labeled Fraction (%)')
        ax.set_ylabel('Accuracy')

        # Find the overall highest accuracy and its position
        overall_max_accuracy = accuracy_df['accuracy'].max()
        overall_max_acc_row = accuracy_df[accuracy_df['accuracy'] == overall_max_accuracy].iloc[0]

        # Create a list of unique labeled percentages
        labeled_percents = sorted(accuracy_df['labeled_percentage'].unique())

        # Calculate the x_position for the overall highest accuracy
        x_base = labeled_percents.index(overall_max_acc_row['labeled_percentage'])
        num_models = len(accuracy_df['model'].unique())
        model_index = list(accuracy_df['model'].unique()).index(overall_max_acc_row['model'])
        # Adjusting the x_position to center it on the correct bar
        x_position = x_base + (model_index + 0.5 - num_models / 2) * (0.8 / num_models)

        # Place a single red star marker above the highest accuracy of all
        ax.scatter(x=x_position, y=overall_max_accuracy + 0.02, color='red', marker='*', s=100)

        # Place the legend below the plot
        ax.legend(title='Base Learners', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=num_models)

        # Adjust layout to include the legend
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure if a base filename is provided
        if base_filename:
            full_filename = f"sf_{base_filename}"
            save_figure(fig, full_filename)
        plt.close(fig)  # Close the figure after saving

        plt.show()


    """Confusion Matrices"""
    
    def plot_confusion_matrices(self, df, suffix, base_filename='sf'):
        # Filter out unique combinations of model and labeled_percentage
        unique_combinations = df[['model', 'labeled_percentage']].drop_duplicates()

        for index, row in unique_combinations.iterrows():
            model = row['model']
            labeled_percentage = row['labeled_percentage']

            # Extract confusion matrix data
            cm_data = df[(df['model'] == model) & (df['labeled_percentage'] == labeled_percentage)]['confusion_matrix'].iloc[0]
            cm_array = np.array(cm_data)

            # Define figure size (width, height) in inches
            figure_size = (6, 4)  # You can adjust these values as needed

            # Plot the confusion matrix with the specified figure size
            fig, ax = plt.subplots(figsize=figure_size)
            sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', ax=ax)

            # Check if labeled_percentage is a whole number and format accordingly
            formatted_percentage = int(labeled_percentage * 100) if labeled_percentage * 100 == int(labeled_percentage * 100) else labeled_percentage * 100

            # Set title with formatted percentage
            ax.set_title(f'{model} ({formatted_percentage}% Labeled Data)')

            # Set title and labels
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')

            # Define and set the tick labels
            alphabet_labels = ['A', 'B', 'C', 'D', 'E']  # Adjust as per your actual labels
            ax.set_xticklabels(alphabet_labels)
            ax.set_yticklabels(alphabet_labels)

            # Save the figure with the specified suffix
            if base_filename:
                full_filename = f"{base_filename}_{suffix}_confusion_matrix_{model}_{labeled_percentage}"
                save_figure(fig, full_filename)
            plt.close(fig)  # Close the figure after saving



    # adjust to see the differences between test and train for each model between each labeled percentage
    def generate_log_loss_and_error_tables(self, training_df, testing_df):
        models = training_df['model'].unique()  # Get unique models
        tables = {}

        for model in models:
            # Filter data for the current model
            train_df_model = training_df[training_df['model'] == model]
            test_df_model = testing_df[testing_df['model'] == model]

            # Generate tables for log loss and error for each model
            log_loss_table = self.generate_metric_table(train_df_model, test_df_model, 'train_log_loss', 'test_log_loss', model)
            error_table = self.generate_metric_table(train_df_model, test_df_model, 'train_error', 'test_error', model)

            tables[model] = {'log_loss': log_loss_table, 'error': error_table}

        return tables

    def generate_metric_table(self, training_df, testing_df, train_metric, test_metric, model_name):
        # Create a DataFrame to store the metrics
        metrics_df = pd.DataFrame()

        # Extract the metrics for each labeled fraction
        for labeled_percentage in sorted(training_df['labeled_percentage'].unique()):
            train_metric_value = training_df[training_df['labeled_percentage'] == labeled_percentage][train_metric].mean()
            test_metric_value = testing_df[testing_df['labeled_percentage'] == labeled_percentage][test_metric].mean()
            
            metrics_df = metrics_df.append({
                'labeled_percentage': labeled_percentage,
                f'{train_metric} (train)': train_metric_value,
                f'{test_metric} (test)': test_metric_value
            }, ignore_index=True)

        # Set labeled_percentage as the index
        metrics_df.set_index('labeled_percentage', inplace=True)

        return metrics_df


    """Pseudo Labels Info"""

    def visualize_pseudo_label_distribution(self, data, balancing_variant, filename='pseudo_label_distribution'):
        palette = sns.color_palette()
        colors = {
            'RandomForest': palette[-4],
            'XGBoost': palette[-5],
            'LightGBM': palette[-2],
        }

        # Convert the data into a pandas DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Prepare a custom palette that matches models in your data
        custom_palette = [colors[model] for model in data['model'].unique()]

        # Plot
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(x="labeled_percentage", y="pseudo_label_count", hue="model", data=data, palette=custom_palette)

        plt.title(f'Pseudo-Labels Count ({balancing_variant})')
        plt.xlabel('Labeled Fraction (%)')
        plt.ylabel('Count')
        plt.legend(title='')

        # Set custom x-axis tick labels
        labeled_percentages = [f"{int(labeled_percentage * 100)}" for labeled_percentage in data['labeled_percentage'].unique()]
        ax.set_xticklabels(labeled_percentages)

        # Save the figure before showing the plot
        fig = ax.get_figure()
        save_figure(fig, f'{filename}_{balancing_variant}')  # Include balancing variant in filename

        plt.close(fig)

        plt.show()



    def visualize_pseudo_label_models(self, data, balancing_variant, filename='pseudo_label_models'):
        # Convert the data into a pandas DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Retrieve the default Seaborn palette and extract the last 5 colors
        default_palette = sns.color_palette()
        last_five_colors = default_palette[-5:]

        # Plot using the last five colors of the default palette
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(x="model", y="pseudo_label_count", hue="labeled_percentage", data=data, palette=last_five_colors)

        plt.title(f'Pseudo-Labels Count ({balancing_variant})', fontsize=18)
        plt.xlabel('')
        plt.ylabel('Count', fontsize=16) 
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tick_params(axis='x', labelsize=14)  # Increase x-tick label size
        plt.tick_params(axis='y', labelsize=14)  # Increase y-tick label size

        # Create a new legend for each subplot
        handles, _ = ax.get_legend_handles_labels()
        new_labels = ['20', '40', '60', '80', '100']
        plt.legend(handles, new_labels, title='Labeled Fraction (%)', fontsize=14, title_fontsize=14)

        # Save the figure before showing the plot
        fig = ax.get_figure()
        save_figure(fig, f'{filename}_{balancing_variant}')  # Include balancing variant in filename

        plt.close(fig)  # Close the figure after saving

        plt.show()

    # def visualize_pseudo_label_models(data_frames, smote_types, filename='pseudo_label_models'):
    #     # Set the desired bar width
    #     bar_width = 0.15

    #     # Create a figure with subplots - one for each SMOTE type
    #     fig, axes = plt.subplots(1, len(smote_types), figsize=(20, 6), sharey=True, dpi=500)

    #     # Retrieve the default Seaborn palette and extract the last 5 colors
    #     default_palette = sns.color_palette()
    #     last_five_colors = default_palette[-5:]

    #     for idx, (smote_type, data) in enumerate(zip(smote_types, data_frames.values())):
    #         ax = axes[idx]

    #         # Convert the data into a pandas DataFrame if it's not already
    #         if not isinstance(data, pd.DataFrame):
    #             data = pd.DataFrame(data)

    #         sns.barplot(x="model", y="pseudo_label_count", hue="labeled_percentage", data=data, palette=last_five_colors, ax=ax)

    #         ax.set_title(f'Pseudo-Labels Count ({smote_type})')
    #         ax.set_xlabel('')
    #         ax.set_ylabel('Count' if idx == 0 else '')

    #         # Create a new legend for each subplot
    #         handles, _ = ax.get_legend_handles_labels()
    #         new_labels = ['20', '40', '60', '80', '100']
    #         ax.legend(handles, new_labels, title='Labeled Fraction (%)')

    #     # Save the figure before showing the plot
    #     plt.savefig(f'{filename}.png')
        
    #     plt.close(fig)  # Close the figure after saving

    #     plt.show()


    """Feature Importances"""

    def plot_model_feature_importances(self, df, model_name, balancing_variant, base_filename='feature_importance'):
        # Filter DataFrame for the selected model
        model_columns = [col for col in df.columns if model_name in col]
        df_model = df[model_columns]
        df_model.columns = [col.split('_')[-1].replace('_', ' ') for col in model_columns]
        df_model['features'] = df.index  # Add features as a column
        df_model = df_model.reset_index(drop=True).melt(id_vars='features', var_name='labeled_percentage', value_name='importance')

        # Pivot the DataFrame for the heatmap
        df_pivot = df_model.pivot(index="features", columns="labeled_percentage", values="importance")

        # Modify the index of df_pivot to replace underscores and split if necessary
        df_pivot.index = [' '.join(feature.replace('_', ' ').split()) for feature in df_pivot.index]

        # Sort features by importance and select the top 10
        top_10_features = df_pivot.mean(axis=1).sort_values(ascending=False).head(50).index

        # Now select the top 10 features in df_pivot
        df_pivot = df_pivot.loc[top_10_features]

        # Plot with increased figure size
        plt.figure(figsize=(14, 8))  # Adjust the figure height to fit 10 features
        sns.heatmap(df_pivot, annot=False, fmt='.2f', cmap='Blues')
        plt.title(f'Selected Features for {model_name} ({balancing_variant})')

        # Customizing the x-axis labels
        labeled_percentages = [f'{int(float(label) * 100)}' for label in df_pivot.columns]
        plt.xticks(np.arange(len(labeled_percentages)) + 0.5, labeled_percentages)  # Adjusting the position to center

        plt.xlabel('Labeled Fraction (%)')
        plt.ylabel('Feature')
        plt.yticks(rotation=0)  # Keep the feature names horizontal for readability

        plt.tight_layout()
        # Save the figure if a base filename is provided
        if base_filename:
            full_filename = f"{balancing_variant}_{base_filename}_{model_name}_all"
            save_figure(plt, full_filename)  # Call custom save function
        plt.close()  # Close the figure after saving

        return full_filename



    # def plot_features_bar_chart(df, model_name, labeled_percentage, base_filename='features'):
    #     # Construct the column name to filter for the selected model and labeled percentage
    #     column_name = f"{model_name}_{labeled_percentage}"
        
    #     # Check if the column exists in the DataFrame
    #     if column_name not in df.columns:
    #         print(f"Column {column_name} not found in the DataFrame.")
    #         return
        
    #     # Create a DataFrame for plotting
    #     df_plot = df[[column_name]]
    #     df_plot['features'] = df.index  # Ensure that 'features' is the DataFrame index
    #     df_plot = df_plot.rename(columns={column_name: 'importance'})
        
    #     # Sort by importance
    #     df_plot = df_plot.sort_values('importance', ascending=False)
        
    #     # Plot with seaborn
    #     plt.figure(figsize=(10, len(df_plot) // 2))  # Adjust size as needed
    #     barplot = sns.barplot(data=df_plot, y='features', x='importance', dodge=False, palette='viridis')

    #     plt.title(f'Feature Importances for {model_name} at {labeled_percentage} labeled data')
    #     plt.xlabel('Importance')
    #     plt.ylabel('Feature')

    #     plt.tight_layout()

    #     # Save the figure if a base filename is provided
    #     full_filename = f"{base_filename}_{model_name}_{labeled_percentage}.png"
    #     plt.savefig(full_filename)  # Save the figure
    #     plt.show()  # Show the figure
    #     plt.close()  # Close the figure after saving





    """Classification Report"""

    def extract_and_compare_classification_reports(self, df, labeled_percentage):
        # Extract classification reports for each model
        report_columns = ['precision', 'recall', 'f1-score', 'support']
        comparison_data = []

        # Filter the DataFrame for the given labeled percentage
        df_filtered = df[df['labeled_percentage'] == labeled_percentage]

        for _, row in df_filtered.iterrows():
            model = row['model']
            report = row['test_report']

            # Extract metrics for each class label in the report
            for label, metrics in report.items():
                if label in ['accuracy', 'macro avg', 'weighted avg']:
                    continue  # Skip the overall metrics
                comparison_data.append({
                    'model': model,
                    'label': label,
                    'precision': metrics.get('precision', None),
                    'recall': metrics.get('recall', None),
                    'f1-score': metrics.get('f1-score', None),
                    'support': metrics.get('support', None),
                })

        # Convert to DataFrame for easy manipulation and comparison
        comparison_df = pd.DataFrame(comparison_data)

        # Pivot to get models as columns and metrics as rows for each label
        comparison_pivot = comparison_df.pivot_table(
            index='label', 
            columns='model', 
            values=report_columns
        )

        return comparison_pivot
    
    def train_classification_reports(self, df, labeled_percentage):
        # Extract classification reports for each model
        report_columns = ['precision', 'recall', 'f1-score', 'support']
        comparison_data = []

        # Filter the DataFrame for the given labeled percentage
        df_filtered = df[df['labeled_percentage'] == labeled_percentage]

        for _, row in df_filtered.iterrows():
            model = row['model']
            report = row['train_report']

            # Extract metrics for each class label in the report
            for label, metrics in report.items():
                if label in ['accuracy', 'macro avg', 'weighted avg']:
                    continue  # Skip the overall metrics
                comparison_data.append({
                    'model': model,
                    'label': label,
                    'precision': metrics.get('precision', None),
                    'recall': metrics.get('recall', None),
                    'f1-score': metrics.get('f1-score', None),
                    'support': metrics.get('support', None),
                })

        # Convert to DataFrame for easy manipulation and comparison
        comparison_df = pd.DataFrame(comparison_data)

        # Pivot to get models as columns and metrics as rows for each label
        comparison_pivot = comparison_df.pivot_table(
            index='label', 
            columns='model', 
            values=report_columns
        )

        return comparison_pivot
    

    """Hyperparameter Tuning"""

    def compare_best_params_across_runs(self, cv_metrics_df):
        # Dictionary to store the comparison tables for each model
        comparison_tables = {}

        # Loop through each unique model in the DataFrame
        for model in cv_metrics_df['model'].unique():
            # Filter the DataFrame for the current model
            model_df = cv_metrics_df[cv_metrics_df['model'] == model]

            # List to store the processed rows for the model
            processed_rows = []

            for _, row in model_df.iterrows():
                # Prepare a dictionary to hold the hyperparameters for the current row
                param_dict = {'labeled_percentage': row['labeled_percentage']}

                # Convert the string representation of dictionary to actual dictionary if needed
                best_params = row['best_params']
                if isinstance(best_params, str):
                    best_params = json.loads(best_params.replace("'", "\""))

                # Add the best parameters to the row dictionary
                param_dict.update(best_params)

                # Append the row dictionary to the processed rows list
                processed_rows.append(param_dict)

            # Convert the processed rows to a DataFrame
            model_params_df = pd.DataFrame(processed_rows)

            # Set the labeled_percentage as the index
            model_params_df.set_index('labeled_percentage', inplace=True)

            # Store the DataFrame in the comparison_tables dictionary with the model as the key
            comparison_tables[model] = model_params_df

        return comparison_tables

    








