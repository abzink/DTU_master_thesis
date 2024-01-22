import os
import matplotlib.pyplot as plt

# Function to save figures in the 'figures' directory
def save_figure(fig, filename):
    """
    Save the given matplotlib figure in both PNG and SVG formats in a 'figures' directory.
    
    Parameters:
    - fig: matplotlib figure object to be saved
    - filename: base name of the file (without extension)
    """
    # Determine the root directory based on the script location
    root_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Set the path for the 'figures' directory
    figures_directory = os.path.join(root_directory, 'figures')
    
    # Create the 'figures' directory if it doesn't exist
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)
    
    # Define the paths for saving the figure in PNG and SVG formats
    png_filename = os.path.join(figures_directory, filename + ".png")
    svg_filename = os.path.join(figures_directory, filename + ".svg")

    # Save the figures
    fig.savefig(png_filename, format='png', dpi=500)
    fig.savefig(svg_filename, format='svg')
