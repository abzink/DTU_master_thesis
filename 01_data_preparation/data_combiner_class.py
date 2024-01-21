# python data_combiner.py --data_paths data1.csv data2.csv data3.csv --mapping_path products_labeled.csv --output_all data_all.csv


import pandas as pd
import argparse

class DataCombiner:
    def __init__(self, data_paths, mapping_path):
        """
        Initialize the DataCombiner object with paths of the data files 
        and the mapping file.
        
        :param data_paths: List of paths to data files to be combined.
        :param mapping_path: Path to categories mapping file.
        """
        self.data_paths = data_paths
        self.mapping_path = mapping_path
        self.df = pd.DataFrame()  # Initialize an empty DataFrame to store concatenated data.
    
    def combine_data(self):
        """
        Read and concatenate data from the provided paths.
        """
        for path in self.data_paths:
            temp_df = pd.read_csv(path, sep=';')
            self.df = pd.concat([self.df, temp_df], ignore_index=True)
    
    def add_category(self):
        """
        Add category column from mapping file which includes 257 labeled products to the main dataset.
        """
        df_cat = pd.read_csv(self.mapping_path, sep=';')
        
        # Extract OBITNO from the column with OBITNO and product description
        df_cat['OBITNO'] = df_cat['OBITNO'].str.split(' ').str[0]
        self.df['OBITNO'] = self.df['OBITNO'].str.strip()
        
        self.df = pd.merge(self.df, df_cat[['OBITNO', 'category']], on='OBITNO', how='left')
        self.df['category'] = self.df['category'].fillna(-1)  # Assign -1 label when category is not known
    
    def save(self, output_path):
        """
        Save processed data to the specified output path.
        
        :param output_path: Path for saving the combined data.
        """
        self.df.to_csv(output_path, index=False)
        
if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Combine and save data from multiple sources.')
        
    # Add arguments
    parser.add_argument('--data_paths', nargs='+', help='List of paths to data files to be combined.')
    parser.add_argument('--mapping_path', help='Path to categories mapping file.')
    parser.add_argument('--output_all', help='Path for saving the combined data.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize DataCombiner with provided arguments and perform operations
    combiner = DataCombiner(args.data_paths, args.mapping_path)
    combiner.combine_data()
    combiner.add_category()
    combiner.save(args.output_all)
