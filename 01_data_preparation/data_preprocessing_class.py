import pandas as pd
import sys
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

class DataProcessor:
    def __init__(self, filename):
        """
        Initialize the DataProcessor object and read the input file into a DataFrame.
        
        :param filename: Path to the input CSV file.
        """
        self.df = pd.read_csv(filename)

    def clean_data(self):
        """
        Perform initial cleaning of the DataFrame by removing unnecessary columns, converting data types,
        handling missing values, and renaming columns.
        """

        # Filter data using MMITCL column which belongs to metadata (is dropped before main analysis) - filter out 'new sales products'
        
        # Define values to filter out
        MMITCL_filter_out = [
                    54210, 54220, 54230, 54240, 54250, 54260, 54270, 54280, 54290, 54300,
                    54310, 54320, 54330, 54340, 54350, 54360, 54370, 54380, 54390, 54400,
                    54410, 54420, 54430, 54440, 54450, 55030, 55040, 55050, 55060, 55070,
                    55080, 55090, 55100, 55210, 55250, 55270, 55290, 55300, 55400, 55710,
                    56110, 56130, 56370, 56380, 56390, 56500, 58000, 58010, 58040, 58050,
                    58060, 58070, 58080, 58090, 58100, 59120, 59130, 59140, 59150, 59160,
                    59170, 59180, 59190, 59200, 59300, 59310, 59320, 59330, 59340, 59350,
                    59360, 59370, 59380, 59390, 59400, 59410, 59420, 59430, 59440, 59450,
                    71010
                ]
        
        # Drop all rows where MMITCL is in the list above
        self.df = self.df[~self.df['MMITCL'].isin(MMITCL_filter_out)]
        
        # Drop unnecessary columns
        self.df.drop(['OBPONR', 'MBBCOS', 'MMITGR', 'MMITCL', 'OKCUNM', 'MBLEAT'], axis=1, inplace=True)

        # Convert strings to floats where ',' is used as a decimal separator
        for col in ['NET_LINE', 'MMNEWE']:
            self.df[col] = self.df[col].str.replace(',', '.').astype(float)

        # Convert 'OBRGDT' to datetime format
        self.df['OBRGDT'] = pd.to_datetime(self.df['OBRGDT'], format='%Y%m%d')
        
        # Convert 'OBORQT' to numeric, handling errors by setting them to NaN
        self.df['OBORQT'] = pd.to_numeric(self.df['OBORQT'], errors='coerce')
        
        # Ensure 'DESCRIPT' is string type
        self.df['DESCRIPT'] = self.df['DESCRIPT'].astype(str)

        # Remove spaces from certain columns
        for col in ['OBCUOR', 'OACUNO', 'OKCSCD']:
            self.df[col] = self.df[col].str.strip()

        # Drop rows where 'OBORQT' is missing
        self.df.dropna(subset=['OBORQT'], inplace=True)
        
        # Convert 'OBORQT' to integer
        self.df['OBORQT'] = self.df['OBORQT'].astype(int)

        # Filter out rows containing certain strings in product name
        self.df = self.df[~self.df['OBITNO'].str.contains('TEST|test|WORK', case=False)]
        
        # Drop duplicate rows and reset index
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # Rename columns for clarity
        new_column_names = {
            'OBRGDT': 'date', 'OBORQT': 'quantity', 'NET_LINE': 'price', 
            'MMNEWE': 'weight', 'DESCRIPT': 'description', 'OBCUOR': 'order_id', 
            'OACUNO': 'customer_id', 'OKCSCD': 'customer_country', 'OBITNO': 'product_id'
        }
        self.df.rename(columns=new_column_names, inplace=True)
        
    def date_features(self):
        """
        Extract various features from the 'date' column and drop the original 'date' column.
        """
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df.drop(['date'], axis=1, inplace=True)

    def aggregate(self):
        """
        Perform data aggregation by product_id and compute various aggregated features.
        """
        df_product = self.df.groupby('product_id').agg(
        description =('description', lambda x: x.value_counts().index[0]), # most frequent description
        month_mode=('month', lambda x: x.value_counts().index[0]),
        quarter_mode=('quarter', lambda x: x.value_counts().index[0]),
        year_mode=('year', lambda x: x.value_counts().index[0]),
        day_week_mode=('day_of_week', lambda x: x.value_counts().index[0]),
        quantity_sum=('quantity', 'sum'), # how much was sold in total (taking into account returns)
        price_sum=('price', 'sum'), # how much money was made in total (taking into account returns)
        unit_weight=('weight', lambda x: x.value_counts().index[0]), # most frequent weight
        customer_country_mode=('customer_country', lambda x: x.value_counts().index[0]),
        customer_country_count=('customer_country', 'nunique'),
        customer_id_count=('customer_id', 'nunique'),
        category=('category', lambda x: x.value_counts().index[0]), # there is always one category asssigned so most freqwuent is the only one
        ).reset_index()

        # From Clement's recommendations
        df_product = df_product[df_product['quantity_sum'] >= 5]
        df_product = df_product[df_product['price_sum'] >= 50]
        df_product['unit_price_mean'] = (df_product['price_sum'] / df_product['quantity_sum']).round(4)

        self.df = df_product

    def preprocess_text(self):
        """
        Preprocess the 'description' column by converting to lowercase, 
        removing punctuation, and removing digits.
        """
        # Copy description column to save original data
        self.df['description_original'] = self.df['description']
        
        # Ensure that 'description' column contains strings
        self.df['description'] = self.df['description'].astype(str)
        
        # Convert to lowercase
        self.df['description'] = self.df['description'].str.lower()

        # Remove any digits
        self.df['description'] = self.df['description'].apply(lambda s: ''.join([i for i in s if not i.isdigit()]))
        
        # Define a translation table to replace punctuation with space
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        
        # Remove any punctuation signs and replace with spaces
        self.df['description'] = self.df['description'].apply(lambda s: s.translate(translator))

        # drop stopwords
        stop = stopwords.words('english')
        self.df['description'] = self.df['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

        # remove the words that are the same characters repeated,for example 'sss'
        self.df['description'] = self.df['description'].apply(lambda x: ' '.join([word for word in x.split() if len(set(word)) > 1]))

        # remove word 'unique' from description
        self.df['description'] = self.df['description'].apply(lambda x: ' '.join([word for word in x.split() if word != 'unique']))

        # Remove extra spaces
        self.df['description'] = self.df['description'].apply(lambda s: ' '.join(s.split()))

    def save(self, output_filename):
        """
        Save the processed DataFrame to a CSV file.
        
        :param output_filename: Path to the output CSV file.
        """
        self.df.to_csv(output_filename, index=False)
        print(f"Processed data saved to {output_filename}")

    def save_labeled(self, labeled_output_filename):
        """
        Filter out rows where category is '-1', and save the resulting DataFrame to a CSV file.
        
        :param labeled_output_filename: Path to the output CSV file for labeled data.
        """
        # Create a copy of the DataFrame, removing rows where category is '-1'
        data_labeled = self.df[self.df.category != '-1']
        
        # Save the resulting labeled DataFrame to the specified file
        data_labeled.to_csv(labeled_output_filename, index=False)
        print(f"Labeled data saved to {labeled_output_filename}")   


if __name__ == "__main__":
    # Check if the filename is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <filename>")
        sys.exit(1)

    # Initialize DataProcessor with input filename from command-line argument
    input_filename = sys.argv[1]
    processor = DataProcessor(input_filename)
    
    # Execute data cleaning, feature extraction, and aggregation methods
    processor.clean_data()
    processor.date_features()
    processor.aggregate()
    processor.preprocess_text()
    
    # Define output filename and save the processed DataFrame
    output_filename = "data/processed_" + input_filename.split("/")[-1]
    labeled_output_filename = "data/processed_labeled_" + input_filename.split("/")[-1] 
    processor.save(output_filename)
    processor.save_labeled(labeled_output_filename)
