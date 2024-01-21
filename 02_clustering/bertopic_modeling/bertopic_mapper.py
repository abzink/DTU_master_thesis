import pandas as pd

def merge_and_process(file1, file2):
    # Load the data
    df_docs_topics = pd.read_csv(file1)
    df = pd.read_csv(file2)

    # Join and process the dataframes
    df_master = df.join(df_docs_topics)
    df_master.rename({'Category': 'category_bertopic', 'Topic': 'topic'}, axis='columns', inplace=True)
    df_master.drop(['Document'], axis=1, inplace=True)

    # Save the result to the desired location
    df_master.to_csv('./data/processed_data_all_bertopic_vol_2.csv', index=False)
