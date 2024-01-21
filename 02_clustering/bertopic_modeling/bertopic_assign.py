import os
import pandas as pd
import numpy as np
import re
import shutil

BASE_DIR = 'data/bert_topics_50_vol_2/'
KEYWORDS_FILE = 'data/categories_keywords.csv'


def get_files_by_pattern(pattern):
    """Fetch filenames from BASE_DIR that match a given pattern."""
    return [file for file in os.listdir(BASE_DIR) if pattern in file]

def extract_topic_set_id(filename):
    """Extract a substring starting from 'neighbors' to the end using regex."""
    match = re.search(r'neighbors.*', filename)
    if match:
        return match.group(0)
    return None

def assign_categories_to_topics(topic_file):
    """Assign categories to topics based on keyword matches."""
    topics_df = pd.read_csv(BASE_DIR + topic_file)
    keywords_df = pd.read_csv(KEYWORDS_FILE, sep=';')

    def get_categories(representation, topic):
        """Inner function to categorize topics based on keywords."""
        if topic == '-1':
            return '-1'

        words = [word.strip().replace("'", "") for word in representation.strip("[]").split(",")]

        matched_keywords = []
        i = 0
        while i < len(words):
            word = words[i]
            if i < len(words) - 1:
                bigram = word + " " + words[i+1]
                if bigram in keywords_df['keyword'].values:
                    matched_keywords.append(bigram)
                    i += 2
                    continue
            if word not in matched_keywords and word in keywords_df['keyword'].values:
                matched_keywords.append(word)
            i += 1

        matched_categories = keywords_df[keywords_df['keyword'].isin(matched_keywords)].groupby('category')['keyword'].count()

        if len(matched_categories) != 1:
            return '-1'
        else:
            return matched_categories.index[0]

    topics_df['category'] = topics_df.apply(lambda row: get_categories(row['Representation'], str(row['Topic'])), axis=1)

    return topics_df

def compute_topic_metrics(topic_df, all_topics_df, categorized_doc_count, total_doc_count):
    """Calculate topic-based metrics."""
    unique_categories = topic_df[topic_df['category'] != '-1']
    ratio_topic_unique = len(unique_categories) / len(all_topics_df)
    ratio_products_categorized = categorized_doc_count / total_doc_count
    return ratio_topic_unique, ratio_products_categorized


def categorize_products_by_topic(topic_df, doc_topic_file):
    """Assign categories to documents based on their topic."""
    doc_topic_df = pd.read_csv(BASE_DIR + doc_topic_file)
    doc_topic_df = doc_topic_df[doc_topic_df['Topic'] != '-1']
    merged_df = doc_topic_df.merge(topic_df[['Topic', 'category']], on='Topic', how='left')
    merged_df.rename(columns={'category': 'Category'}, inplace=True)
    
    # Overwrite category for products that have "service kit" in their Document but are not in Category A
    merged_df.loc[(merged_df['Document'].str.contains('service kit', case=False)) & (merged_df['Category'] != 'A'), 'Category'] = 'A'
    merged_df.loc[(merged_df['Document'].str.contains('tool', case=False)) & (merged_df['Category'] != 'E'), 'Category'] = 'E'
    
    save_name = doc_topic_file if 'categorized_' in doc_topic_file else 'categorized_' + doc_topic_file
    merged_df.to_csv(BASE_DIR + save_name, index=False)
    
    # Count the number of products that have been assigned a category
    categorized_doc_count = len(merged_df[~merged_df['Category'].isna() & (merged_df['Category'] != '-1')])
    
    return merged_df, categorized_doc_count


def main():
    """Main function to orchestrate the categorization of topics and products."""
    topic_files = get_files_by_pattern('topic_repr_neighbors')
    doc_topic_files = get_files_by_pattern('topic_doc_neighbors')
    topics_eval_data = []

    for topic_file in topic_files:
        all_topics_df = pd.read_csv(BASE_DIR + topic_file)
        topic_df = assign_categories_to_topics(topic_file)
        topic_df.to_csv(BASE_DIR + 'categorized_' + topic_file, index=False)

        for doc_topic_file in doc_topic_files:
            if extract_topic_set_id(doc_topic_file) == extract_topic_set_id(topic_file):
                categorized_df, categorized_doc_count = categorize_products_by_topic(topic_df, doc_topic_file)
                total_doc_count = len(categorized_df)

                ratio_topic_unique, ratio_products_categorized = compute_topic_metrics(topic_df, all_topics_df, categorized_doc_count, total_doc_count)
                
                topics_eval_data.append({
                    'topic_set_id': extract_topic_set_id(topic_file),
                    'ratio_topic_unique': ratio_topic_unique,
                    'ratio_products_categorized': ratio_products_categorized,
                    'num_topics': len(all_topics_df)
                })

    topics_eval_df = pd.DataFrame(topics_eval_data)
    topics_eval_df.drop_duplicates(inplace=True)
    topics_eval_df.sort_values(by='ratio_products_categorized', ascending=False, inplace=True)
    topics_eval_df.to_csv(BASE_DIR + 'topics_eval.csv', index=False)

    best_topic_set_id = topics_eval_df.iloc[0]['topic_set_id']
    files_to_copy = [
        'categorized_topic_repr_' + best_topic_set_id,
        'categorized_topic_doc_' + best_topic_set_id
    ]

    for file_name in files_to_copy:
        source_path = os.path.join(BASE_DIR, file_name)
        dest_path = os.path.join('data', file_name)
        
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
            print(f"Copied {file_name} to 'data' directory.")
        else:
            print(f"File {file_name} not found.")

if __name__ == "__main__":
    main()
