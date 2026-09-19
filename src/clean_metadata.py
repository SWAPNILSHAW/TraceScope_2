import pandas as pd

def clean_metadata(input_path, output_path):
    print("-" * 60)
    print("Cleaning:", input_path)
    try:
        data = pd.read_csv(input_path)
        print("Original data shape:", data.shape)
        data = data.drop_duplicates()
        data = data.dropna(how='all')
        print("Cleaned data shape:", data.shape)
        data.to_csv(output_path, index=False)
        print("File saved successfully at:", output_path)
    except Exception as e:
        print("ERROR processing", input_path)
        print(str(e))

if __name__ == "__main__":
    # Official cleaning
    clean_metadata(
        'processed_data/official/metadata_features.csv',           # Input: Official metadata
        'processed_data/official/official_metadata_cleaned.csv'    # Output: Official cleaned
    )

    # Wikipedia cleaning
    clean_metadata(
        'processed_data/wikipedia/metadata_features.csv',     # Input: Wikipedia metadata
        'processed_data/wikipedia/wikipedia_metadata_cleaned.csv'  # Output: Wikipedia cleaned
    )
