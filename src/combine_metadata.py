import pandas as pd

def combine_metadata(official_csv, wikipedia_csv, output_csv):
    df_official = pd.read_csv(official_csv)
    df_wiki = pd.read_csv(wikipedia_csv)
    df_combined = pd.concat([df_official, df_wiki], ignore_index=True)
    df_combined = df_combined.drop_duplicates()
    df_combined = df_combined.dropna(how='all')
    df_combined.to_csv(output_csv, index=False)
    print("Metadata combining done! Output:", output_csv)

if __name__ == "__main__":
    combine_metadata(
        'processed_data/official/official_metadata_cleaned.csv',
        'processed_data/wikipedia/wikipedia_metadata_cleaned.csv',
        'processed_data/combined_metadata.csv'
    )