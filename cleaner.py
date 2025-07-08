import pandas as pd

def clean_url_data(input_file, output_file, url_column='url'):
    """
    Simple URL data cleaner that:
    - Removes rows with null/empty URLs
    - Removes duplicate URL rows (keeping first occurrence)
    - Prints count of URLs by phishing class
    """
    # Load the data
    df = pd.read_csv(input_file)
    original_count = len(df)
    
    # 1. Remove rows where URL is null/empty
    df_clean = df.dropna(subset=[url_column])
    df_clean = df_clean[df_clean[url_column].astype(str).str.strip() != '']
    
    # 2. Remove duplicate URLs (keeping first occurrence)
    df_clean = df_clean.drop_duplicates(subset=[url_column], keep='first')
    
    # Print class counts
    print("\nURL Count by Class:")
    print(df_clean['phishing'].value_counts().to_string())
    print(f"\nOriginal rows: {original_count}")
    print(f"Cleaned rows: {len(df_clean)}")
    print(f"Rows removed: {original_count - len(df_clean)}")
    
    # Save cleaned data
    df_clean.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to {output_file}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python clean.py input.csv output.csv [url_column_name]")
        sys.exit(1)
        
    url_column = sys.argv[3] if len(sys.argv) > 3 else 'url'
    clean_url_data(sys.argv[1], sys.argv[2], url_column)