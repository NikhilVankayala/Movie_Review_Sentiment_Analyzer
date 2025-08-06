import os
import pandas as pd

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} was not found.")
        return None
    
    try:
        # Load the CSV file. The IMDb dataset is often encoded in 'latin1' or 'ISO-8859-1'.
        # We'll try a few common encodings to be safe.
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 decoding failed. Trying 'ISO-8859-1' encoding...")
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        
        print(f"Successfully loaded data from {file_path}")
        return df

    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

if __name__ == "__main__":
    # Define the path to the raw data file, based on our project structure
    # You'll need to make sure you have the 'data/raw/' directory created
    # and the 'IMDB_Dataset.csv' file placed inside it.
    project_root = os.path.dirname(os.path.abspath(__file__))
    file_name = "IMDB_Dataset.csv"
    file_path = os.path.join(project_root, '..', 'data', 'raw', file_name)

    # Load the data and perform a quick check
    movie_reviews_df = load_data(file_path)

    if movie_reviews_df is not None:
        print("\n--- Data loaded successfully! ---")
        print("DataFrame shape:", movie_reviews_df.shape)
        print("\nFirst 5 rows of the DataFrame:")
        print(movie_reviews_df.head())
    else:
        print("\n--- Failed to load data. Please check the file path and name. ---")
