# --- Imports from your provided code ---
import re
import string
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- NLTK Data Check and Download ---
# This ensures that the required NLTK data is available before we use it.
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# --- Preprocessing functions from your provided code ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def remove_html(text):
    """Removes HTML tags from a string."""
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    """Removes punctuation from a string."""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_stopwords(text):
    """Removes common English stopwords from a string."""
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

def stem_text(text):
    """Applies stemming to the words in a string."""
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_tokens)

def preprocess_text(text):
    """
    Combines all preprocessing steps into a single function.
    Handles non-string inputs gracefully.
    """
    if not isinstance(text, str):
        return ""
        
    text = remove_html(text)
    text = text.lower()  # Convert to lowercase
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    
    return text

# --- Data Loading function from your provided code ---
def load_data(file_path):
    """
    Loads a CSV file and handles potential encoding errors.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file at {file_path} was not found.")
        return None
    
    try:
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

# --- Main execution block ---
if __name__ == "__main__":
    # Define file paths.
    # We assume the raw data is in 'data/raw/' and the preprocessed data
    # will be saved to 'data/preprocessed/'.
    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(project_root, '..', 'data', 'raw', 'IMDB_Dataset.csv')
    preprocessed_data_dir = os.path.join(project_root, '..', 'data', 'preprocessed')
    preprocessed_data_path = os.path.join(preprocessed_data_dir, 'preprocessed_movie_reviews.csv')
    
    # Create the preprocessed directory if it doesn't exist.
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    
    # Step 1: Load the raw data.
    print("Loading raw data...")
    df = load_data(raw_data_path)
    
    if df is not None:
        # Step 2: Apply the preprocessing function to the 'review' column.
        print("\nPreprocessing movie reviews...")
        # We use a `.copy()` to avoid a SettingWithCopyWarning from pandas.
        df_processed = df.copy()
        df_processed['preprocessed_text'] = df_processed['review'].apply(preprocess_text)
        
        # Step 3: Rename the sentiment column to match the training script's expectations.
        # This makes sure the two scripts are compatible.
        df_processed.rename(columns={'sentiment': 'sentiment_label'}, inplace=True)
        
        # Display the first few rows of the preprocessed data for verification.
        print("\nPreprocessing complete. Here's a preview:")
        print(df_processed[['preprocessed_text', 'sentiment_label']].head())

        # Step 4: Save the preprocessed data to a new CSV file.
        print("\nSaving preprocessed data to CSV...")
        df_processed.to_csv(preprocessed_data_path, index=False)
        print(f"Data successfully saved to '{preprocessed_data_path}'")
    else:
        print("\nData loading failed. Preprocessing aborted.")
