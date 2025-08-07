import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

def train_and_evaluate_model(file_path):
    """
    Loads preprocessed data, trains a Logistic Regression model, and
    evaluates its performance.

    Args:
        file_path (str): The path to the preprocessed data CSV file.

    Returns:
        tuple: A tuple containing the trained TfidfVectorizer and the
               trained LogisticRegression model, or (None, None) if an
               error occurs.
    """
    if not os.path.exists(file_path):
        print(f"Error: The preprocessed data file at {file_path} was not found.")
        print("Please run the preprocessing script first.")
        return None, None
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"An error occurred while loading the preprocessed data: {e}")
        return None, None
    
    if df.shape[0] < 2:
        print("Error: The dataset contains too few rows for training.")
        return None, None

    print("--- Data loaded successfully for training ---")
    print(f"Total rows: {df.shape[0]}")
    
    # Split the data into features (X) and labels (y)
    X = df['preprocessed_text'].astype(str) # Ensure text is treated as a string
    y = df['sentiment_label']
    
    # Split the data into training and testing sets
    # We use a stratify argument to ensure the same proportion of classes
    # is present in both the training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n--- Training model... ---")
    
    # Step 1: Feature Extraction using TF-IDF
    # TF-IDF stands for Term Frequency-Inverse Document Frequency.
    # It converts text into a numerical format, giving more weight to
    # words that are important in a document but not common across all documents.
    vectorizer = TfidfVectorizer(max_features=5000) # Use the top 5000 most frequent words
    
    # Fit the vectorizer on the training data and transform it
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Transform the test data using the *same* fitted vectorizer
    X_test_vec = vectorizer.transform(X_test)
    
    # Step 2: Model Training using Logistic Regression
    # Logistic Regression is a simple but effective model for classification tasks.
    model = LogisticRegression(max_iter=1000) # Increase max_iter for convergence
    model.fit(X_train_vec, y_train)
    
    print("--- Model training complete ---")
    
    # Step 3: Model Evaluation
    print("\n--- Evaluating model performance ---")
    y_pred = model.predict(X_test_vec)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return vectorizer, model

def save_model(vectorizer, model, output_dir):
    """
    Saves the trained TfidfVectorizer and the Logistic Regression model
    to the specified directory.

    Args:
        vectorizer (TfidfVectorizer): The trained TF-IDF vectorizer.
        model (LogisticRegression): The trained classification model.
        output_dir (str): The directory where the model and vectorizer will be saved.
    """
    # Create the directory if it does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define file paths for saving
    model_path = os.path.join(output_dir, 'sentiment_model.pkl')
    vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    
    print("\n--- Saving the trained model and vectorizer... ---")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print("Saving complete.")

# --- Main execution block ---
if __name__ == "__main__":
    # Get the directory of the current script
    project_root = Path(__file__).parent.parent
    
    # Define paths for the preprocessed data and the output directory
    preprocessed_data_path = project_root / 'data' / 'preprocessed' / 'preprocessed_movie_reviews.csv'
    model_output_dir = project_root / 'models'

    # Train and evaluate the model
    trained_vectorizer, trained_model = train_and_evaluate_model(preprocessed_data_path)
    
    # Save the model and vectorizer if training was successful
    if trained_model and trained_vectorizer:
        save_model(trained_vectorizer, trained_model, model_output_dir)