import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

def stem_text(text):
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_tokens)

def preprocess_text(text):
    # Check if the input is a string. If not, return an empty string or handle appropriately.
    if not isinstance(text, str):
        return ""
        
    text = remove_html(text)
    text = text.lower()  # Convert to lowercase
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    
    return text

if __name__ == "__main__":
    # Example usage:
    sample_review = "This movie was absolutely amazing! I loved the acting and the plot. <br /><br />It was not boring at all. What a great film."
    print("Original review:")
    print(sample_review)
    
    processed_review = preprocess_text(sample_review)
    
    print("\nProcessed review:")
    print(processed_review)
