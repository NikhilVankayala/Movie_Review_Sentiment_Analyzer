# app/app.py
import joblib
from flask import Flask, request, render_template, jsonify
from pathlib import Path

# Initialize the Flask app
app = Flask(__name__)

# Define paths to the trained model and label encoder
# Path should be relative to the project root directory
project_root = Path(__file__).parent.parent
models_dir = project_root / 'models'
pipeline_path = models_dir / 'sentiment_pipeline.pkl'
encoder_path = models_dir / 'label_encoder.pkl'

# Load the trained model and label encoder once when the app starts
sentiment_pipeline = None
label_encoder = None
try:
    if pipeline_path.exists() and encoder_path.exists():
        sentiment_pipeline = joblib.load(pipeline_path)
        label_encoder = joblib.load(encoder_path)
        print("Model and label encoder loaded successfully.")
    else:
        print("Error: Model or label encoder not found.")
        print(f"Expected model path: {pipeline_path}")
        print(f"Expected encoder path: {encoder_path}")
        print("Please ensure you have run 'train_model.py' and the files are in the 'models' directory.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

# --- Routes ---

@app.route('/')
def home():
    """Renders the main page of the application."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the web page."""
    if not sentiment_pipeline or not label_encoder:
        # Return a more user-friendly error message for the front end
        return jsonify({'error': 'Prediction service is unavailable. Model not loaded.'}), 500

    try:
        # Get the review text from the form submission
        data = request.form.get('review_text')
        if not data:
            return jsonify({'error': 'No review text provided.'}), 400

        # Predict the sentiment
        prediction_encoded = sentiment_pipeline.predict([data])
        predicted_sentiment = label_encoder.inverse_transform(prediction_encoded)[0]

        # Return the prediction as a JSON response
        return jsonify({'prediction': predicted_sentiment})
    except Exception as e:
        # Handle any errors during prediction
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True)