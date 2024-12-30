from flask import Flask, request, jsonify
import joblib
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask App
app = Flask(__name__)

# Load pre-trained model and vectorizer
model = joblib.load('decision_tree_model.pkl')  # Replace with your trained Decision Tree model
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Replace with your saved TF-IDF vectorizer

# Home route for testing
@app.route('/')
def home():
    return jsonify({'message': 'Backend is running!'})


# Predict method
@app.route('/predict', methods=['POST'])
def predict_category():
    try:
        # Get data from request
        data = request.get_json()
        descriptions = data.get('descriptions')  # List of attraction descriptions
        
        if not descriptions:
            return jsonify({'error': 'No descriptions provided'}), 400
        
        # Preprocess descriptions and transform using TF-IDF vectorizer
        transformed_data = vectorizer.transform(descriptions)
        
        # Predict categories using the Decision Tree model
        predictions = model.predict(transformed_data)
        
        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Generate-itinerary method
@app.route('/generate-itinerary', methods=['POST'])
def generate_itinerary():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract user preferences
        preferences = data.get('preferences')  # User preferences dict
        attractions = data.get('attractions')  # List of attraction details
        
        if not preferences or not attractions:
            return jsonify({'error': 'Missing preferences or attractions'}), 400
        
        # Process user preferences
        time_blocks = preferences.get('schedule_tensity', 3)
        start_time = preferences.get('start_time', '09:00')
        mode_of_transport = preferences.get('mode_of_transport', 'MTR')
        
        # Generate a simple itinerary (this can be extended to use advanced algorithms)
        itinerary = []
        current_time = start_time
        
        for attraction in attractions:
            name = attraction.get('name')
            location = attraction.get('location')
            duration = attraction.get('duration', 1)  # Default duration = 1 hour
            
            # Add to itinerary
            itinerary.append({
                'name': name,
                'location': location,
                'start_time': current_time,
                'duration': f'{duration} hour(s)',
                'transport': mode_of_transport
            })
            
            # Increment current time (basic scheduling logic for simplicity)
            # In production, consider time parsing libraries like datetime
            current_time = f"{int(current_time.split(':')[0]) + duration}:00"
        
        # Return the generated itinerary
        return jsonify({'itinerary': itinerary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)

