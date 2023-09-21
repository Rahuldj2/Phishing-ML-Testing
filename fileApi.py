# Import necessary libraries
from flask import Flask, request, jsonify
import pickle
from URLFeatureExtraction import featureExtraction
from flask_cors import CORS

# Load the trained model
with open('Phishing-ML-Testing\XGBoostClassifier.pickle.dat', 'rb') as model_file:
    model = pickle.load(model_file)

# Create a Flask app
app = Flask(__name__)

# Allow cross-origin requests (CORS)
CORS(app)

# Define a route for predicting
@app.route('/process_url', methods=['POST'])
def process_url():
    # Get the URL from the POST request
    data = request.get_json(force=True)
    url = data['url']
    
    # Perform feature extraction on the URL (similar to what you did during training)
    features = featureExtraction("https://www.youtube.com/watch?v=I1refTZp-pg")
    # print(features)
    # Make predictions using the trained model
    prediction = model.predict([features])
    
    # Return the prediction as JSON response
    response = {'prediction': int(prediction[0])}
    # print(response)
    
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
