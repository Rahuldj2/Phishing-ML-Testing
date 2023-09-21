import xgboost as xgb
import pickle
from flask import Flask, request, jsonify
import numpy as np
from URLFeatureExtraction import featureExtraction

app = Flask(__name__)

# Load the machine learning model
# with open('Phishing-ML-Testing\XGBoostClassifier.pickle.dat', 'rb') as model_file:
#     model = pickle.load(model_file)


# Load the model using the new version of XGBoost
model = xgb.Booster(model_file='MyClassifier.model')
# print(model)


@app.route('/process', methods=['POST'])
def process():
    try:
        # Get the input data from the POST request
        # data = request.json
        features = featureExtraction(
            "https://www.youtube.com/watch?v=I1refTZp-pg")
        print("Yes")
        print(features)
        dmatrix = xgb.DMatrix(np.array([features]))

        # Make predictions using the loaded model
        prediction = model.predict(dmatrix)
        # Make predictions using the loaded model
        # prediction = model.predict(np.array([features]))
        print(prediction)
        print("Woww")
        # Convert the prediction to a list and return as JSON
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
