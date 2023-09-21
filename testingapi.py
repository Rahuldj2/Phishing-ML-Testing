import pickle
from flask import Flask, request, jsonify
import numpy as np
from URLFeatureExtraction import featureExtraction

import xgboost as xgb

# Load the model using the new version of XGBoost
model = xgb.Booster(model_file='MyClassifier.model')

# Extract features from the URL
features = featureExtraction("https://www.youtube.com/watch?v=I1refTZp-pg")

# Convert the features to a DMatrix object
dmatrix = xgb.DMatrix(np.array([features]))

# Make predictions using the loaded model
prediction = model.predict(dmatrix)
print(prediction)
