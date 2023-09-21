#importing basic packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Loading the data
data0 = pd.read_csv('Phishing-ML-Testing\\urldata.csv')
data0.head()

#Dropping the Domain column
data = data0.drop(['Domain'], axis = 1).copy()

# Sepratating & assigning features and target columns to X & y
y = data['Label']
X = data.drop('Label',axis=1)
# X.shape, y.shape

# Splitting the dataset into train and test sets: 80-20 split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, random_state = 12)
# X_train.shape, X_test.shape

#XGBoost Classification model
from xgboost import XGBClassifier

# instantiate the model
model = XGBClassifier(learning_rate=0.4,max_depth=7)
#fit the model
model.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_test_xgb = model.predict(X_test)
print(X_test)
y_train_xgb = model.predict(X_train)

# save XGBoost model to file
import pickle
pickle.dump(model, open("My.pickle.dat", "wb"))
     
import xgboost as xgb
model.save_model('MyClassifier.model')