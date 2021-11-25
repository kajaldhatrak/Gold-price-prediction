import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle

data=pd.read_csv('static/gld_price_data.csv')

X = data.drop(['Date','GLD'],axis=1) # features from 
Y = data['GLD'] # target variable
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))

#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1447.16,78.47,15.18,1.47]]))