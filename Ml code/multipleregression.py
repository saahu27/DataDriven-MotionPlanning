# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:53:37 2021

@author: sahru
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

# Load the data
sales = pd.read_csv(r"C:\Users\sahru\Downloads\mlr05.csv")

print(sales.describe())

#Preparing the Data

X = sales[['X2', 'X3', 'X4',
       'X5', 'X6']]
Y = sales['X1']

#splitting data to Training and Test Sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=7, train_size =20, shuffle=False, random_state=0)

#Training the Algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#To see what are the Coefficients Chosen
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

#to Predict
y_pred = regressor.predict(X_test)

#to compare the output values of the test set with predictions
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
print(df)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))