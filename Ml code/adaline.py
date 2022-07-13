# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 16:53:11 2021

@author: sahru
"""

class CustomAdaline(object):
    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.0001):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate

    def fit(self, X, y):
            
        rgen = np.random.RandomState(self.random_state)
        #Draw random samples from a normal (Gaussian) distribution.
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        for _ in range(self.n_iterations):
              activation_function_output = self.activation_function(self.net_input(X))
              errors = y - activation_function_output
              self.coef_[1:] = self.coef_[1:] + self.learning_rate*(X.T).dot(errors)
              self.coef_[0] = self.coef_[0] + self.learning_rate*errors.sum() 
    def net_input(self, X):
            weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]
            return weighted_sum
    def activation_function(self, X):
            return X
    def predict(self, X):
        return np.where(self.activation_function(self.net_input(X)) >= 0.0, 1, 0) 
    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):

            output = self.predict(xi)
            if(target != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count)/total_data_count
        return self.score_

import numpy as np
np.random.seed(100)
w = [-1,1,-1]
X_training = np.ones([100,3])
X_random = 12*np.random.random((100,2))
X_training[0:100, 0:2] = X_random
Y_training = np.sign(np.dot(w, X_training.T))
x1 = X_training[:,0:2]


X_test = np.ones([10000,3])
X_random1 = 12*np.random.random((10000,2))
X_test[0:10000, 0:2] = X_random1
Y_test = np.sign(np.dot(w, X_test.T))
x2 = X_test[:,0:2]

adaline = CustomAdaline(n_iterations = 100,learning_rate=0.0001)
adaline.fit(x1, Y_training)
s=adaline.score(x2, Y_test)

#p=prcptrn.score(x1, Y_training)
print(s)

import matplotlib.pyplot as plt

X=x1[:,0]
Y=x1[:,1]
plt.scatter(X, Y)
plt.show()






















