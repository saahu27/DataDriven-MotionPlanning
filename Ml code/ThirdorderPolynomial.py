# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:33:59 2021

@author: sahru
"""
import numpy as np
import pandas as pd
images = pd.read_csv(r"C:\Users\sahru\Downloads\mnist_train_binary.csv")
        
images_test = pd.read_csv(r"C:\Users\sahru\Downloads\mnist_test_binary.csv")

X_test_preprocessed = images_test.iloc[:,1:]
X_test_preprocessed = X_test_preprocessed/255

Y_test = images_test['label']
Y = images['label']

#12163,1

X_train_preprocessed = images.iloc[:,1:]
X_train_preprocessed = X_train_preprocessed/255

X_train_preprocessed=np.reshape(np.ravel(X_train_preprocessed),(X_train_preprocessed.shape[0],28,28))

X_test_preprocessed=np.reshape(np.ravel(X_test_preprocessed),(X_test_preprocessed.shape[0],28,28))


Intensity = []
Symmetry = []

O = np.ones(12163)
   
for i in range(0,len(X_train_preprocessed)):
    Intensity.append(X_train_preprocessed[i].mean())
    Symmetry.append(np.mean(abs(X_train_preprocessed[i] - np.flip(X_train_preprocessed[i]))))


Intensity = np.array(Intensity)
Symmetry = np.array(Symmetry)
X_train_processed = np.column_stack((O,Intensity,Symmetry))

# adding bias to the input
np.random.seed(42)
bias = np.random.uniform(low=0.0,high=1.0)
X_train_processed[1:,:] = X_train_processed[1:,:] 


X_transformed = np.column_stack((O,Intensity,Symmetry,Intensity**2,Intensity * Symmetry,Symmetry**2,(Intensity)**3,(Intensity)**2 * Symmetry,Intensity * (Symmetry)**2,(Symmetry)**3,))
Intensity_test = []
Symmetry_test = []

   
for i in range(0,len(X_test_preprocessed)):
    Intensity_test.append(X_test_preprocessed[i].mean())
    Symmetry_test.append(np.mean(abs(X_test_preprocessed[i] - np.flip(X_test_preprocessed[i]))))
Intensity_test = np.array(Intensity_test)
Symmetry_test = np.array(Symmetry_test)
O = np.ones(2027)   
X_test_processed = np.column_stack((O,Intensity_test,Symmetry_test))
X_transformed_test = np.column_stack((O,Intensity_test,Symmetry_test,Intensity_test**2,Intensity_test * Symmetry_test,Symmetry_test**2,(Intensity_test)**3,(Intensity_test)**2 * Symmetry_test,Intensity_test * (Symmetry_test)**2,(Symmetry_test)**3,))

X_dagger = np.dot(np.linalg.inv(np.dot(X_transformed.T, X_transformed)), X_transformed.T)
weights = np.dot(X_dagger, Y)
#print(self.X_dagger.shape)

R,C = X_transformed.shape

for i in range(0,len(Y)):
    if(Y[i] == 5):
        Y[i] = -1
for epoch in range(0,2):
    i = np.random.randint(low=0,high=R)
    s = np.array(np.dot(weights.T,np.array(X_transformed[i])))
    
    print(s)
    if((Y[i] * s) <= 1) :
        weights = weights + 0.1*(((Y[i])-(s))* X_transformed[i])
 

X_dagger1 = np.dot(np.linalg.inv(np.dot(X_transformed.T, X_transformed)), X_transformed.T)
Y_pred = np.sign(np.dot(weights.T, X_dagger1)).T

m,n = X_transformed.shape
for i in range(0,len(Y)):
    if(Y[i] == 5):
        Y[i] = 1
    else:
        Y[i] = -1

E_in = sum(Y_pred != Y)  / m
print("third order polynomial training error",E_in)

X_dagger1 = np.dot(np.linalg.inv(np.dot(X_transformed_test.T, X_transformed_test)), X_transformed_test.T)
Y_pred = np.sign(np.dot(weights.T, X_dagger1)).T
m,n = X_transformed.shape
for i in range(0,len(Y)):
    if(Y[i] == 5):
        Y[i] = 1
    else:
        Y[i] = -1

E_in = sum(Y_pred != Y_test)  / m
print("third order polynomial test error",E_in)

print(X_transformed_test.shape)

