# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 10:57:26 2021

@author: sahru
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


#To plot the figures
    # pixels = np.array(X.iloc[3333,:], dtype='uint8')
    # pixels = pixels.reshape((28, 28))
    # plt.title('Label is {Y}'.format(Y=Y))
    # plt.imshow(pixels, cmap='gray')
    # plt.show()


#Training the Algorithm



# #to compare the output values of the test set with predictions
# df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
# print(df)

# from sklearn import metrics
# print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))


# to compare our model's accuracy with sklearn model
from sklearn.linear_model import LogisticRegression

images = pd.read_csv(r"C:\Users\sahru\Downloads\mnist_train_binary.csv")
        
images_test = pd.read_csv(r"C:\Users\sahru\Downloads\mnist_test_binary.csv")
    
X_test_preprocessed = images_test.iloc[:,1:]
X_test_preprocessed = X_test_preprocessed/255
Y_test = images_test['label']
Y = images['label']

for i in range(0,len(Y)):
    if(Y[i] ==5):
        Y[i] = 1
    else:
        Y[i] = -1
    

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
        
X_train_processed = np.column_stack((O,Intensity,Symmetry))
    
    #adding bias to the input
np.random.seed(42)
bias = np.random.uniform(low=-1.0,high=1.0)
X_train_processed[:,1:] = X_train_processed[:,1:] + bias
    
    # X_tranformed = np.column_stack((O,(Intensity)**3,(Symmetry)**3,))
    
Intensity_test = []
Symmetry_test = []
    
for i in range(0,len(X_test_preprocessed)):
    Intensity_test.append(X_test_preprocessed[i].mean())
    Symmetry_test.append(np.mean(abs(X_test_preprocessed[i] - np.flip(X_test_preprocessed[i]))))
    
O = np.ones(2027)   
X_test_processed = np.column_stack((O,Intensity_test,Symmetry_test))
X_test_processed[:,1:] = X_test_processed[:,1:] + bias
    
    

    
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_processed, Y)


#to Predict
Y_pred = regressor.predict(X_train_processed)

count = 0 
correctly_classified = 0 

for count in range( np.size( Y_pred ) ) :  
        
    if Y[count] == Y_pred[count] :            
        correctly_classified = correctly_classified + 1
          
        # if Y_test[count] == Y_pred1[count] :            
        #     correctly_classified1 = correctly_classified1 + 1
              
    count = count + 1
          
print( "Accuracy on test set by our model       :  ", ( correctly_classified / count ) * 100 )


    
model1 = LogisticRegression()    
model1.fit( X_train_processed,Y)
Y_pred = model1.predict(X_train_processed)     
    
    # measure performance    
correctly_classified = 0    
correctly_classified1 = 0
      
    # counter    
count = 0 
for count in range( np.size( Y_pred ) ) :  
        
    if Y[count] == Y_pred[count] :            
        correctly_classified = correctly_classified + 1
          
        # if Y_test[count] == Y_pred1[count] :            
        #     correctly_classified1 = correctly_classified1 + 1
              
    count = count + 1
          
print( "Accuracy on test set by our model q      :  ", ( correctly_classified / count ) * 100 )
    
    
    
    
    #Insample Error
m,n = X_train_processed.shape
E_in = sum(Y_pred != Y)  / m
print("In sample error: ", E_in)
    




































