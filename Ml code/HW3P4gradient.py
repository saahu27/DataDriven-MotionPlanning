# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 15:44:09 2021

@author: sahru
"""
import numpy as np
import pandas as pd
import sklearn.linear_model
import matplotlib.pyplot as plt



class LogitRegression() :
    def __init__( self, learning_rate, iterations ) :        
        self.learning_rate = learning_rate        
        self.iterations = iterations
          
    # Function for model training    
    def fit( self, X, Y ) :        
        # no_of_training_examples, no_of_features        
        self.m, self.n = X.shape        
        # weight initialization        
        self.W = np.zeros( self.n )                
        self.X = X        
        self.Y = Y
          
        # gradient descent learning
                  
        for i in range( self.iterations ) :            
            self.update_weights() 
            
        return self
      
    # Helper function to update weights in gradient descent
      
    def update_weights( self ) :
        
            gradient = self.batch_gradient()
            self.W = self.W - (self.learning_rate * gradient)
    
    def batch_gradient(self):
        Gradient_Ein = 0
        for i in range(0,self.m + 1):
            Gradient_Ein  = Gradient_Ein + (-(self.X[i] * self.Y[i]) / ( 1 + np.exp( self.Y[i] * np.matmul( self.X[i],self.W) ) ))
            
        return Gradient_Ein/self.m
        
        
def main() :
    
    # Load the data
    images = pd.read_csv(r"C:\Users\sahru\Downloads\mnist_train_binary.csv")

    images_test = pd.read_csv(r"C:\Users\sahru\Downloads\mnist_test_binary.csv")

    X_test = images_test.iloc[:,1:]

    Y_test = images_test['label']

    X = images.iloc[:,1:]

    Y = images['label']
    
    X_train_preprocessed = images.iloc[:,1:]
    X_train_preprocessed = X_train_preprocessed/255
        
    X_train_preprocessed=np.reshape(np.ravel(X_train_preprocessed),(X_train_preprocessed.shape[0],28,28))
    
    
     
    
    Intensity = []
    Symmetry = []
    
    O = np.ones(12163)
    
    for i in range(0,len(X_train_preprocessed)):
        Intensity.append(X_train_preprocessed[i].mean())
        Symmetry.append(np.mean(abs(X_train_preprocessed[i] - np.flip(X_train_preprocessed[i]))))
        
    X_train_processed = np.column_stack((O,Intensity,Symmetry))
    
    # adding bias to the input
    np.random.seed(42)
    bias = np.random.uniform(low=0.0,high=1.0)
    X_train_processed[:,1:] = X_train_processed[:,1:]
    
    # X_tranformed = np.column_stack((O,(Intensity)**3,(Symmetry)**3,))
    



    #To plot the figures
    # pixels = np.array(X.iloc[3333,:], dtype='uint8')
    # pixels = pixels.reshape((28, 28))
    # plt.title('Label is {Y}'.format(Y=Y))
    # plt.imshow(pixels, cmap='gray')
    # plt.show()
    
    
    model = LogitRegression( learning_rate = 0.01, iterations = 1 )
      
    model.fit(X,Y)    
      
    # Prediction on test set
    Y_pred = model.predict( X_test )    
      
    # measure performance    
    correctly_classified = 0    
    correctly_classified1 = 0
      
    # counter    
    count = 0    
    for count in range( np.size( Y_pred ) ) :  
        
        if Y_test[count] == Y_pred[count] :            
            correctly_classified = correctly_classified + 1
              
        count = count + 1
          
    print( "Accuracy on test set by our model       :  ", ( 
      correctly_classified / count ) * 100 )
    
if __name__ == "__main__" :     
    main()
    
        