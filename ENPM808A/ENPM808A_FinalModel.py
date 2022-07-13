# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:12:33 2021

@author: sahru
"""

filenames = []

import os
import shutil

filenames_counter = 0

for subdir, dirs, files in os.walk(r'C:\Users\sahru\OneDrive\Desktop\New folder'):
    #print(files)
    for file in files:
        filenames.append(file)
        filenames_counter = filenames_counter + 1
print(filenames_counter)

filenames_test = []

import os
import shutil

filenames_counter = 0

for subdir, dirs, files in os.walk(r'C:\Users\sahru\OneDrive\Desktop\test_prac'):
    #print(files)
    for file in files:
        filenames_test.append(file)
        filenames_counter = filenames_counter + 1
print(filenames_test)

file_paths = []
for file_name in filenames :
    full_path = os.path.join(r'C:\Users\sahru\OneDrive\Desktop\vvv', file_name)
    file_paths.append(full_path) 
    
file_paths_test = []
for file_name in filenames_test :
    full_path = os.path.join(r'C:\Users\sahru\OneDrive\Desktop\test_prac', file_name)
    file_paths_test.append(full_path)

import pandas as pd
df = pd.concat((pd.read_csv(file,header = None).assign(filename=file) for file in file_paths),ignore_index = True)
dftest = pd.concat((pd.read_csv(file,header = None).assign(filename=file) for file in file_paths_test),ignore_index = True)

import numpy as np
import numpy.linalg as LA

def preprocess(df):
    laser1 = df.iloc[:,0:60].mean(axis=1)
    laser2 = df.iloc[:,60:120].mean(axis=1)
    laser3 = df.iloc[:,120:180].mean(axis=1)
    laser4 = df.iloc[:,180:240].mean(axis=1)
    laser5= df.iloc[:,240:300].mean(axis=1)
    laser6 = df.iloc[:,300:360].mean(axis=1)
    laser7 = df.iloc[:,360:420].mean(axis=1)
    laser8 = df.iloc[:,420:480].mean(axis=1)
    laser9 = df.iloc[:,480:540].mean(axis=1)
    laser10 = df.iloc[:,540:600].mean(axis=1)
    laser11 = df.iloc[:,600:660].mean(axis=1)
    laser12= df.iloc[:,660:720].mean(axis=1)
    laser13 = df.iloc[:,720:780].mean(axis=1)
    laser14= df.iloc[:,780:840].mean(axis=1)
    laser15= df.iloc[:,840:900].mean(axis=1)
    laser16= df.iloc[:,900:960].mean(axis=1)
    laser17= df.iloc[:,960:1020].mean(axis=1)
    laser18= df.iloc[:,1020:1080].mean(axis=1)
    relative_local_distance_x = (df.iloc[:,1080] - df.iloc[:,1088])
    relative_local_distance_y = (df.iloc[:,1081] - df.iloc[:,1089])
    local_orientation_x = (df.iloc[:,1082] - df.iloc[:,1090])
    local_orientation_y = (df.iloc[:,1083] - df.iloc[:,1091])
    relative_final_distance_x = (df.iloc[:,1084] - df.iloc[:,1088])
    relative_final_distance_y = (df.iloc[:,1085] - df.iloc[:,1089])
    final_orientation_x = (df.iloc[:,1086] - df.iloc[:,1090])
    final_orientation_y = (df.iloc[:,1087] - df.iloc[:,1091])
    rest = (df.iloc[:,1092:1094])
    train = pd.concat([laser1,laser2,laser3,laser4,laser5,laser6,laser7,laser8,laser9,laser10,laser11,laser12,laser13,laser14,laser15,laser16,laser17,laser18,relative_local_distance_x,relative_local_distance_y,local_orientation_x,local_orientation_y,relative_final_distance_x,relative_final_distance_y,final_orientation_x,final_orientation_y,rest],axis=1,join='inner')
    return train

train = preprocess(df)
test = preprocess(dftest)

X_train = train.iloc[:,0:26]
Y_train = train.iloc[:,26:28]

X_test = test.iloc[:,0:26]
Y_test = test.iloc[:,26:28]

import tensorflow as tf
from tensorflow import keras

from keras.regularizers import l1
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(32, activation='relu',input_shape = (26,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4,activation='relu'),
    keras.layers.Dense(2),
])


model.compile(optimizer = 'adam',loss = 'mean_absolute_error',metrics=['accuracy'])

history = model.fit(X_train,Y_train,epochs=100,batch_size=128)

p1 = model.predict(X_train)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_train, p1))

p = model.predict(X_test)
print(mean_absolute_error(Y_test, p))

from sklearn.metrics import r2_score
c = r2_score(Y_train, p1)
print(c)

from sklearn.metrics import r2_score
c = r2_score(Y_test, p)
print(c)

import matplotlib.pyplot as plt
loss=history.history['loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'y',label='Training Loss')
plt.title('Training loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show



























