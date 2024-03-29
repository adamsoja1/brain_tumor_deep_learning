import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import normalize



def image_load_generator_x(path,files,batch_size):
    
    
    

    L = len(files)
    while True:
        batch_start = 0
        batch_size_end = batch_size
        while batch_start < L:
            limit = min(batch_size_end,L)
            
            files_batched = files[batch_start:limit]
            
            #loading data
            x_train = []
            
            for file in files_batched:
                
                X_train = np.load(f'{path}/brain/{file}')
                x_train.append(X_train)
            
                

                
                
            
                
            x_train = np.array(x_train)
            x_train = x_train/255
            x_train = x_train.astype('float32')
            yield(x_train)
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            
            
            
            
def image_load_generator_mask(path,files,batch_size):



    L = len(files)
    while True:
        batch_start = 0
        batch_size_end = batch_size
        while batch_start < L:
            limit = min(batch_size_end,L)
            
            files_batched = files[batch_start:limit]
            
            #loading data

            y_train = []
            
            for file in files_batched:
                Y_train = np.load(f'{path}/mask/{file}')
                Y_train = Y_train.reshape(160,160,4)
                y_train.append(Y_train)
                
                
            
            
            

            
            
            y_train = np.array(y_train)
            
            
            

            yield(y_train)
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            

            
            
            
def image_load_generator_mask_nocat(path,batch_size):


    files = os.listdir(f'{path}/mask')
    L = len(files)
    while True:
        batch_start = 0
        batch_size_end = batch_size
        while batch_start < L:
            limit = min(batch_size_end,L)
            
            files_batched = files[batch_start:limit]
            
            #loading data

            y_train = []
            
            for file in files_batched:
                Y_train = np.load(f'{path}/mask/{file}')
                y_train.append(Y_train)
                

            

            
            
            y_train = np.array(y_train)
            
            y_train = y_train.reshape(4*batch_size,240,240)
            
            
            yield(np.array(y_train))
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            
            
            
            
import nibabel as nib            
def image_load_generator_nib(path,files,batch_size):
    
    
    

    L = len(files)
    while True:
        batch_start = 0
        batch_size_end = batch_size
        while batch_start < L:
            limit = min(batch_size_end,L)
            
            files_batched = files[batch_start:limit]
            
            #loading data
            x_train = []
            
            for file in files_batched:
                
                nib = nib.load()
                x_train.append(X_train)
            
                

                
                
            
                
            x_train = np.array(x_train)
            x_train[x_train==9] = 0
            x_train = x_train/255
            x_train = x_train.astype('float32')
            yield(x_train)
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            
            