import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import normalize



def image_load_generator_noaug(path,files,batch_size):
    
    
    

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
                X_train = X_train.reshape(1,160,160,4)
                x_train.append(np.array(X_train))
            
                

                
                
            
            l = len(x_train)    
            x_train = np.array(x_train)
            x_train = x_train/255
            x_train = x_train.reshape(l,160,160,4)
            yield(x_train)
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            
            
            
            
def image_load_generator_mask_noaug(path,files,batch_size):



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
            

            
            
            

            
            
import nibabel as nib            
def image_load_generator_nib(path,files,batch_size,contrast):
    import nibabel as nib            

    
    

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
                
                nib = nib.load(f'{path}/{file}/{file}_{contrast}.nii.gz')
                x_train.append(nib)
            
                

                
                
            
                
            x_train = np.array(x_train)
            yield(x_train)
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            
            
            
