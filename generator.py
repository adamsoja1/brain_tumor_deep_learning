import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

def image_load_generator_x(path,batch_size):
    

    
    files = os.listdir(f'{path}/brain')
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
            x_train = x_train.reshape(x_train.shape[0],240,240)
 
            from keras.utils import normalize
            x_train = normalize(x_train,axis=1)
        
           
            yield(np.array(x_train))
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            
            
            
            
def image_load_generator_mask(path,batch_size):


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
            
            y_train = y_train.reshape(y_train.shape[0],240,240)
            
            
            from sklearn.preprocessing import LabelEncoder
            labelencoder = LabelEncoder()
            n,h,w = y_train.shape
            y_train_reshape = y_train.reshape(-1,1)
            y_train_reshape_encode = labelencoder.fit_transform(y_train_reshape)
            y_train_encoded = y_train_reshape_encode.reshape(n,h,w)


            y_train = np.expand_dims(y_train_encoded, axis=3)


            
            from keras.utils import to_categorical
            train_masks_cat = to_categorical(y_train,num_classes = 4)
            y_train_cat = train_masks_cat.reshape((y_train.shape[0],y_train.shape[1],y_train.shape[2],4))
            

            yield(np.array(y_train_cat))
            
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
                
                flipped = tf.image.flip_left_right(Y_train)
                y_train.append(flipped)
                
                rotated = tf.image.rot90(Y_train)
                y_train.append(rotated)
                
                rotated2 = tf.image.rot90(rotated)
                y_train.append(rotated2)
                
            

            
            
            y_train = np.array(y_train)
            
            y_train = y_train.reshape(y_train.shape[0],240,240)
            
            
            yield(np.array(y_train))
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            
            
            
            
            
def image_load_generator_mask_binary(path,batch_size):


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
                
                mask = np.array(Y_train)
            
                mask_1 = mask.copy()

                mask_1[mask_1 == 1] = 1
                mask_1[mask_1 == 2] = 0
                mask_1[mask_1 == 4] = 0


                mask_2 = mask.copy()
                mask_2[mask_2 == 1] = 0
                mask_2[mask_2 == 2] = 1
                mask_2[mask_2 == 4] = 0

                mask_4 = mask.copy()
                mask_4[mask_4 == 1] = 0
                mask_4[mask_4 == 2] = 0
                mask_4[mask_4 == 4] = 1


                mask = np.stack([mask_1, mask_2, mask_4])

                mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))
                

                y_train.append(mask)
            
            

            
            

            y_train = np.array(y_train)
   
            y_train = y_train.reshape(batch_size,3,240,240,1)
            
            yield(y_train)
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            

            