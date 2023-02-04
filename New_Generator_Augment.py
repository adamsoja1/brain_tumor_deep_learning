import tensorflow as tf
import numpy as np
import os
from keras.utils import normalize
import albumentations as a
import random
from keras.utils import to_categorical


transform_all = a.Compose([
    a.augmentations.geometric.transforms.Affine(p=0.7,rotate=3),
    a.augmentations.geometric.transforms.Affine(p=0.7,rotate=5),
    a.augmentations.geometric.transforms.Affine(p=0.7,rotate=-7),
    a.augmentations.geometric.transforms.Affine(p=0.7,rotate=-5),    
    a.augmentations.geometric.transforms.Affine(p=0.7,rotate=-3),
    a.augmentations.geometric.transforms.Affine(p=0.7,rotate =7),
    a.augmentations.geometric.transforms.Affine(p=0.7,rotate =10),
    a.augmentations.geometric.transforms.Affine(p=0.7,rotate =-10),

    
    ])



def Generator_augment(path,files,batch_size):
    
    L = len(files)
    while True:
        batch_start = 0
        batch_size_end = batch_size
        while batch_start < L:
            limit = min(batch_size_end,L)
            
            files_batched = files[batch_start:limit]
            
            #loading data
            x_train = []
            y_train = []
            for file in files_batched:
       

                X_train = np.load(f'{path}/brain/{file}')
                Y_train = np.load(f'{path}/mask/{file}')

                transformed = transform_all(image = X_train, mask = np.argmax(Y_train,axis=-1))
                
                brain = transformed['image']
                mask = transformed['mask']
                
                
                #x_train.append(X_train)
                x_train.append(brain)                
             
                #y_train.append(np.argmax(Y_train,axis=-1))
                y_train.append(mask)

                
                
            
            l = len(x_train)    
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            y_train = to_categorical(y_train,num_classes = 4)
            
            
            x_train = x_train.reshape(l,160,160,4)
            y_train = y_train.reshape(l,160,160,4)
            x_train = x_train.astype('float32')
            y_train = y_train.astype('float32')
            
            yield(x_train,y_train)
            
            batch_start +=batch_size
            batch_size_end +=batch_size