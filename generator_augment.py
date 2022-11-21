import tensorflow as tf
import numpy as np
import os
from keras.utils import normalize
import albumentations as a
import random

random.seed(0)

transform_vertflip = a.augmentations.geometric.transforms.VerticalFlip(p=1)
transform_horplif =a.augmentations.geometric.transforms.Affine(p=1, rotate=270)
transform = a.augmentations.geometric.transforms.Affine(p=1,scale=1.4)
rotate = a.augmentations.geometric.transforms.Affine(p=1,scale=1.4)
rotate2 = a.HorizontalFlip(p=1)
rotate3 = a.augmentations.geometric.transforms.Affine(p=1,scale=1.2,rotate =90 )
rotate4 = a.augmentations.geometric.transforms.Affine(p=1,scale=1.3,translate_px=30)
rotate5 = a.augmentations.geometric.transforms.Affine(p=1,scale=1.3,translate_px=-30)


random.seed(0)
def image_load_generator_x(path,files,batch_size):
    
    random.seed(0)

    

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
                random.seed(0)

                X_train = np.load(f'{path}/brain/{file}')
                X_train = X_train.reshape(160,160,4)
                brain1 = transform_vertflip(image = X_train)['image']
                brain2 = transform_horplif(image = X_train)['image']
                brain3 = transform(image = X_train)['image']
                brain4 = rotate(image = X_train)['image']
                brain5 = rotate2(image = X_train)['image']
                brain6 = rotate3(image = X_train)['image']
                brain7 = rotate4(image = X_train)['image']
                brain8 = rotate5(image = X_train)['image']
                
                x_train.append(X_train)
                x_train.append(brain1)
                x_train.append(brain2)
                x_train.append(brain3)
                x_train.append(brain4)
                x_train.append(brain5)
                x_train.append(brain6)
                x_train.append(brain7)
                x_train.append(brain8)
                
                

                
                
            
            l = len(x_train)    
            x_train = np.array(x_train)
            x_train = x_train/255
            x_train = x_train.reshape(l,160,160,4)
            yield(x_train)
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            
            
            
            
def image_load_generator_mask(path,files,batch_size):

    random.seed(0)


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
                random.seed(0)

                Y_train = np.load(f'{path}/mask/{file}')
                Y_train = Y_train.reshape(160,160,4)
                
                mask1 = transform_vertflip(image = Y_train)['image']
                mask2 = transform_horplif(image = Y_train)['image']
                mask3 = transform(image = Y_train)['image']
                mask4 = rotate(image = Y_train)['image']
                mask5 = rotate2(image = Y_train)['image']
                mask6 = rotate3(image = Y_train)['image']
                mask7 = rotate4(image = Y_train)['image']
                mask8 = rotate5(image = Y_train)['image']

                
                y_train.append(Y_train)
                y_train.append(mask1)
                y_train.append(mask2)
                y_train.append(mask3)
                y_train.append(mask4)
                y_train.append(mask5)
                y_train.append(mask6)
                y_train.append(mask7)
                y_train.append(mask8)
                
            
            
            

            
            v = len(y_train)
            y_train = np.array(y_train)
            y_train = y_train.reshape(v,160,160,4)
            
            

            yield(y_train)
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            