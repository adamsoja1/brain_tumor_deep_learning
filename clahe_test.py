import numpy as np 
import keras
import matplotlib.pyplot as plt
import random
import sys
import os

import cv2

from generator import image_load_generator_x,image_load_generator_mask,image_load_generator_mask_nocat,image_load_generator_mask_binary


batch_size = 10
files = os.listdir('brains/train/mask')



test_generator = image_load_generator_x('brains/train',files,batch_size)
mask_generator = image_load_generator_mask('brains/train',files,batch_size)



X_test = next(test_generator)
Y_test = next(mask_generator)
Y_test = np.argmax(Y_test,axis=3)

img = X_test[2] * 255
img = img.astype('uint8')

clahe = cv2.createCLAHE(clipLimit = 5)
            
imag_flair = clahe.apply(img) + 5

plt.imshow(img,cmap='gray')
plt.show()
plt.imshow(imag_flair,cmap = 'gray')
plt.show()
              

import os
import numpy as np
def count_classes(path):
    
    files  = os.listdir(f'{path}/mask')
    zeros = 0
    first = 0
    second = 0
    fourth = 0
    
    for file in files:
        mask = np.load(f'{path}/mask/{file}')
        mask = np.argmax(mask,axis=-1)
        calculate_zeros = np.sum(mask==0)
        zeros += calculate_zeros
        
        calculate_ones = np.sum(mask==1)
        first += calculate_ones
    
        calculate_two = np.sum(mask==2)
        second +=calculate_two
        
        calculate_fours = np.sum(mask==4)
        fourth +=calculate_fours

        

        
    return zeros,first,second,fourth


zero,first,second,fourth = count_classes('brains/train')