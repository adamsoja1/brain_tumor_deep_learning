import numpy as np 
import keras
import matplotlib.pyplot as plt
import random
import sys
import os
import albumentations as A
import cv2

from generator import image_load_generator_noaug,image_load_generator_mask_noaug
from generator_augment import image_load_generator_x,image_load_generator_mask


batch_size = 10
files = os.listdir('brains/train/mask')


test_generator = image_load_generator_x('brains/train',files,batch_size)
mask_generator = image_load_generator_mask('brains/train',files,batch_size)


datagen = zip(test_generator,mask_generator)



X_test,Y_test = next(datagen)



Y_test = np.argmax(Y_test,axis=-1)

for number in range(0,20):
    plt.imshow(X_test[number][:,:,2],cmap = 'gray')
    plt.show()
    plt.imshow(Y_test[number],cmap = 'gray')
    plt.show()