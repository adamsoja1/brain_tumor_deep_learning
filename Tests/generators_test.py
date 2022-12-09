import numpy as np 
import keras
import matplotlib.pyplot as plt
import random
import sys
import os
import albumentations as A
import cv2
sys.path.insert(1,r"C:\Users\adame\Desktop\Inzynierka")
from Data_preparation.generator import image_load_generator_noaug,image_load_generator_mask_noaug
from Data_preparation.generator_augment import image_load_generator_x,image_load_generator_mask
from New_Generator_Augment import Generator_augment
import random
np.random.seed(2021)

batch_size = 30
files = os.listdir('../brains/train/mask')
random.seed(0)
random.shuffle(files)
test_generator = image_load_generator_x('../brains/train',files,batch_size)
mask_generator = image_load_generator_mask('../brains/train',files,batch_size)


#datagen = zip(test_generator,mask_generator)



datagen=Generator_augment('../brains/train',files,batch_size)




X_test,Y_testt = next(datagen)



Y_test = np.argmax(Y_testt,axis=-1)

for number in range(0,20):
    plt.imshow(X_test[number][:,:,0],cmap = 'gray')
    plt.show()
    plt.imshow(Y_test[number],cmap = 'gray')
    plt.show()


