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
files = os.listdir('./brains/train/brain')
random.shuffle(files)
random.seed(0)
random.shuffle(files)
test_generator = image_load_generator_noaug('./brains/train',files,batch_size)
mask_generator = image_load_generator_mask_noaug('./brains/train',files,batch_size)

#datagen = zip(test_generator,mask_generator)




datagen=Generator_augment('./brains/train',files,batch_size)




X_test = next(mask_generator)


X_test = np.argmax(X_test,axis=-1)

imgs = []
for number in range(0,30):
    plt.imshow(X_test[number],cmap = 'gray')
    plt.show()
    imgs.append(X_test[number])


fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(30, 35))

# Iteruj przez każdy obraz i wyświetl go w odpowiednim subplot
for ax, image in zip(axs.flat, imgs):
    ax.imshow(image, cmap='gray')





fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(30, 10))

# Iteruj przez każdy obraz i wyświetl go w odpowiednim subplot
for ax, image in zip(axs.flat, Y_test):
    ax.imshow(image, cmap='gray')

