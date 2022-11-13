from generator import image_load_generator_nib
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random
import cv2

path = 'RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
files = os.listdir(path)

batch_size = 20
contrast = '_flair'
generator = image_load_generator_nib(path,files,batch_size,contrast)


images = next(generator)

images = (images/images.max()) * 255

number = random.randint(0,batch_size-1)

img = images[5]
            
img = img.astype('uint8')
            
clahe = cv2.createCLAHE(clipLimit = 2)

imag_flair = clahe.apply(img) 
            
imag_flair = imag_flair.astype('float32')

imagex = images[5]



imagex = imagex[40:200 ,40:200]

plt.imshow(imagex,cmap='gray')
plt.show()
plt.imshow(imag_flair,cmap='gray')







