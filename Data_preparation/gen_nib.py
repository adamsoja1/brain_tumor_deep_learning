from generator import image_load_generator_nib
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random
import cv2
import nibabel as nib
path = '../RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
files = os.listdir(path)

batch_size = 20
contrast = 't1ce'
generator = image_load_generator_nib(path,files,batch_size,contrast)


images = next(generator)



number = random.randint(0,batch_size-1)



            


imagex1 = images[5]

imagex = imagex1[:,:,[70]]

imagex = imagex[40:200 ,40:200]

plt.imshow(imagex,cmap='gray')
plt.show()
plt.imshow(imag_flair,cmap='gray')







