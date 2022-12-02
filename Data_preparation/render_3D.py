import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.misc
import os
import imageio
from generator import image_load_generator_nib


path = 'RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
files = os.listdir(path)
batch_size = 5
contrast = 'seg'
generator = image_load_generator_nib(path,files,batch_size,contrast)
images = next(generator)



image  = images[0].get_fdata()
image = image/image.max() * 255
imgs = []

for i in range(155):
    im = np.array(image[:,:,i])
    im = im.astype('uint8')
    imgs.append(im)




    
imageio.mimsave('tumor1.gif',imgs,**{'duration':0.1})
