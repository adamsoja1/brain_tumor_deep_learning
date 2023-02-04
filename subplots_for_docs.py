import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


imgs = []
pliki  = os.listdir('xxxx')

for i in pliki:
    img = plt.imread(f'xxxx/{i}')
    imgs.append(img)
    




fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(40, 45))

# Iteruj przez każdy obraz i wyświetl go w odpowiednim subplot
for ax, image in zip(axs.flat, imgs):
    ax.imshow(image)
    ax.axis('off')
# Pokaż subplots
plt.show()


path = 'RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
files = os.listdir('RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021')

clas1 = 0
clas2 = 0
clas3 = 0

for file in files:
    
    mask = nib.load(f'{path}/{file}/{file}_seg.nii.gz').get_fdata()
    for i in range(155):
        maska = mask[:,:,[i]]
        
        ones = np.sum(maska==1)
        clas1+=ones
        
        twos = np.sum(maska==2)
        clas2+=twos
        
        fours = np.sum(maska==4)
        clas3+=fours
            

daney = ['rdzeń nowotworu','opuchlizna','naciekający nowotwór']
danex = [clas1,clas2,clas3]
sns.barplot(y = daney,x=danex, palette = 'magma')








from Preparation import DataGenerators
import random
import os
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import random
import matplotlib.pyplot as plt
path_train = 'brains/train'
path_train = 'brains/train'
path_valid = 'brains/valid'
batch_size = 5

Generators = DataGenerators(path_train, path_valid, batch_size)
files = os.listdir('brains/train/brain')
val_datagen,steps_val = Generators.Validation_Datagen()


x,y = next(val_datagen)

images=[]
for i in range(4):
    images.append(y[0][:,:,i])

images.append(np.argmax(y[0],axis=-1))

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(30, 35))

# Iteruj przez każdy obraz i wyświetl go w odpowiednim subplot
for ax, image in zip(axs.flat, images):
    ax.imshow(image, cmap='gray')

# Pokaż subplots
plt.show()

plt.imshow(np.argmax(y[0],axis=-1),cmap='gray')


import pandas as pd

df = pd.read_csv('csv_files/testing for iou loss the best model.csv')
