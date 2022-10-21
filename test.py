
import os
import shutil
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

class DataLoader:
    def __init__(self,path):
        self.path = path
        
    def get_list_files(self):
        
        return sorted(os.listdir(self.path))
    
    def get_random_images(self,channel,which_chan):
        random_brain_dir = random.choice(self.get_list_files())
        exact_path = self.path + '/' + random_brain_dir

        number_of_image = random_brain_dir[-5:]
        
        image_flair = nib.load(f'{exact_path}/BraTS2021_{number_of_image}_flair.nii.gz').get_fdata()
        image_seg = nib.load(f'{exact_path}/BraTS2021_{number_of_image}_seg.nii.gz').get_fdata()
        image_t1 = nib.load(f'{exact_path}/BraTS2021_{number_of_image}_t1.nii.gz').get_fdata()
        image_t1ce = nib.load(f'{exact_path}/BraTS2021_{number_of_image}_t1ce.nii.gz').get_fdata()
        image_t2 = nib.load(f'{exact_path}/BraTS2021_{number_of_image}_t2.nii.gz').get_fdata()
        
        which_channel =which_chan
        if which_channel == 1:
            mid_slice_flair = image_flair[channel,:,:]
            mid_slice_t1 = image_t1[channel,:,:]
            mid_slice_t1ce = image_t1ce[channel,:,:]
            mid_slice_t2 = image_t2[channel,:,:]
        
        elif which_channel == 2:
            
            mid_slice_flair = image_flair[:,channel,:]
            mid_slice_t1 = image_t1[:,channel,:]
            mid_slice_t1ce = image_t1ce[:,channel,:]
            mid_slice_t2 = image_t2[:,channel,:]
            
            
            
        elif which_channel == 3:
            mid_slice_flair = image_flair[:,:,channel]
            mid_slice_t1 = image_t1[:,:,channel]
            mid_slice_t1ce = image_t1ce[:,:,channel]
            mid_slice_t2 = image_t2[:,:,channel]
            
            
        
        fig,axes = plt.subplots(2, 3, figsize=(30, 25))    

        fig.suptitle(f'Mozg nr {number_of_image}', fontsize=40)
        mid_slice_seg = image_seg[:,:,channel]
        fig.patch.set_facecolor('white')

        axes[0,0].imshow(mid_slice_flair.T,cmap='gray', origin='lower')
        axes[0,0].set_title('Flair')
            
        axes[0,1].imshow(mid_slice_t1.T,cmap='gray', origin='lower')
        axes[0,1].set_title('t1')
            
        axes[0,2].imshow(mid_slice_t1ce.T,cmap='gray', origin='lower')
        axes[0,2].set_title('t1ce')
            
        axes[1,0].imshow(mid_slice_t2.T,cmap='gray', origin='lower')
        axes[1,0].set_title('t2')
            
        axes[1,1].imshow(mid_slice_seg.T,cmap='gray', origin='lower')
        axes[1,1].set_title('t1seg')
            
        axes[1,2].imshow(mid_slice_seg.T, origin='lower')
        axes[1,2].set_title('t1seg')
            
        
    def get_data_train(self):
        path_dirs = self.path
        files = self.get_list_files()[:1200]
        
        brain = []
        seg = []
  
        for file in files:
            image_seg = nib.load(f'{self.path}/{file}/BraTS2021_{file[-5:]}_seg.nii.gz').get_fdata()            
            
            mid_slice_seg = image_seg[:,:,75]
            
            seg.append(np.array(mid_slice_seg))
           
            
            
        for file in files:
            image_t1ce = nib.load(f'{self.path}/{file}/BraTS2021_{file[-5:]}_t1ce.nii.gz').get_fdata()
            mid_slice_t1ce = image_t1ce[:,:,75]
            brain.append(np.array(mid_slice_t1ce))
            
        return np.array(seg),np.array(brain)
            

            
load = DataLoader('RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021')


y_train,x_train = load.get_data_train()
x_train = np.array(x_train)
y_train = np.array(y_train)




from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n,h,w = y_train.shape
y_train_reshape = y_train.reshape(-1,1)
y_train_reshape_encode = labelencoder.fit_transform(y_train_reshape)
y_train_encoded = y_train_reshape_encode.reshape(n,h,w)
np.unique(y_train_encoded)

y_train = np.expand_dims(y_train_encoded, axis=3)

from keras.utils import normalize
x_train = np.expand_dims(x_train,axis=3)
x_train = normalize(x_train,axis=1)




from sklearn.model_selection import train_test_split
X1,X_test,y1,y_test = train_test_split(x_train,y_train,train_size = 0.8,test_size=0.2)
X_train,X_do_not,y_train,y_do_not = train_test_split(X1,y1,train_size = 0.8,test_size=0.2)




from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train,num_classes = 4)
y_train_cat = train_masks_cat.reshape((y_train.shape[0],y_train.shape[1],y_train.shape[2],4))

test_masks_cat = to_categorical(y_test,num_classes = 4)
y_test_cat = test_masks_cat.reshape((y_test.shape[0],y_test.shape[1],y_test.shape[2],4))



from sklearn.utils.class_weight import compute_class_weight
class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                 classes = np.unique(y_train_reshape_encode),
                                                 y = y_train_reshape_encode)


from keras.models import *
from keras.layers import *
from keras import initializers
from keras.optimizers import *
from keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D, UpSampling2D

import numpy as np 
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.losses import categorical_crossentropy
def unet(input_size = (240,240,1)):
    
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu' ,padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu' ,padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(4, 1,activation ='softmax',)(conv9)

    model = Model(inputs,  conv10)

    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    #model.summary()



    return model


model = unet(input_size = (240,240,1))




model.summary()

model.fit(X_train,y_train_cat,                    
                    batch_size = 4, 
                    verbose=1, 
                    epochs=20, 
                    validation_data=(X_test, y_test_cat), 
                    class_weight=class_weights,
                    shuffle=False)