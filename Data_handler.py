from sklearn.preprocessing import LabelEncoder
from keras.utils import normalize
import os
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import random
import time
start = time.time()

main_folder = 'RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
base_dir = 'brains'
def Directories(main_folder,base_dir):

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        
    train_dir  = os.path.join(base_dir,'train')
    test_dir  = os.path.join(base_dir,'test')
    valid_dir  = os.path.join(base_dir,'valid')
    
    for directory in (train_dir, valid_dir, test_dir):
        if not os.path.exists(directory):
            os.mkdir(directory)
            
    train_brain = os.path.join(train_dir, 'brain')
    train_mask = os.path.join(train_dir, 'mask')
    
    test_brain = os.path.join(test_dir, 'brain')
    test_mask = os.path.join(test_dir, 'mask')
    
    valid_brain = os.path.join(valid_dir, 'brain')
    valid_mask = os.path.join(valid_dir, 'mask')
    
    
    dirs = [train_brain,train_mask, test_brain, test_mask, valid_brain, valid_mask ]
    for directory in dirs:
        if not os.path.exists(directory):
            os.mkdir(directory)
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.mkdir(directory)
            
Directories(main_folder,base_dir)        
def to_categorical_masks(mask):
            
    y_train = mask

    labelencoder = LabelEncoder()
    n,h,w = y_train.shape
    y_train_reshape = y_train.reshape(-1,1)
    y_train_reshape_encode = labelencoder.fit_transform(y_train_reshape)
    y_train_encoded = y_train_reshape_encode.reshape(n,h,w)


    y_train = np.expand_dims(y_train_encoded, axis=3)



    train_masks_cat = to_categorical(y_train,num_classes = 4)

    y_train_cat = train_masks_cat.reshape((y_train.shape[0],y_train.shape[1],y_train.shape[2],4))
                
            
    return y_train_cat




def normalization(images):
    import cv2
    scaler = MinMaxScaler()
    

    
    images[0]=scaler.fit_transform(images[0].reshape(-1,images[0].shape[-1])).reshape(images[0].shape)
    images[1]=scaler.fit_transform(images[1].reshape(-1,images[1].shape[-1])).reshape(images[1].shape)
    images[2]=scaler.fit_transform(images[2].reshape(-1,images[2].shape[-1])).reshape(images[2].shape)
    images[3]=scaler.fit_transform(images[3].reshape(-1, images[3].shape[-1])).reshape(images[3].shape)
    
    
    

        
    return images[0],images[1],images[2],images[3]
        
        
        




def data_split(path,train_size,val_size,test_size):

    
    
    #normalizacja przy uzyciu clahe
    
    files  = os.listdir(path)
    random.Random(0).shuffle(files)


    train_dataset,split_dataset = train_test_split(files,test_size = 1- train_size,random_state = 0)
    test_dataset,valid_dataset = train_test_split(split_dataset,train_size = 0.25,random_state = 0)
    
    
    for file in train_dataset:
        
        image_t1ce = nib.load(f'{path}/{file}/{file}_t1ce.nii.gz').get_fdata()
        image_flair = nib.load(f'{path}/{file}/{file}_flair.nii.gz').get_fdata()
        image_t1 = nib.load(f'{path}/{file}/{file}_t1.nii.gz').get_fdata()
        image_t2 = nib.load(f'{path}/{file}/{file}_t2.nii.gz').get_fdata()

        image_seg = nib.load(f'{path}/{file}/{file}_seg.nii.gz').get_fdata()
        
        
        
        
        img_t2 = image_t2[:,:,70:110]
        img_flair = image_flair[:,:,70:110]
        img_t1 = image_t1[:,:,70:110]
        img_t1ce = image_t1ce[:,:,70:110]
        
        
        
        masks = image_seg[:,:,70:110]
        
        
   
        
        for image in range(0,masks.shape[2]):
            
            
                        
            imag_t1 = np.array(img_t1[:,:,[image]])
            imag_t1ce = np.array(img_t1ce[:,:,[image]])
            imag_flair = np.array(img_flair[:,:,[image]])
            imag_t2 = np.array(img_t2[:,:,[image]])
            mask = np.array(masks[:,:,[image]])
            
            imag_flair = imag_flair[40:200 ,40:200]
            imag_t1ce = imag_t1ce[40:200 ,40:200]
            imag_t1 = imag_t1[40:200 ,40:200]
            imag_t2 = imag_t2[40:200 ,40:200]

            
            images = [imag_flair,imag_t1ce,imag_t1,imag_t2]

            imag_flair,imag_t1ce,imag_t1,imag_t2 = normalization(images)
        
 
            imag_t2 = imag_t2.reshape(160,160)
            imag_t1 = imag_t1.reshape(160,160)             
            imag_t1ce = imag_t1ce.reshape(160,160)
            imag_flair = imag_flair.reshape(160,160)
            
            

            imag = np.stack([imag_t1,imag_t1ce,imag_flair,imag_t2],axis=2)
            

            mask = mask[40:200 ,40:200]
   
                

                
            y_train_cat = to_categorical_masks(mask)
    
            

            
            
            
            np.save(f'brains/train/mask/{file}{image}',y_train_cat.astype(np.uint8))
            np.save(f'brains/train/brain/{file}{image}',imag.astype(np.float32))
            

            
            
    
    for file in test_dataset:
        
        image_t1ce = nib.load(f'{path}/{file}/{file}_t1ce.nii.gz').get_fdata()
        image_flair = nib.load(f'{path}/{file}/{file}_flair.nii.gz').get_fdata()
        image_t1 = nib.load(f'{path}/{file}/{file}_t1.nii.gz').get_fdata()
        image_t2 = nib.load(f'{path}/{file}/{file}_t2.nii.gz').get_fdata()

        image_seg = nib.load(f'{path}/{file}/{file}_seg.nii.gz').get_fdata()
        
        
        
        
        img_t2 = image_t2[:,:,70:110]
        img_flair = image_flair[:,:,70:110]
        img_t1 = image_t1[:,:,70:110]
        img_t1ce = image_t1ce[:,:,70:110]
        
        
        
        masks = image_seg[:,:,70:110]
       
        for image in range(0,masks.shape[2]):
            
            
                        
            imag_t1 = np.array(img_t1[:,:,[image]])
            imag_t1ce = np.array(img_t1ce[:,:,[image]])
            imag_flair = np.array(img_flair[:,:,[image]])
            imag_t2 = np.array(img_t2[:,:,[image]])
            mask = np.array(masks[:,:,[image]])

            
  

    
            imag_flair = imag_flair[40:200 ,40:200]
            imag_t1ce = imag_t1ce[40:200 ,40:200]
            imag_t1 = imag_t1[40:200 ,40:200]
            imag_t2 = imag_t2[40:200 ,40:200]
            mask = mask[40:200 ,40:200]
            
            images = [imag_flair,imag_t1ce,imag_t1,imag_t2]

            imag_flair,imag_t1ce,imag_t1,imag_t2 = normalization(images)
 
            imag_t2 = imag_t2.reshape(160,160)
            imag_t1 = imag_t1.reshape(160,160)
            imag_t1ce = imag_t1ce.reshape(160,160)                       
            imag_flair = imag_flair.reshape(160,160)
            
            imag = np.stack([imag_t1,imag_t1ce,imag_flair,imag_t2],axis=2)
            
            

                
            y_train_cat = to_categorical_masks(mask)
    
            

            np.save(f'brains/test/mask/{file}{image}',y_train_cat.astype(np.uint8))
            np.save(f'brains/test/brain/{file}{image}',imag.astype(np.float32))
            

            
        
    for file in valid_dataset:
        
        image_t1ce = nib.load(f'{path}/{file}/{file}_t1ce.nii.gz').get_fdata()
        image_flair = nib.load(f'{path}/{file}/{file}_flair.nii.gz').get_fdata()
        image_t1 = nib.load(f'{path}/{file}/{file}_t1.nii.gz').get_fdata()
        image_t2 = nib.load(f'{path}/{file}/{file}_t2.nii.gz').get_fdata()

        image_seg = nib.load(f'{path}/{file}/{file}_seg.nii.gz').get_fdata()

        img_t2 = image_t2[:,:,70:110]
        img_flair = image_flair[:,:,70:110]
        img_t1 = image_t1[:,:,70:110]
        img_t1ce = image_t1ce[:,:,70:110]
        
        
        
        masks = image_seg[:,:,70:110]
         
        
        for image in range(0,masks.shape[2]):
            
            
                        
            imag_t1 = np.array(img_t1[:,:,[image]])
            imag_t1ce = np.array(img_t1ce[:,:,[image]])
            imag_flair = np.array(img_flair[:,:,[image]])
            imag_t2 = np.array(img_t2[:,:,[image]])
            mask = np.array(masks[:,:,[image]])

       
            imag_flair = imag_flair[40:200 ,40:200]
            imag_t1ce = imag_t1ce[40:200 ,40:200]
            imag_t1 = imag_t1[40:200 ,40:200]
            imag_t2 = imag_t2[40:200 ,40:200]
            mask = mask[40:200 ,40:200]
            
            
            images = [imag_flair,imag_t1ce,imag_t1,imag_t2]

            imag_flair,imag_t1ce,imag_t1,imag_t2 = normalization(images)
            
             
        
 
            imag_t2 = imag_t2.reshape(160,160)
            imag_t1 = imag_t1.reshape(160,160)
            imag_t1ce = imag_t1ce.reshape(160,160)         
            imag_flair = imag_flair.reshape(160,160)
            imag = np.stack([imag_t1,imag_t1ce,imag_flair,imag_t2],axis=2)
                      
   
                
            y_train_cat = to_categorical_masks(mask)
    
   
            np.save(f'brains/valid/mask/{file}{image}',y_train_cat.astype(np.uint8))
            np.save(f'brains/valid/brain/{file}{image}',imag.astype(np.float32))
            
    return train_dataset,valid_dataset,test_dataset


#x,y,z = data_split(main_folder,0.7,0.2,0.10)    



def data_split_if_tumors(path,train_size,val_size,test_size):

    
    
    #normalizacja przy uzyciu clahe
    
    files  = os.listdir(path)
    random.Random(0).shuffle(files)

    train_dataset,split_dataset = train_test_split(files,test_size = 1- train_size,random_state = 0)
    
    test_dataset,valid_dataset = train_test_split(split_dataset,train_size = 0.25,random_state = 0)
    
    
    for file in train_dataset:
        
        image_t1ce = nib.load(f'{path}/{file}/{file}_t1ce.nii.gz').get_fdata()
        image_flair = nib.load(f'{path}/{file}/{file}_flair.nii.gz').get_fdata()
        image_t1 = nib.load(f'{path}/{file}/{file}_t1.nii.gz').get_fdata()
        image_t2 = nib.load(f'{path}/{file}/{file}_t2.nii.gz').get_fdata()

        image_seg = nib.load(f'{path}/{file}/{file}_seg.nii.gz').get_fdata()
        
        
        
        
        img_t2 = image_t2[:,:,0:150]
        img_flair = image_flair[:,:,0:150]
        img_t1 = image_t1[:,:,0:150]
        img_t1ce = image_t1ce[:,:,0:150]
        
        
        
        masks = image_seg[:,:,0:150]
        
   
        
        for image in range(0,masks.shape[2]):
            mask = np.array(masks[:,:,[image]])
            if len(np.unique(mask)) > 1:
                
                
            
                imag_t1 = np.array(img_t1[:,:,[image]])
                imag_t1ce = np.array(img_t1ce[:,:,[image]])
                imag_flair = np.array(img_flair[:,:,[image]])
                imag_t2 = np.array(img_t2[:,:,[image]])
                mask = np.array(masks[:,:,[image]])
                

                

                
                
                
                
                
                imag_flair = imag_flair[40:200 ,40:200]
                imag_t1ce = imag_t1ce[40:200 ,40:200]
                imag_t1 = imag_t1[40:200 ,40:200]
                imag_t2 = imag_t2[40:200 ,40:200]
                
               


                
                images = [imag_flair,imag_t1ce,imag_t1,imag_t2]
    
                imag_flair,imag_t1ce,imag_t1,imag_t2 = normalization(images)
            
     
                imag_t2 = imag_t2.reshape(160,160)
                imag_t1 = imag_t1.reshape(160,160)             
                imag_t1ce = imag_t1ce.reshape(160,160)
                imag_flair = imag_flair.reshape(160,160)
                
                
    
                imag = np.stack([imag_t1ce,imag_flair,imag_t2,imag_t1],axis=-1)
                
    
                mask = mask[40:200 ,40:200]
       
                    
    
                    
                y_train_cat = to_categorical_masks(mask)
    
    
    
    
    
                np.save(f'brains/train/mask/{file}{image}',y_train_cat.astype(np.uint8))
                np.save(f'brains/train/brain/{file}{image}',imag.astype(np.float32))
              

            
            
    
    for file in test_dataset:
        
        
        image_t1ce = nib.load(f'{path}/{file}/{file}_t1ce.nii.gz').get_fdata()
        image_flair = nib.load(f'{path}/{file}/{file}_flair.nii.gz').get_fdata()
        image_t1 = nib.load(f'{path}/{file}/{file}_t1.nii.gz').get_fdata()
        image_t2 = nib.load(f'{path}/{file}/{file}_t2.nii.gz').get_fdata()

        image_seg = nib.load(f'{path}/{file}/{file}_seg.nii.gz').get_fdata()
        
        
        
        
        img_t2 = image_t2[:,:,0:150]
        img_flair = image_flair[:,:,0:150]
        img_t1 = image_t1[:,:,0:150]
        img_t1ce = image_t1ce[:,:,0:150]
        
        
        
        masks = image_seg[:,:,0:150]
        
   
        
        for image in range(0,masks.shape[2]):
            mask = np.array(masks[:,:,[image]])
            if len(np.unique(mask)) > 1:
                
                
            
                        
                imag_t1 = np.array(img_t1[:,:,[image]])
                imag_t1ce = np.array(img_t1ce[:,:,[image]])
                imag_flair = np.array(img_flair[:,:,[image]])
                imag_t2 = np.array(img_t2[:,:,[image]])
                mask = np.array(masks[:,:,[image]])
                


                
                
                imag_flair = imag_flair[40:200 ,40:200]
                imag_t1ce = imag_t1ce[40:200 ,40:200]
                imag_t1 = imag_t1[40:200 ,40:200]
                imag_t2 = imag_t2[40:200 ,40:200]
    

                
                images = [imag_flair,imag_t1ce,imag_t1,imag_t2]
    
                imag_flair,imag_t1ce,imag_t1,imag_t2 = normalization(images)
            
     
                imag_t2 = imag_t2.reshape(160,160)
                imag_t1 = imag_t1.reshape(160,160)             
                imag_t1ce = imag_t1ce.reshape(160,160)
                imag_flair = imag_flair.reshape(160,160)
                
                
    
                imag = np.stack([imag_t1ce,imag_flair,imag_t2,imag_t1],axis=-1)
                
    
                mask = mask[40:200 ,40:200]
       
                    
    
                    
                y_train_cat = to_categorical_masks(mask)
    
    
    
    
    
    
                np.save(f'brains/test/mask/{file}{image}',y_train_cat.astype(np.uint8))
                np.save(f'brains/test/brain/{file}{image}',imag.astype(np.float32))
              
            

            
        
    for file in valid_dataset:
        
        
        image_t1ce = nib.load(f'{path}/{file}/{file}_t1ce.nii.gz').get_fdata()
        image_flair = nib.load(f'{path}/{file}/{file}_flair.nii.gz').get_fdata()
        image_t1 = nib.load(f'{path}/{file}/{file}_t1.nii.gz').get_fdata()
        image_t2 = nib.load(f'{path}/{file}/{file}_t2.nii.gz').get_fdata()

        image_seg = nib.load(f'{path}/{file}/{file}_seg.nii.gz').get_fdata()
        
        
        
        
        img_t2 = image_t2[:,:,0:150]
        img_flair = image_flair[:,:,0:150]
        img_t1 = image_t1[:,:,0:150]
        img_t1ce = image_t1ce[:,:,0:150]
        
        
        
        masks = image_seg[:,:,0:150]
        
        
        
   
        
        for image in range(0,masks.shape[2]):
            mask = np.array(masks[:,:,[image]])
            if len(np.unique(mask)) > 1:
                
                
            
                        
                imag_t1 = np.array(img_t1[:,:,[image]])
                imag_t1ce = np.array(img_t1ce[:,:,[image]])
                imag_flair = np.array(img_flair[:,:,[image]])
                imag_t2 = np.array(img_t2[:,:,[image]])
                mask = np.array(masks[:,:,[image]])
                
                
                
                
                imag_flair = imag_flair[40:200 ,40:200]
                imag_t1ce = imag_t1ce[40:200 ,40:200]
                imag_t1 = imag_t1[40:200 ,40:200]
                imag_t2 = imag_t2[40:200 ,40:200]
                

                
    
                
                images = [imag_flair,imag_t1ce,imag_t1,imag_t2]
    
                imag_flair,imag_t1ce,imag_t1,imag_t2 = normalization(images)
            
     
                imag_t2 = imag_t2.reshape(160,160)
                imag_t1 = imag_t1.reshape(160,160)             
                imag_t1ce = imag_t1ce.reshape(160,160)
                imag_flair = imag_flair.reshape(160,160)
                
                
    
                imag = np.stack([imag_t1ce,imag_flair,imag_t2,imag_t1],axis=-1)
                
    
                mask = mask[40:200 ,40:200]
       
                    
    
                    
                y_train_cat = to_categorical_masks(mask)
    
    
    
    
    
                np.save(f'brains/valid/mask/{file}{image}',y_train_cat.astype(np.uint8))
                np.save(f'brains/valid/brain/{file}{image}',imag.astype(np.float32))
              
            

    return train_dataset,valid_dataset,test_dataset

x,y,z = data_split_if_tumors(main_folder,0.7,0.25,0.1)    

with open('test_data.txt', 'w') as f:
    for line in z:
        f.write(line)
        f.write('\n')
        
with open('valid_data.txt', 'w') as f:
    for line in z:
        f.write(line)
        f.write('\n')
        
with open('train_data.txt', 'w') as f:
    for line in z:
        f.write(line)
        f.write('\n')
        
stop = time.time()

tm = stop - start











