import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import nibabel as nib

def data_split(path,train_size,val_size,test_size):  
    #normalizacja przy uzyciu clahe
    files  = os.listdir(path)
    train_dataset,split_dataset = train_test_split(files,test_size = 1- train_size,random_state = 0)
    test_dataset,valid_dataset = train_test_split(split_dataset,train_size = 0.25,random_state = 0)
    for file in train_dataset:
        
        image_t1ce = nib.load(f'{path}/{file}/{file}_t1ce.nii.gz').get_fdata()
        image_flair = nib.load(f'{path}/{file}/{file}_flair.nii.gz').get_fdata()
        image_t1 = nib.load(f'{path}/{file}/{file}_t1.nii.gz').get_fdata()
        image_t2 = nib.load(f'{path}/{file}/{file}_t2.nii.gz').get_fdata()

        image_seg = nib.load(f'{path}/{file}/{file}_seg.nii.gz').get_fdata()
        
        
        
        
        img_t2 = image_t2[:,:,1:154]
        img_flair = image_flair[:,:,1:154]
        img_t1 = image_t1[:,:,1:154]
        img_t1ce = image_t1ce[:,:,1:154]
        
        
        
        masks = image_seg[:,:,1:154]
        
        
   
        
        for image in range(1,154):
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









                imag_t1 = (imag_t1/imag_t1.max()) * 255
                imag_t1ce = (imag_t1ce/imag_t1ce.max()) * 255
                imag_flair = (imag_flair/imag_flair.max()) * 255
                imag_t2 = (imag_t2/imag_t2.max()) * 255



                imag_t2 = imag_t2.reshape(160,160)



                imag_t1 = imag_t1.reshape(160,160)





                imag_t1ce = imag_t1ce.reshape(160,160)


                imag_flair = imag_flair.reshape(160,160)



                imag = np.stack([imag_t1,imag_t1ce,imag_flair,imag_t2],axis=2)

















                mask = mask[40:200 ,40:200]


                y_train = mask

                labelencoder = LabelEncoder()
                n,h,w = y_train.shape
                y_train_reshape = y_train.reshape(-1,1)
                y_train_reshape_encode = labelencoder.fit_transform(y_train_reshape)
                y_train_encoded = y_train_reshape_encode.reshape(n,h,w)


                y_train = np.expand_dims(y_train_encoded, axis=3)



                train_masks_cat = to_categorical(y_train,num_classes = 4)

                y_train_cat = train_masks_cat.reshape((y_train.shape[0],y_train.shape[1],y_train.shape[2],4))






                np.save(f'brains/train/mask/{file}{image}',y_train_cat)
                np.save(f'brains/train/brain/{file}{image}',imag)
            

            
            
    
    for file in test_dataset:
        
        
        image_t1ce = nib.load(f'{path}/{file}/{file}_t1ce.nii.gz').get_fdata()
        image_flair = nib.load(f'{path}/{file}/{file}_flair.nii.gz').get_fdata()
        image_t1 = nib.load(f'{path}/{file}/{file}_t1.nii.gz').get_fdata()
        image_t2 = nib.load(f'{path}/{file}/{file}_t2.nii.gz').get_fdata()

        image_seg = nib.load(f'{path}/{file}/{file}_seg.nii.gz').get_fdata()
        
        
        
        
        img_t2 = image_t2[:,:,1:154]
        img_flair = image_flair[:,:,1:154]
        img_t1 = image_t1[:,:,1:154]
        img_t1ce = image_t1ce[:,:,1:154]
        
        
        
        masks = image_seg[:,:,1:154]
        
        
   
        
        for image in range(0,154):
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









                imag_t1 = (imag_t1/imag_t1.max()) * 255
                imag_t1ce = (imag_t1ce/imag_t1ce.max()) * 255
                imag_flair = (imag_flair/imag_flair.max()) * 255
                imag_t2 = (imag_t2/imag_t2.max()) * 255



                imag_t2 = imag_t2.reshape(160,160)



                imag_t1 = imag_t1.reshape(160,160)





                imag_t1ce = imag_t1ce.reshape(160,160)


                imag_flair = imag_flair.reshape(160,160)



                imag = np.stack([imag_t1,imag_t1ce,imag_flair,imag_t2],axis=2)

















                mask = mask[40:200 ,40:200]


                y_train = mask

                labelencoder = LabelEncoder()
                n,h,w = y_train.shape
                y_train_reshape = y_train.reshape(-1,1)
                y_train_reshape_encode = labelencoder.fit_transform(y_train_reshape)
                y_train_encoded = y_train_reshape_encode.reshape(n,h,w)


                y_train = np.expand_dims(y_train_encoded, axis=3)



                train_masks_cat = to_categorical(y_train,num_classes = 4)

                y_train_cat = train_masks_cat.reshape((y_train.shape[0],y_train.shape[1],y_train.shape[2],4))






                np.save(f'brains/train/mask/{file}{image}',y_train_cat)
                np.save(f'brains/train/brain/{file}{image}',imag)
            
            

            
        
    for file in valid_dataset:
        
        
        image_t1ce = nib.load(f'{path}/{file}/{file}_t1ce.nii.gz').get_fdata()
        image_flair = nib.load(f'{path}/{file}/{file}_flair.nii.gz').get_fdata()
        image_t1 = nib.load(f'{path}/{file}/{file}_t1.nii.gz').get_fdata()
        image_t2 = nib.load(f'{path}/{file}/{file}_t2.nii.gz').get_fdata()

        image_seg = nib.load(f'{path}/{file}/{file}_seg.nii.gz').get_fdata()
        
        
        
        
        img_t2 = image_t2[:,:,1:154]
        img_flair = image_flair[:,:,1:154]
        img_t1 = image_t1[:,:,1:154]
        img_t1ce = image_t1ce[:,:,1:154]
        
        
        
        masks = image_seg[:,:,1:154]
        
        
   
        
        for image in range(0,154):
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









                imag_t1 = (imag_t1/imag_t1.max()) * 255
                imag_t1ce = (imag_t1ce/imag_t1ce.max()) * 255
                imag_flair = (imag_flair/imag_flair.max()) * 255
                imag_t2 = (imag_t2/imag_t2.max()) * 255



                imag_t2 = imag_t2.reshape(160,160)



                imag_t1 = imag_t1.reshape(160,160)





                imag_t1ce = imag_t1ce.reshape(160,160)


                imag_flair = imag_flair.reshape(160,160)



                imag = np.stack([imag_t1,imag_t1ce,imag_flair,imag_t2],axis=2)

















                mask = mask[40:200 ,40:200]


                y_train = mask

                labelencoder = LabelEncoder()
                n,h,w = y_train.shape
                y_train_reshape = y_train.reshape(-1,1)
                y_train_reshape_encode = labelencoder.fit_transform(y_train_reshape)
                y_train_encoded = y_train_reshape_encode.reshape(n,h,w)


                y_train = np.expand_dims(y_train_encoded, axis=3)



                train_masks_cat = to_categorical(y_train,num_classes = 4)

                y_train_cat = train_masks_cat.reshape((y_train.shape[0],y_train.shape[1],y_train.shape[2],4))






                np.save(f'brains/train/mask/{file}{image}',y_train_cat)
                np.save(f'brains/train/brain/{file}{image}',imag)
            

            