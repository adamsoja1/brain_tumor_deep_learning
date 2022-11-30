import tensorflow as tf
import nibabel as nib
import numpy as np
import os

class DataLoader(tf.keras.utils.Sequence):
    
    def __init__(self,images,labels,batch_size,
                 path):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.path = path
        self.batch_end = batch_size
        self.batch_start = 0
    
    def __len__(self):
        return np.math.ceil(len(self.images)/self.batch_size)
   
    def __iter__(self):
        self.batch_start +=self.batch_size
        self.batch_end +=self.batch_size 
        return self
        
    def __get_data(self,files):
        brain = []
        seg = []
        
        for file in files:
            image_flair = nib.load(f'{self.path}/{file}/{file}_flair.nii.gz').get_fdata()
            image_seg = nib.load(f'{self.path}/{file}/{file}_seg.nii.gz').get_fdata()
            
            brain.append(image_flair[:,:,70:120])
            seg.append(image_seg[:,:,70:120])
            
        return np.array(brain),np.array(seg)
    
    def __getitem__(self):
        self.batch_end = min(int(self.batch_end),int(len(self.images)))
        files = self.images[self.batch_start:self.batch_end]
        brain,seg = self.__get_data(files)
        self.__iter__()
        return brain,seg
    
    
path = 'RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
images = os.listdir(path)
data_loader = DataLoader(images,images,2,path)

x,y = data_loader.__getitem__()
