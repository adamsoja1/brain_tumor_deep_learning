#datagen function preparation
import os
import random

from Data_preparation.generator_augment import image_load_generator_x,image_load_generator_mask
from Data_preparation.generator import image_load_generator_noaug,image_load_generator_mask_noaug





class DataGenerators:
    
    def __init__(self,path_train,path_valid,batch_size):
        self.path_train = path_train
        self.path_valid = path_valid
        self.batch_size = batch_size
        
    
    
    def Get_files(self,path):
        files = os.listdir(f'{path}/brain')

        return files


    def Training_Datagen(self):
       
        train_files = self.Get_files(self.path_train)
  
        brain_datagen = image_load_generator_x(self.path_train,train_files,self.batch_size)
        mask_datagen = image_load_generator_mask(self.path_train,train_files,self.batch_size)
        steps = len(train_files) // self.batch_size
        
        return zip(brain_datagen,mask_datagen),steps
    
        
    def Validation_Datagen(self):
        valid_files = self.Get_files(self.path_valid)

        brain_datagen = image_load_generator_noaug(self.path_valid,valid_files,self.batch_size)
        mask_datagen = image_load_generator_mask_noaug(self.path_valid,valid_files,self.batch_size)
        steps = len(valid_files) // self.batch_size
        
        return zip(brain_datagen,mask_datagen),steps
        
    