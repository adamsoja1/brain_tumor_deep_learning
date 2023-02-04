
import os
import wandb
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import random
from wandb.keras import WandbCallback
from KerasModels.unet_model import multi_unet_model

from Data_preparation.generator_augment import image_load_generator_x,image_load_generator_mask
from Data_preparation.generator import image_load_generator_noaug,image_load_generator_mask_noaug
from Metrics.losses_metrics import dice_coef_background,dice_coef_class1,dice_coef_class2,dice_coef_class3,dice_loss,dice_mean,iou_loss
from Preparation import DataGenerators

from KerasModels.unet_2 import unet
from New_Generator_Augment import Generator_augment
import keras.backend as K
descr = 'testing for iou loss the best model'

wandb.init(descr)
path_train = 'brains/train'
path_valid = 'brains/valid'
batch_size = 5

files = os.listdir('brains/train/brain')

random.Random(8).shuffle(files)
Generators = DataGenerators(path_train, path_valid, batch_size)


 
train_datagen,steps_train = Generators.Training_Datagen()
val_datagen,steps_val = Generators.Validation_Datagen()

datagen=Generator_augment('brains/train',files,batch_size)

#dice_loss  = sm.losses.DiceLoss(class_weights=np.array([0.01,0.9,0.9,0.8]))  # zmniejszyc background

total_loss = dice_loss 


metrics = [sm.metrics.IOUScore(), sm.metrics.FScore(), dice_coef_background, dice_coef_class1, dice_coef_class2, dice_coef_class3,dice_mean]


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

def jacard_coef(y_true, y_pred):
    
    
    total_loss =0
    for i in range(4):
        if i==0:
            continue
        
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        
        intersection = K.sum(y_true_f * y_pred_f)
        
        loss = (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
        
        total_loss = total_loss + loss
    return total_loss/3

def jacard_coef_loss(y_true, y_pred):
    return 1-jacard_coef(y_true, y_pred)

#model = unet(input_size = (160,160,4))

#early stopping dodac do callback

model = multi_unet_model(n_classes=4, IMG_HEIGHT=160, IMG_WIDTH=160, IMG_CHANNELS=4)

model.compile(optimizer=opt, loss=jacard_coef_loss, metrics=[metrics])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.summary()



history = model.fit(datagen,
          steps_per_epoch = steps_train,
          verbose=1,
          epochs=30, 
          validation_data=val_datagen, 
          validation_steps = steps_val,
          shuffle=False,
          callbacks = [WandbCallback(),]
          
)
                    


model.save(f'models/{descr}.h5')
import pandas as pd

hist_df = pd.DataFrame(history.history) 
hist_df.to_csv(f'csv_files/{descr}.csv')




from Metrics.metrics_test_generator import IOU_metric,DICE_metrics
from Data_preparation.generator import image_load_generator_mask_noaug,image_load_generator_noaug

path = 'brains/test'
batch_size = 10
files = os.listdir(f'{path}/brain')



test_generator = image_load_generator_noaug(path,files,batch_size)
mask_generator = image_load_generator_mask_noaug(path,files,batch_size)


IOU_metric(batch_size,test_generator,mask_generator,files,model)
DICE_metrics(batch_size,test_generator,mask_generator,files,model)
import pandas as pd
hist_df = pd.DataFrame(history.history) 

