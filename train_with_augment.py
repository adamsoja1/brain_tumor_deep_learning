


import os
import wandb
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import random
from wandb.keras import WandbCallback


from Data_preparation.generator_augment import image_load_generator_x,image_load_generator_mask
from Data_preparation.generator import image_load_generator_noaug,image_load_generator_mask_noaug

from KerasModels.unet_model import multi_unet_model
from Preparation import DataGenerators


#from KerasModels.unet_2 import unet
from New_Generator_Augment import Generator_augment

from Metrics.losses_metrics import dice_coef_background,dice_coef_class1,dice_coef_class2,dice_coef_class3,dice_loss

path_train = 'brains/train'
path_valid = 'brains/valid'
batch_size = 5

files = os.listdir('brains/train/brain')

random.Random(8).shuffle(files)
Generators = DataGenerators(path_train, path_valid, batch_size)




train_datagen,steps_train = Generators.Training_Datagen()
val_datagen,steps_val = Generators.Validation_Datagen()

datagen=Generator_augment('brains/train',files,batch_size)

#dice_loss = sm.losses.DiceLoss(class_weights=np.array([0, 1, 1, 0.9]))  # zmniejszyc background
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss 


metrics = [sm.metrics.IOUScore(), sm.metrics.FScore(),dice_coef_background,dice_coef_class1,dice_coef_class2,dice_coef_class3]


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)



#model = unet(input_size = (160,160,3))



model = multi_unet_model(n_classes=4, IMG_HEIGHT=160, IMG_WIDTH=160, IMG_CHANNELS=4)
model.compile(optimizer=opt, loss=dice_loss, metrics=[metrics])


#model.summary()

wandb.init(project="custom_dice_loss batch5", entity="adamsoja")
history = model.fit(datagen,
          steps_per_epoch = steps_train,
          verbose=1,
          epochs=30, 
          validation_data=val_datagen, 
          validation_steps = steps_val,
          shuffle=False,
          callbacks=[WandbCallback()],)
                    

model.save('models/new_model_XD.h5')

from PlotsFunctions import make_plot,make_stacked_plot

metrics = ['dice_coef_class1','dice_coef_class2','dice_coef_class3']

#segment dla zwyklych dice
for metric in metrics:
    make_plot(history,metrics[i],metrics[i])



make_stacked_plot(history,metrics,'Stack_')

make_plot(history,'iou_score','iou_score')


from Metrics.metrics_test_generator import IOU_metric,DICE_metrics
from Data_preparation.generator import image_load_generator_mask_noaug,image_load_generator_noaug

path = 'brains/test'
batch_size = 150
files = os.listdir(f'{path}/brain')


test_generator = image_load_generator_noaug(path,files,batch_size)
mask_generator = image_load_generator_mask_noaug(path,files,batch_size)


IOU_metric(batch_size,test_generator,mask_generator,files,model)
DICE_metrics(batch_size,test_generator,mask_generator,files,model)
