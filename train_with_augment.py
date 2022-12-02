import os
import wandb
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import random
from wandb.keras import WandbCallback




from KerasModels.unet_model import multi_unet_model 
from Preparation import DataGenerators

#wandb.init(project="Model_augmentations_4chan2_class_metrics30epochs_0001_2_dropouts_onlyDICE__different_augmentation_littlerotations_uint8", entity="adamsoja")
#from KerasModels.unet_2 import unet



from Metrics.losses_metrics import dice_coef_background,dice_coef_class1,dice_coef_class2,dice_coef_class3
from keras.losses import CategoricalCrossentropy



np.random.seed(8)

path_train = 'brains/train/brain'
path_valid = 'brains/valid/brain'
batch_size =9


Generators = DataGenerators(path_train, path_valid, batch_size)




train_datagen,steps_train = Generators.Training_Datagen()
val_datagen,steps_val = Generators.Validation_Datagen()



dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 1])) 
total_loss = dice_loss


metrics = [sm.metrics.IOUScore(), sm.metrics.FScore(),dice_coef_background,dice_coef_class1,dice_coef_class2,dice_coef_class3]


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)


#model = unet((160,160,4))

model = multi_unet_model(n_classes=4, IMG_HEIGHT=160, IMG_WIDTH=160, IMG_CHANNELS=4)
#model = sm.Unet('resnet34', input_shape=(160, 160, 4), encoder_weights=None,classes=4,activation='softmax')
model.compile(optimizer=opt, loss=total_loss, metrics=[metrics])

#model.summary()


history = model.fit(train_datagen,
          steps_per_epoch = steps_train,
          verbose=1,
          epochs=20, 
          validation_data=val_datagen, 
          validation_steps = steps_val,
          shuffle=False,
          callbacks=[WandbCallback()],)
                    

model.save('models/modelaug1_allmetrics_combinedlosses_drp22_2.h5')






from Metrics.metrics_test_generator import IOU_metric,DICE_metrics



path = 'brains/test'
batch_size = 150
files = os.listdir(f'{path}/brain')

test_generator = image_load_generator_noaug(path,files,batch_size)
mask_generator = image_load_generator_mask_noaug(path,files,batch_size)




IOU_metric(batch_size,test_generator,mask_generator,files,model)

DICE_metrics(batch_size,test_generator,mask_generator,files,model)