import numpy as np
from unet2nodrp import multi_unet_model 
from generator_augment import image_load_generator_x,image_load_generator_mask
import os
import segmentation_models as sm
import tensorflow as tf
import random
from unet_2 import unet
from generator import image_load_generator_noaug,image_load_generator_mask_noaug
import wandb
from wandb.keras import WandbCallback
from losses_metrics import dice_coef_background,dice_coef_class1,dice_coef_class2,dice_coef_class3
random.seed(8)

config = {"lr":0.001,"batch_size":10,"weights":[0.01, 1, 1, 1]}
wandb.init(project="Model_augmentations_4chan2_class_metrics30epochs_001", entity="adamsoja",config=config)


files = os.listdir(f'brains/train/brain')
files_val = os.listdir(f'brains/valid/brain')
random.shuffle(files)

batch_size = 10
 
train_X = image_load_generator_x('brains/train',files,batch_size)
train_mask_gen = image_load_generator_mask('brains/train',files,batch_size)

val_X = image_load_generator_noaug('brains/valid',files_val,batch_size)
val_mask_gen = image_load_generator_mask_noaug('brains/valid',files_val,batch_size)

steps_train = len(os.listdir('brains/train/mask')) // batch_size
steps_val = len(os.listdir('brains/valid/mask')) // batch_size

steps_val = steps_val-1

dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.01, 1, 1, 1])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1*focal_loss)
metrics = [sm.metrics.IOUScore(), sm.metrics.FScore(),dice_coef_background,dice_coef_class1,dice_coef_class2,dice_coef_class3]


train_datagen = zip(train_X,train_mask_gen)
val_datagen = zip(val_X,val_mask_gen)


opt = tf.keras.optimizers.Adam(learning_rate=0.001)


#model = unet((160,160,4))

model = multi_unet_model(n_classes=4, IMG_HEIGHT=160, IMG_WIDTH=160, IMG_CHANNELS=4)
#model = sm.Unet('resnet34', input_shape=(160, 160, 4), encoder_weights=None,classes=4,activation='softmax')
model.compile(optimizer=opt, loss=total_loss, metrics=[metrics])

#model.summary()


history = model.fit(train_datagen,
          steps_per_epoch = steps_train,
          verbose=1,
          epochs=30, 
          validation_data=val_datagen, 
          validation_steps = steps_val,
          shuffle=False,
          callbacks=[WandbCallback()],)
                    

model.save('models/modelaug1_allmetrics.h5')








