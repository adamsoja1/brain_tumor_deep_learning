import numpy as np
from unet_model import multi_unet_model 
from generator import image_load_generator_x,image_load_generator_mask
import os
import segmentation_models as sm
import tensorflow as tf
import random
from unet_2 import unet

import wandb
from wandb.keras import WandbCallback

wandb.init(project="Model2_CLahe_dice160x160", entity="adamsoja")


files = os.listdir(f'brains/train/brain')
files_val = os.listdir(f'brains/valid/brain')
random.shuffle(files)

batch_size = 25

train_X = image_load_generator_x('brains/train',files,batch_size)
train_mask_gen = image_load_generator_mask('brains/train',files,batch_size)

val_X = image_load_generator_x('brains/valid',files_val,batch_size)
val_mask_gen = image_load_generator_mask('brains/valid',files_val,batch_size)

steps_train = len(os.listdir('brains/train/mask')) // batch_size
steps_val = len(os.listdir('brains/valid/mask')) // batch_size


dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.012, 0.543, 0.495, 1.0])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss
metrics = [sm.metrics.IOUScore(), sm.metrics.FScore()]


train_datagen = zip(train_X,train_mask_gen)
val_datagen = zip(val_X,val_mask_gen)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)


#model = unet((240,240,1))

model = multi_unet_model(n_classes=4, IMG_HEIGHT=160, IMG_WIDTH=160, IMG_CHANNELS=1)
model.compile(optimizer=opt, loss=total_loss, metrics=[metrics])

#model.summary()


history = model.fit(train_datagen,
          steps_per_epoch = steps_train,
          verbose=1,
          epochs=50, 
          validation_data=val_datagen, 
          validation_steps = steps_val,
          shuffle=False,
          callbacks=[WandbCallback()],)
                    

model.save('models/dice_160x160.h5')


x = [1.0, 98.20064623542751, 89.0883531885646, 179.757707413642]


for i in range(4):
    print(x[i]/x[-1])