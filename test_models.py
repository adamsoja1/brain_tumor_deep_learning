import tensorflow as  tf
import os
import numpy as np 
from generator import image_load_generator_x,image_load_generator_mask
import keras
import matplotlib.pyplot as plt
import random
import tensorflow as  tf
path = 'brains/test'
batch_size = 60
files = os.listdir(f'{path}/brain')

test_generator = image_load_generator_x(path,files,batch_size)
mask_generator =image_load_generator_mask(path,files,batch_size)

model = tf.keras.models.load_model('models/focal_dice2.h5',compile=False)


X_test = next(test_generator)
Y_test = next(mask_generator)



y_pred = model.predict(X_test)


predicted_img=np.argmax(y_pred,axis=3)



number = random.randint(0,40)

brain = X_test[number]
plt.figure(figsize=(30, 20))
plt.subplot(241)
plt.title('Zdjecie mozgu')
plt.imshow(brain,cmap='gray')
plt.subplot(242)
plt.title('Segmentacja przez eksperta')
plt.imshow(np.argmax(Y_test[number],axis=-1),cmap='gray')
plt.subplot(243)
plt.title('Segmentacja przez sieć')
plt.imshow(predicted_img[number],cmap='gray')

plt.show()