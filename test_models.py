import tensorflow as  tf
import os
import numpy as np 
from Data_preparation.generator import image_load_generator_noaug,image_load_generator_mask_noaug
import matplotlib.pyplot as plt
import random
import tensorflow as  tf
from Preparation import DataGenerators



from Metrics.metrics_test_generator import IOU_metric,DICE_metrics
path = 'brains/test'


path_train = 'brains/train'
path_valid = 'brains/test'
batch_size = 30

files = os.listdir('brains/train/brain')

random.Random(8).shuffle(files)
Generators = DataGenerators(path_train, path_valid, batch_size)

val_datagen,steps_val = Generators.Validation_Datagen()





model = tf.keras.models.load_model('models/ostateczny_model.h5',compile=False)

X_test,Y_test = next(val_datagen)



y_pred = model.predict(X_test)


predicted_img=np.argmax(y_pred,axis=3)

#sprawdzić ktory kolor to jaka klasa
#learning rate na koncu
#loss/augmentacja/architektura najpierw
#regularyzacja
#batch_size

zdjecia_ekspert = np.argmax(Y_test,axis=-1)
zdjecia_siec = np.argmax(y_pred,axis=-1)

zdjecia_ekspert = (zdjecia_ekspert/zdjecia_ekspert.max()) * 255
zdjecia_siec = (zdjecia_siec/zdjecia_siec.max()) * 255

eskpert = []
siec = []

for i in range(zdjecia_ekspert.shape[0]):
    eskpert.append(zdjecia_ekspert[i,:,:])
    siec.append(zdjecia_siec[i,:,:])
    

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(30, 15))

# Iteruj przez każdy obraz i wyświetl go w odpowiednim subplot
for ax, image in zip(axs.flat, eskpert):
    ax.imshow(image,cmap='gray')

# Pokaż subplots
plt.show()




fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(30, 15))

# Iteruj przez każdy obraz i wyświetl go w odpowiednim subplot
for ax, image in zip(axs.flat, siec):
    ax.imshow(image,cmap = 'gray')

# Pokaż subplots
plt.show()




from Metrics.metrics_test_generator import IOU_metric,DICE_metrics
from Data_preparation.generator import image_load_generator_mask_noaug,image_load_generator_noaug

path = 'brains/test'
batch_size = 10
files = os.listdir(f'{path}/brain')



test_generator = image_load_generator_noaug(path,files,batch_size)
mask_generator = image_load_generator_mask_noaug(path,files,batch_size)


IOU_metric(batch_size,test_generator,mask_generator,files,model)
DICE_metrics(batch_size,test_generator,mask_generator,files,model)
