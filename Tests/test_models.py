import tensorflow as  tf
import os
import numpy as np 
from generator import image_load_generator_noaug,image_load_generator_mask_noaug
import keras
import matplotlib.pyplot as plt
import random
import tensorflow as  tf

from metrics_test_generator import IOU_metric,DICE_metrics
path = 'brains/test'
batch_size = 150
files = os.listdir(f'{path}/brain')

test_generator = image_load_generator_noaug(path,files,batch_size)
mask_generator = image_load_generator_mask_noaug(path,files,batch_size)


model = tf.keras.models.load_model('models/modelaug1_allmetrics_combinedlosses_drp.h5',compile=False)

IOU_metric(batch_size,test_generator,mask_generator,files,model)

DICE_metrics(batch_size,test_generator,mask_generator,files,model)


X_test = next(test_generator)
Y_test = next(mask_generator)



y_pred = model.predict(X_test)


predicted_img=np.argmax(y_pred,axis=3)

#sprawdzić ktory kolor to jaka klasa
#learning rate na koncu
#loss/augmentacja/architektura najpierw
#regularyzacja
#batch_size




number = random.randint(0,40)

brain = X_test[number]
plt.figure(figsize=(30, 149))
plt.subplot(241)
plt.title('Zdjecie mozgu')
plt.imshow(brain[:,:,1],cmap='gray')
plt.subplot(242)
plt.title('Segmentacja przez eksperta')
plt.imshow(np.argmax(Y_test[number],axis=-1),cmap='gray')
plt.subplot(243)
plt.title('Segmentacja przez sieć')
plt.imshow(predicted_img[number],cmap='gray')

plt.show()


from keras.metrics import MeanIoU
n_classes = 4
masks = np.argmax(Y_test,axis=3)
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(masks,predicted_img)
print("Mean IoU =", IOU_keras.result().numpy())


values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for background is: ", class1_IoU)
print("IoU for label 1 is: ", class2_IoU)
print("IoU for label 2 is: ", class3_IoU)
print("IoU for label 4 is: ", class4_IoU)
