import numpy as np
from keras.metrics import MeanIoU



def IOU_metric(batch_size,generator_brain,generator_mask,files_list,model):
    steps = files_list // batch_size
    IOU_class1 = []
    IOU_class2 = []
    IOU_class3 = []
    IOU_class4 = []
    n_classes = 4
    for step in steps:
        
        X_test = next(generator_brain)
        Y_test = next(generator_mask)
        y_pred = model.predict(X_test)
        
       
        Y_test = np.argmax(Y_test,axis=3)
        y_pred = np.argmax(y_pred,axis=3)
        IOU_keras = MeanIoU(num_classes=n_classes)  
        IOU_keras.update_state(Y_test,y_pred)
        
        
        values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
        class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
        class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
        class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
        class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])
        
        IOU_class1.append(class1_IoU)
        IOU_class2.append(class2_IoU)
        IOU_class3.append(class3_IoU)
        IOU_class4.append(class4_IoU)
        
        
    print("IoU for background is: ", np.mean(IOU_class1))
    print("IoU for label 1 is: ", np.mean(IOU_class2))
    print("IoU for label 2 is: ", np.mean(IOU_class3))
    print("IoU for label 4 is: ", np.mean(IOU_class4))

    
        
    
    