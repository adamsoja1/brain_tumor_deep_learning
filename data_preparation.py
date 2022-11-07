from sklearn.preprocessing import LabelEncoder
import os
from keras.utils import to_categorical
import shutil

#sklearn.utils.shuffle() 
# array1 = np.array([[1, 1], [2, 2], [3, 3]])
# array2 = np.array([1, 2, 3])

# array1_shuffled, array2_shuffled = sklearn.utils.shuffle(array1, array2)

# print(array1_shuffled)
# OUTPUT
# [[3 3]
#  [1 1]
#  [2 2]]
# print(array2_shuffled)
# OUTPUT
# [3 1 2]


main_folder = 'RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
base_dir = 'brains'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
    
train_dir  = os.path.join(base_dir,'train')
test_dir  = os.path.join(base_dir,'test')
valid_dir  = os.path.join(base_dir,'valid')
for directory in (train_dir, valid_dir, test_dir):
    if not os.path.exists(directory):
        os.mkdir(directory)
        
brain3_dir = os.path.join(train_dir, 'brain3')
seg3_dir = os.path.join(train_dir, 'seg3')
dirs = [brain3_dir,seg3_dir ]
for directory in dirs:
    if not os.path.exists(directory):
        os.mkdir(directory)



def data_split(path,train_size,test_size,val_size):
    files  = os.listdir(path)
    
    
    
    
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        
    train_dir  = os.path.join(base_dir,'train')
    test_dir  = os.path.join(base_dir,'test')
    valid_dir  = os.path.join(base_dir,'valid')
    for directory in (train_dir, valid_dir, test_dir):
        if not os.path.exists(directory):
            os.mkdir(directory)
            
    brain3_dir = os.path.join(train_dir, 'brain3')
    seg3_dir = os.path.join(train_dir, 'seg3')
    dirs = [brain3_dir,seg3_dir ]
    for directory in dirs:
        if not os.path.exists(directory):
            os.mkdir(directory)

