from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import initializers
def unet(input_size = (160,160,4)):
    ''' U-Net
    Ags:
        args (argparse):    Arguments parsed in command-lind
        input_size(tuple):  Model input size
    '''

    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    
    
    
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same',
                kernel_initializer='he_normal')(conv5),conv4], axis=3)
    
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',
                kernel_initializer='he_normal')(conv6),conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2),padding='same',
                kernel_initializer='he_normal')(conv7),conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',)(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',
                kernel_initializer='he_normal')(conv8),conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(4, (1, 1), activation='relu',
                    kernel_initializer='he_normal')(conv9)
    conv10 = Activation('softmax')(conv10)
    model = Model(inputs=[inputs], outputs=[conv10])

    return model