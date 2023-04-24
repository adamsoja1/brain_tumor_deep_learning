
# Brain tumor segmenation model

This project is a segmentation model to detect neoplastic changes in the brain


## Brats 2021 dataset


The dataset was obtained from the BraTS 2021 edition competition. The data contains 1251 folders with the corresponding brain number. Each folder contains five separate files, i.e: an area of neoplastic changes marked by an expert, i.e. masks, and a picture of the brain in four different types: t1, t1ce, t2 and flair. Dataset includes images of brains containing glioblastoma multiforme and lower grade glioblastoma with pathological confirmed diagnosis.



| Brain | Tumor |
|-------|-------|
| ![Brain Gif](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjc0NzNiMWM4MDBiNDY5MGYzNGM2MjY1NTRmMzJjNTVjM2MwYTAzZiZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/sD2iwnhJzK1oNBuiYI/giphy.gif) | ![Tumor Gif](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZGUxYTFiMjlmMjc0ZjhjODE3Njc3NWMzYTY4NmZkOWVhN2U0NmY2YyZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/gLGF7DO4b1LKY1Ydk6/giphy.gif) |





## U-net
U-Net is a convolutional neural network architecture that was originally designed for image segmentation tasks. It was proposed by researchers from the Computer Science Department of the University of Freiburg in 2015.

The U-Net architecture consists of two main parts: an encoder and a decoder. The encoder takes an input image and applies a series of convolutional and pooling layers to extract high-level features. The decoder then takes the features from the encoder and uses them to generate a segmentation map of the input image.

One of the key features of U-Net is its skip connections, which allow the network to combine high-level features from the encoder with low-level features from the decoder. This helps to preserve spatial information and improve the accuracy of the segmentation.

#### Schema of tested model

![Imgur](https://i.imgur.com/yBkNxOE.png)


## Metrics and loss

Dice and Jaccard (IoU) Intersection Over Union similarity coefficients were used to test the similarity between two sets of data. In this case, masks marked by an expert were compared to masks marked by the network. Both Dice and IoU are limited between 0 and 1.


The value of the coefficients equal to 1 means that the model performed the segmentation process perfectly. It is important to use both metrics because they differ in how they work. With semantic segmentation, one metric may show a high score and the other a low score, making the score unreliable. However, the project tracked DICE metrics for individual classes and the averaged IoU metric across all classes.

Loss function formula: 1-DICE. By subtracting the DICE metric, which is in the range 0-1 from one, the loss will decrease and this is how the model learns.

![metrics](https://imgur.com/cynWByf.png)


                      
## Methodology
Each photo was cropped to contain only pixels describing the brain. It was considered that the most optimal for all brain slices would be to remove the background pixels to the size of 160 by 160 pixels. Then, a stack of flair, t1, t1ce, and t2 images was created. Using the numpy library, the stack was transformed into a photo containing 4 channels, where the channel represented the type of photo.

Each image of the mask was changed in such a way that the following values were present in the matrices: 0, 1, 2, 3. Then the entire matrix of each image was changed to categorical values using the keras library function "to_categorical". The function is responsible for changing a mask photo with one channel (the same as a grayscale image) to a binary photo containing four channels (coloured photos have 3 channels), where in the first channel the values 1 mean the background - class 0, in the second channel
tumor core - class 1, in the third canal swelling - class 2, in the fourth canal leaking tumor - class 3. 

In other words, if you define the pixel representing the background, its value will be: [1,0,0,0], for the pixel with the core of the tumor: [0,1,0,0], and so on. In addition, similarly to the images of the brain, the masks were cut to the size of 160x160.
In the end, the images of the brain and the mask were of shape 160x160x4.

Then augmentation was performed using small rotation on previously saved images without extending further dataset. This approach allowed faster computations.

Tested parameters:
![params](https://imgur.com/lipjXXx.png)


## Results
![res](https://imgur.com/dRq6ajB.png)

Blue  -  Training data, Orange - Validation Data

| Dice Blue  -  Training data, Orange - Validation Data | IoU Blue  -  Training data, Orange - Validation Data |
|-------|-------|
| ![res2](https://imgur.com/GvSb9B4.png) | ![res3](https://imgur.com/f9lM9a3.png) |

## Examples
#### a)  shows ground thruths
#### b)  shows model predictions
![exampl](https://imgur.com/EHXO9zc.png)
## Technologies used
**Programming language:** Python 3.8

**Model:** Tensorflow, keras

**Image analysis:** Numpy, albumentations, opencv, PILLOW

**Visualization:** seaborn, matplotlib

**Hardware:** RTX 3060 Notebook 6GB, Intel Core i5 10500H


### Computation on RTX 3060 Notebook version
A single learning process of such a network took an average of 13 hours. For this reason, parameter tuning was not performed as it would take too long and could damage the hardware as the average GPU temperature was in the 80-90°C range during training. The figure below shows a graph generated by the wandb portal showing the temperature of the graphics card.

**Y axis - temperature**

**X axis - hours**




![computation](https://imgur.com/x3woGen.png)
## Authors

- [@adamsoja1](https://github.com/adamsoja1)


