# Deep-Learning
Convolutional neural networks (CNNs)
State-of-the-art image classification is performed with convolutional neural networks (CNNs) that use convolution layers to extract features from images and pooling layers to downsize images so features can be detected at various resolutions. Let's use Keras to build a CNN and train it to differentiate between photos containing Arctic foxes, polar bears, and walruses. CNNs perform best when trained with thousands (or tens of thousands) of images per class. In this example, we'll use 300 images for training (100 per class) and 120 for testing.

Load training and testing images

The first step is to load the images that will be used for training and testing and to label the images with 0 for Arctic foxes, 1 for polar bears, and 2 for walruses. We'll start by defining a function for loading images from the file system and affixing labels to them, and another function for displaying images. We will also define four Python lists to hold the images used for training and testing (x_train and x_test) and the labels used for training and testing (y_train and y_test).



import os
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
%matplotlib inline

def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        img = image.load_img(os.path.join(path, file), target_size=(224, 224, 3))
        images.append(image.img_to_array(img))
        labels.append((label))
        
    return images, labels

def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)

x_train = []
y_train = []
x_test = []
y_test = []
