import os
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# torchvision contains many popular computer vision datasets, deep neural network architectures, and image processing modules.
import torchvision

from torchvision import datasets
from torchvision.transforms import transforms

# save_image: torchvision.utils provides this module to easily save PyTorch tensor images.
from torchvision.utils import make_grid, save_image

# Dataloader: eases the task of making iterable training and testing sets.
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

##########################
# Transformation on Images
##########################
transform = transforms.Compose([transforms.ToTensor()])                 # converts images or numpy arrays to tensors

##################
# Dataset Loading
##################

train_set = datasets.CIFAR10(root='./data/', train =True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data/', train=False, transform=transform)

print("no of training images:",len(train_set))
print("no of test images:",len(test_set))

#classes = train_set.classes
#print(classes)

batch_size = 64

########################
# preparing data loaders 
########################

train_loader = DataLoader(train_set, batch_size, num_workers=2, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size*2, num_workers=2, pin_memory=True)

training_batch = next(iter(train_loader))
test_batch = next(iter(test_set))
print(type(training_batch))                             #  First element of list is a list of Image Tensor 
print(len(training_batch))                              # 2  ( )
print(type(training_batch[1]))
print(training_batch[0].shape)                          # shape of all images in a batch
print(training_batch[1].shape)                          # this shows the size of labels
print(training_batch[0][0].shape)                       # Returns shape of 1st image out of 32 images
print(training_batch[1][0])

noise_factor = 0.15
#################################
# Adding noise to training images
#################################
train_dataiter = iter(train_loader)
train_images, train_labels = next(train_dataiter)                                                
train_images_noisy = train_images + noise_factor * torch.randn(train_images.shape) # shape = torch.Size([32, 3, 32, 32])
train_images_noisy = np.clip(train_images_noisy, 0., 1.)                 # clip to make the values fall between 0 and 1

#############################################
# AFunction to plot original and Noisy Images
#############################################

def showOrigDec(orig, noise, num=10):
    import matplotlib.pyplot as plt
    n = num
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(train_images[i].permute(1,2,0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display noisy
        ax = plt.subplot(2, n, i +1 + n)
        plt.imshow(train_images_noisy[i].permute(1,2,0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.figtext(0.5,0.95, "ORIGINAL IMAGES", ha="center", va="top", fontsize=14, color="r")
    plt.figtext(0.5,0.5, "NOISY IMAGES", ha="center", va="top", fontsize=14, color="r")
    plt.subplots_adjust(hspace = 0.3 )
        
    plt.show()

    """ Given a Tensor representing the image, use .permute() to put the channels as the last dimension.
        Not using permute method will throw an error
    """
    