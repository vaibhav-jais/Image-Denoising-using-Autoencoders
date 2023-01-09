# Image-Denoising-using-Autoencoders with skip connections

## Architecture

The basic architecture used for the denoising is a fully convolutional auto-encoder. The network has two main parts: 

(a) **Encoder** - A sequence of  3x3 convolutional layers followed by Pooling, Batch Normalization layer and a ReLU non-linearity layer. The encoder learns to extract the image features at each step separating them from the noise and compresses the input image into a lower dimensional latent representation. Encoder acts as a feature extractor.

(b) **Decoder** - A sequence of deconvolutional layers symmetric to the convolutional layers which reconstructs the input images. Deconvolutional layers work roughly like a combination of convolutional and upsampling layers.

The corresponding encoder and decoder layers are connected by shortcut connections (skip connections). The main idea behind using the shortcut connections are used to pass feature maps directly from an early layer of an encoder to a later layer of the decoder. This helps the decoder form more clearly defined decompressions of the input image.

## Noise Type

In this project,  the images were corrupted with **Pixel-level Gaussian noise** for training and testing. In this Pixel-level Gaussian noise, given an image x, a random Gaussian noise with mean 0 and standard deviation &sigma is added to each pixel uniformly

## Training Pipeline

The Dataset used for the project is **CIFAR-10** dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The network is trained on training images of 64 batch size. MSE error is used as the loss function and the ADAM optimizer with learning rate of 0.001 gave the least training loss. The Deep learning framework used is PyTorch.





## Results:

Once the model was trained, the performance of the model was checked on test dataset. The noise was added to the test images and the **"noisy"** images were used as the input for the model and the model reconstructed the denoised images which is shown below:

![Denoising Results](https://user-images.githubusercontent.com/87435328/211412600-421c7789-6643-4440-98f4-c756b94ec108.png)
