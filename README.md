# Image-Denoising-using-Autoencoders with skip connections

## Architecture




The basic architecture used for the denoising is a fully convolutional auto-encoder. The network has two main parts: 

(a) **Encoder** - A sequence of  3x3 convolutional layers followed by Pooling, Batch Normalization layer and a ReLU non-linearity layer. which compresses the input image into a lower dimensional latent representation. Encoder acts as a feature extractor.

(b) **Decoder** - A sequence of deconvolutional layers symmetric to the convolutional layers which reconstructs the input images by decoding (up-sampling operation) the low-dimensional encoded images. 

The corresponding encoder and decoder layers are connected by shortcut connections (skip connections). The main idea behind using the shortcut connections are used to pass feature maps forwardly

## Noise Type
