import torch
# torch.nn contains the deep learning neural network layers such as Linear(), and Conv2d().
import torch.nn as nn
#  functional:  we will use this for activation functions such as ReLU.
import torch.nn.functional as F



# autoencoder model

class Autoencoder(nn.Module):
    # in def_init__    function we define all the layers that we will use to build the network
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder layers
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)                   # in_channel =3, out_channel = 64
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)                  # in_channel =64, out_channel = 32
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)                  # in_channel =32, out_channel = 16
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)                   # in_channel =16, out_channel = 8
        self.pool = nn.MaxPool2d(2, 2)                                           # both kernels and stride with value 2
        
        # decoder layers     ( consists of ConvTranspose Layer )
        # we keep on increasing the dimensionality till we get 64 out_channels in self.dec4.
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)  
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        # nn.Conv2d() layer with 3 output channel so as to reconstruct the original image.
        self.out = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    def forward(self, x):
        # encode
        x = F.relu(self.enc1(x))                                                             # 32 * 32
        x = self.pool(x)                                                                     # 16 * 16
        x1 = F.relu(self.enc2(x))                                                            # 16 * 16
        x = self.pool(x1)                                                                    # 8 * 8 
        x2 = F.relu(self.enc3(x))                                                            # 8 * 8
        x = self.pool(x2)                                                                    # 4 * 4
        x3 = F.relu(self.enc4(x))                                                            # 4 * 4
        x = self.pool(x3) # the latent space representation of input data                    # 2 * 2
        # decode
        x = F.relu(self.dec1(x))                                                             # 4 * 4
        x = x3 + x                                                        
        x = F.relu(self.dec2(x))                                                             # 8 * 8
        x = x2 + x
        x = F.relu(self.dec3(x))                                                             # 16 * 16
        x = x1 + x
        x = F.relu(self.dec4(x))                                                             # 32 * 32
        x = torch.sigmoid(self.out(x))                                                       # 32 * 32
        return x

#net = Autoencoder()
#print(net)
