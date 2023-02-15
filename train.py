from model import Autoencoder
from data import train_loader, noise_factor
# torch.optim contains the deep learning optimizer classes such as MSELoss() and many others as well.
import torch.optim as optim
import torch.nn as nn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

net = Autoencoder()

###############
# loss function
###############
criterion = nn.MSELoss() # mean squared error to calculate the dissimilarity between the original pixel values and the predicted pixel values.

############
# optimizer
############
learning_rate = 0.001
optimizer = optim.Adam(net.parameters(), lr= learning_rate)

num_epochs = 5

###################
# training function
###################

def train(net, train_loader, num_epochs):
  training_loss = []
  for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0

    for images, labels in train_loader:                                          
      # adding random noise to the input training images
      train_image_noisy = images + noise_factor * torch.randn(images.shape)
      train_image_noisy = np.clip(train_image_noisy, 0., 1.)                        # clip to make the values fall between 0 and 1

      optimizer.zero_grad()                                                         # clearing the all optimized gradients to zero at the beginning of each batch
      model_output = net(train_image_noisy)                                         # compute predicted outputs by passing *noisy* images to the model
      loss = criterion(model_output, images)                                        # loss calculation
      loss.backward(retain_graph=True)                                              # backpropagating the gradients to compute gradient of the loss with respect to model parameters
      optimizer.step()                                                              # updating the parameters
      running_loss += loss.item()                                                   # adding the losses to "running_loss" variable

    loss = running_loss / len(train_loader)                                         # print average training statistics
    training_loss.append(loss)
    print('Epoch {} / {}, Train Loss: {:.3f}'.format(
            epoch+1, num_epochs, loss))
    
  return training_loss

train_loss = train(net, train_loader, num_epochs)
plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')