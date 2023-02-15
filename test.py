from model import Autoencoder
from data import test_loader, noise_factor

# obtaining one batch of test images
test_images, test_labels = next(iter(test_loader))                                                
test_images_noisy = test_images + noise_factor * torch.randn(test_images.shape) # shape = torch.Size([32, 3, 32, 32])
test_images_noisy = np.clip(test_images_noisy, 0., 1.)                # clip to make the values fall between 0 and 1

#test_images = test_images.numpy() 
#test_images_noisy=test_images_noisy.numpy()

net = Autoencoder()
# get sample outputs
output = net(test_images_noisy)
output = output.detach()



def Results(orig, noise, denoise, num=10):
    import matplotlib.pyplot as plt
    n = num
    plt.figure(figsize=(20, 6))

    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i+1)
        plt.imshow(test_images[i].permute(1,2,0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy image
        ax = plt.subplot(3, n, i +1 + n)
        plt.imshow(test_images_noisy[i].permute(1,2,0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display denoised image
        ax = plt.subplot(3, n, i +1 + n + n)
        plt.imshow(denoise[i].permute(1,2,0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.figtext(0.5,0.95, "Original test images", ha="center", va="top", fontsize=14, color="r")
    plt.figtext(0.5,0.65, "Noisy test images", ha="center", va="top", fontsize=14, color="g")
    plt.figtext(0.5,0.35, " Denoised test images", ha="center", va="top", fontsize=14, color="b")
    plt.subplots_adjust(hspace = 0.5 )
    plt.show()
    
Results(test_images, test_images_noisy, output)