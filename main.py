import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model.cae import ConvolutionalAE
from model.utilities import train
import matplotlib.pyplot as plt

# Constants
batch_size = 64

# Init model
model = ConvolutionalAE()
print(model)

# Load data
mnist_data = datasets.MNIST('data', train=True, download=True, transform = transforms.ToTensor())
mnist_data = list(mnist_data)[:4096]
train_loader = torch.utils.data.DataLoader(mnist_data, batch_size = batch_size, shuffle=True)

# Start train
max_epochs= 20
outputs = train(model, train_loader, epochs = max_epochs)

# Draw figures
print("Draw figure")
for k in range(0, max_epochs,5):
    plt.figure(figsize=(9, 2))
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i>=9: break
        plt.subplot(2,9, i+1)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i>=9: break
        plt.subplot(2,9,9+i+1)
        plt.imshow(item[0])
    plt.show()


# Draw interpolation
x1 = outputs[max_epochs-1][1][0,:,:,:]
x2 = outputs[max_epochs-1][1][8,:,:,:]

x = torch.stack([x1, x2])
emb = model.encoder(x)
e1 = emb[0]
e2 = emb[1]

emb_values = []
for i in range(0,10):
    e = e1 * (i/10) + e2 * (10-i)/10
    emb_values.append(e)
emb_values = torch.stack(emb_values)

recons = model.decoder(emb_values)
plt.figure(figsize=(10,2))
for i, recon in enumerate(recons.detach().numpy()):
    plt.subplot(2, 10, i+1)
    plt.imshow(recon[0])

plt.show()