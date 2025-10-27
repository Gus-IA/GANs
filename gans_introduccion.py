import torch
import torchvision
import numpy as np
import random 
import matplotlib.pyplot as plt
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# descargamos el dataset fashion mnist con ropa
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)

classes = ("t-shirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot")

# creamos el dataset y normalizamos las imaǵenes
class Dataset(torch.utils.data.Dataset):
  def __init__(self, trainset):
    self.imgs = torch.tensor([np.array(i[0]).flatten() / 255. for i in trainset], dtype=torch.float, device=device)
    self.labels = torch.tensor([i[1] for i in trainset], dtype=torch.long, device=device)

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, ix):
    return self.imgs[ix], self.labels[ix]

train = Dataset(trainset)
len(train)


img, label = train[0]
print(img.shape, img.dtype, img.max(), img.min())

# creamos el dataloader
dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

imgs, labels = next(iter(dataloader))
print(imgs.shape, labels.shape)

# mostramos un ejemplo del dataset
r, c = 3, 5
plt.figure(figsize=(c*3, r*3))
for row in range(r):
    for col in range(c):
        index = c*row + col
        plt.subplot(r, c, index + 1)
        ix = random.randint(0, len(train)-1)
        img, label = train[ix]
        plt.imshow(img.reshape(28,28).cpu())
        plt.axis('off')
        plt.title(classes[label.item()])
plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.show()


# arquitectura de la red neuronal, perceptrón multicapa
def block(n_in, n_out):
  return nn.Sequential(
      nn.Linear(n_in, n_out),
      nn.ReLU(inplace=True)
  )

class MLP(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.fc1 = block(input_size, 150)
    self.fc2 = block(150, 100)
    self.fc3 = nn.Linear(100, output_size)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x
  


n_in, n_out = 30, 28*28
generator = MLP(n_in, n_out)



output = generator(torch.randn(64, 30))
print(output.shape)


plt.imshow(output[0].reshape(28,28).detach().numpy())
plt.show()


discriminator = MLP(28*28, 1)
output = discriminator(torch.randn(64, 28*28))
print(output.shape)