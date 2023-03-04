#!/usr/bin/env python3
import PIL
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

if False:
    mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    mnist_data = list(mnist_data)[:4096]
else:
    # mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    # mnist_data = list(mnist_data)[:4096]
    directory = 'mydata/images'
    classes = os.listdir(directory)
    mnist_data = []
    for i in range(len(classes)):
        c = classes[i]
        image_file_names = os.listdir(directory + '/' + c)
        for image_file_name in image_file_names:
            img = PIL.Image.open(directory + '/' + c + '/' + image_file_name)
            img = img.resize((256, 256))
            mnist_data.append((transforms.ToTensor()(img), i))

print('TRAINING: ', len(mnist_data))

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((len(x), *self.shape))

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # self.encoder = nn.Sequential( # like the Composition layer you built
        #     nn.Conv2d(1, 16, 3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 7)
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, 7),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        #     nn.Sigmoid()
        # )

        memory = 64

        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
            nn.Flatten(),
            nn.Linear(in_features=64*58*58, out_features=memory),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=memory, out_features=64*58*58),
            Reshape(64, 58, 58),
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(model, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=1e-5) # <--
    train_loader = torch.utils.data.DataLoader(mnist_data, 
                                               batch_size=batch_size, 
                                               shuffle=True)
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs


model = Autoencoder()
#max_epochs = 20
#max_epochs = 2**11
max_epochs = 256
outputs = train(model, num_epochs=max_epochs)


for k in range(0, max_epochs, 32):
    plt.figure(figsize=(9, 2))
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(item[0])


imgs = outputs[max_epochs-1][1].detach().numpy()
plt.plot(1, 2, 1)
plt.imshow(imgs[0][0])
plt.plot(1, 2, 2)
plt.imshow(imgs[8][0])
plt.show()

x1 = outputs[max_epochs-1][1][0,:,:,:] # first image
x2 = outputs[max_epochs-1][1][8,:,:,:] # second image
x = torch.stack([x1,x2])     # stack them together so we only call `encoder` once
embedding = model.encoder(x)
e1 = embedding[0] # embedding of first image
e2 = embedding[1] # embedding of second image


embedding_values = []
for i in range(0, 10):
    e = e1 * (i/10) + e2 * (10-i)/10
    embedding_values.append(e)
embedding_values = torch.stack(embedding_values)

recons = model.decoder(embedding_values)


plt.figure(figsize=(10, 2))
for i, recon in enumerate(recons.detach().numpy()):
    plt.subplot(2,10,i+1)
    plt.imshow(recon[0])
plt.subplot(2,10,11)
plt.imshow(imgs[8][0])
plt.subplot(2,10,20)
plt.imshow(imgs[0][0])


def interpolate(index1, index2):
    x1 = mnist_data[index1][0]
    x2 = mnist_data[index2][0]
    x = torch.stack([x1,x2])
    embedding = model.encoder(x)
    e1 = embedding[0] # embedding of first image
    e2 = embedding[1] # embedding of second image


    embedding_values = []
    for i in range(0, 11):
        e = e1 * (i/10) + e2 * (10-i)/10
        embedding_values.append(e)
    embedding_values = torch.stack(embedding_values)

    recons = model.decoder(embedding_values)

    plt.figure(figsize=(10, 2))
    for i, recon in enumerate(recons.detach().numpy()):
        plt.subplot(2,10,i+1)
        plt.imshow(recon[0])
        PIL.Image.fromarray(np.uint8(np.swapaxes(np.swapaxes(recon, 0, 2), 0, 1) * 255)).show()

    plt.subplot(2,10,11)
    plt.imshow(x2[0])
    plt.subplot(2,10,20)
    plt.imshow(x1[0])
    plt.show()

def to_image(output):
    return PIL.Image.fromarray(np.uint8(np.swapaxes(np.swapaxes(output.detach().numpy(), 0, 2), 0, 1) * 255))
def to_images(outputs):
    images = []
    for output in outputs:
        images.append(to_image(output))
    return images

def to_video(pics):
    pics[0].save('temp_result.gif', save_all=True,optimize=False, append_images=pics[1:], loop=0)

def repeat_image(img, n=2**11):
    images = []
    for _ in range(n):
        images.append(img)
        img = model(img)
    return immages


def interpolate_img(coeficients_indexes=[(4, 0.5), (0, -0.5), (37, 1.0)]):
    xs = []
    for (index, coef) in coeficients_indexes:
        xs.append(mnist_data[index][0])
    x = torch.stack(xs)
    embedding = model.encoder(x)

    embedding_values = []
    e = embedding[0] * 0.0
    for i in range(len(embedding)):
        e = e + embedding[i] * coeficients_indexes[i][1]
    embedding_values.append(e)
    embedding_values = torch.stack(embedding_values)

    recons = model.decoder(embedding_values)

    plt.figure(figsize=(10, 2))
    for i, recon in enumerate(recons.detach().numpy()):
        plt.subplot(2,10,i+1)
        # plt.imshow(recon[0])
        # plt.imshow(recon[1])
        # plt.imshow(recon[2])
        PIL.Image.fromarray(np.uint8(np.swapaxes(np.swapaxes(recon, 0, 2), 0, 1) * 255)).show()

    return recons

def interpolate_img_n(coeficients_indexes=[[(mnist_data[37][0], 1.0)], [(mnist_data[4][0], 1.0), (mnist_data[0][0], -1.0)]], n=2**7):

    images = []

    for i in range(1, len(coeficients_indexes)):
        e1 = model.encoder(torch.stack([coeficients_indexes[i][0][0]])) * 0.0
        e2 = e1
        for j in range(len(coeficients_indexes[i-1])):
            embedding = model.encoder(torch.stack([coeficients_indexes[i-1][j][0]]))[0]
            e1 = e1 + embedding * coeficients_indexes[i-1][j][1]

        for j in range(len(coeficients_indexes[i])):
            embedding = model.encoder(torch.stack([coeficients_indexes[i][j][0]]))[0]
            e2 = e2 + embedding * coeficients_indexes[i][j][1]

        embedding_values = []
        n=n-1
        for j in range(n+1):
            embedding_values.append(e2 * (j/n) + e1 * (n-j)/n)
        embedding_values = torch.stack(embedding_values)

        recons = model.decoder(embedding_values)

        images.extend([recons[j] for j in range(len(recons))])

    return images

# interpolate(0, 1)
# interpolate(1, 10)
# interpolate(4, 5)
# interpolate(20, 30)

interpolate_img()
to_video(to_images(interpolate_img_n()))
# to_video(to_images(repeat_image(mnist_data[0][0])))
# to_video(to_images(repeat_image(mnist_data[37][0])))

