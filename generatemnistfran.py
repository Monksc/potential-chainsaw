#!/usr/bin/env python3

from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import numpy as np
import functions as f
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

# Step 1 Get Data

def foo():
    a = ToTensor()
    def f(b):
        noise = np.float32(np.round(np.random.random((1, 28, 28)) - 0.4) * np.random.random((1, 28, 28)) * 0.1)
        # noise = np.float32(np.ones((1,28,28)))
        # print(a(b) + noise)
        return a(b) + noise
    # return noise
    return f

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = foo(),
    download = True,
)
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = foo(),
)

# train_data.data = train_data.train_data / 255.0
# test_data.data = test_data.data

print(train_data.train_labels[:5])

from torch.utils.data import DataLoader
loaders = {
    'train' : torch.utils.data.DataLoader(train_data,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=1),

    'test'  : torch.utils.data.DataLoader(test_data,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=1),
}

# Step 2 Random Noise

data_set_len = len(loaders['train'].dataset.data)
#loaders['train'].dataset.data = loaders['train'].dataset.data / 255.0
#loaders['test'].dataset.data = loaders['test'].dataset.data / 255.0
#loaders['train'].dataset.data += np.round(np.random.random((data_set_len, 28, 28)) - 0.4) * np.random.random((data_set_len, 28, 28)) * 0.1

for i in range(2):
    f.to_image(np.array(loaders['train'].dataset[i][0][0]))

# Step 3 Make model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization

cnn = CNN()
print(cnn)

loss_func = nn.CrossEntropyLoss()

from torch import optim
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)

# Step 4 Train
from torch.autograd import Variable
num_epochs = 1
def train(num_epochs, cnn, loaders):

    cnn.train()

    # Train the model
    total_step = len(loaders['train'])

    loss_values = []
    for epoch in range(num_epochs):
        batch_loss = 0.0
        count = 0
        for i, (images, labels) in enumerate(loaders['train']):

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            batch_loss += loss.item()
            count += 1

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        loss_values.append(batch_loss / count)
    plt.plot(np.array(loss_values), 'r')
    plt.show()

train(num_epochs, cnn, loaders)

# Step 4.1 Test Model And do other stuff with model

def test(data_to_test):
    # Test the model
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        print(data_to_test)
        for images, labels in data_to_test:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)

test(loaders['test'])

sample = next(iter(loaders['test']))
imgs, lbls = sample

actual_number = lbls[:10].numpy()

test_output, last_layer = cnn(imgs[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(f'Prediction number: {pred_y}')
print(f'Actual number: {actual_number}')

cnn(torch.Tensor(np.random.random((100, 1, 28, 28))))

# Step 5 Generate Images

import fran

n = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
im_seed = torch.rand((n, 1, 28, 28))
targets = torch.Tensor(np.array(list(np.eye(10))*(n//10)))
EPS = 40
ITS = 1000
ALP = 1
loss_list=[]
iter_list=[]
frans_data = fran.PGD_attack(cnn, device, im_seed, targets, EPS, ALP, ITS, rand_start=False, loss_list=loss_list, iter_list=iter_list, num_im=5)[0]
out = cnn(frans_data)[0].detach().numpy()
guess = np.eye(10)[np.argmax(out, axis=1)]
print("Frans Images Got ", (n - (targets - guess).sum()) / n, "correct")

while True:
    #im_seed = torch.rand((10, 1, 28, 28))
    im_seed = torch.Tensor(np.array(loaders['train'].dataset.data[:10]).reshape(10, 1, 28, 28))
    targets = torch.Tensor(np.eye(10))
    frans_data = fran.PGD_attack(cnn, device, im_seed, targets, EPS, ALP, ITS, rand_start=False, loss_list=loss_list, iter_list=iter_list, num_im=5)
    images = frans_data[0]
    outputs = cnn(images)
    for i in range(len(images)):
        print('MAKING A', i, outputs[0][i])
        f.to_image(np.array(images[i][0]))
        input()


