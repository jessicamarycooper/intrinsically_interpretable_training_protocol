from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import itertools
import matplotlib.pyplot as plt
import wandb
import argparse
import os
import atexit

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--load', default='')
parser.add_argument('--note', default='')
parser.add_argument('--cuda', default=False)
parser.add_argument('--apl', default=True)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["WANDB_SILENT"] = "true"

proj = "cell_classification"

run = wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name

############################### LOAD DATASET

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ])),
        batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=args.batch_size, shuffle=False)


############################### MODEL DEF

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        activations = []
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        activations.append(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        activations.append(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        activations.append(x)
        x = self.fc2(x)
        activations.append(x)
        return F.log_softmax(x, dim=1), activations


############################### HELPERS


def exit_handler():
    print('Finishing run...')
    run.finish()
    print('Done!')


atexit.register(exit_handler)


def vis_activations(activations):
    l = 0
    for layer in activations:
        l += 1
        # first batch only
        imgs = {}
        fb_layer = layer[0].detach().numpy()
        if len(fb_layer.shape) > 1:
            imgs['Layer {}'.format(l)] = [wandb.Image(i) for i in fb_layer]
        else:
            imgs['Layer {}'.format(l)] = [wandb.Image(np.expand_dims(fb_layer, axis=0))]


def trace_backward(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                # print('Tensor with grad found:', tensor)
                # print(' - gradient:', tensor.grad)
                # print()
            except AttributeError as e:
                trace_backward(n[0])


def concat_with_pad(x_list):
    x_list = [x.reshape(x.shape[0], x.shape[1], -1) for x in x_list]
    max_dim = max([x.shape[1] for x in x_list])
    x_list = [torch.nn.functional.pad(x, (0, 0, max_dim - x.shape[1], 0)) for x in x_list]
    x_list = torch.cat(x_list, dim=-1)
    return x_list


def normalise(x):
    return x - x.min() / x.max()


def dist(actvs):
    actvs = concat_with_pad(actvs)
    penalty = torch.sum(torch.sigmoid(torch.sum(actvs, dim=2)))
    return penalty


############################### TRAIN & TEST FNS

def train(epoch, model, apl=False, vis=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, activations = model(data)
        loss = F.nll_loss(output, target)
        if apl:
            aploss = dist(activations) * 0.001
            loss = loss + aploss
        else:
            aploss = 0.0
        loss.backward()
        optimizer.step()

        wandb.log({'Train Epoch': epoch, ' Train Loss': loss, 'Train APL': aploss})

        if batch_idx % args.save_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tAPL: {:.8f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item(), aploss))
            if vis:
                vis_activations(activations)
    torch.save(model.state_dict(), 'params/' + params_id + '.pth')
    return model


def test(model, vis=False):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_ix, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output, activations = model(data)

        if vis and batch_ix % args.save_freq == 0:
            vis_activations(activations)

        test_loss += F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        wandb.log({'Val Loss': test_loss})

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


############################### TRAIN & TEST MODEL

run = wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name

model = Net()

if args.load is not '':
    state_dict = torch.load('params/' + args.load + ".pth", map_location=torch.device(device))
    model.load_state_dict(state_dict)


optimizer = optim.SGD(model.parameters(), lr=args.lr)

if args.train:
    for epoch in range(1, args['epochs'] + 1):
        model = train(epoch, model, apl=args.apl, vis=False)
test(model, vis=True)

run.finish()

