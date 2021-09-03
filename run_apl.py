from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import numpy as np
import wandb
import argparse
import os
import atexit
import PIL
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--apl_factor', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--save_freq', default=20, type=int)
parser.add_argument('--num_classes', default=21, type=int)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--load', default='')
parser.add_argument('--note', default='')
parser.add_argument('--apl', default=False, action='store_true')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()

os.environ["WANDB_SILENT"] = "true"

proj = "apl"

run = wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name
im_dim = 520


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)


set_seed(0)


############################### LOAD DATASET

def transform(x, y):
    xt = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), transforms.Resize(im_dim, interpolation=PIL.Image.NEAREST)])
    yt = transforms.Compose([transforms.ToTensor(), transforms.Resize(im_dim, interpolation=PIL.Image.NEAREST)])
    x, y = xt(x), (yt(y) * 255).squeeze(0).long()
    y[y == 255] = 0
    return x, y


batch_size = args.batch_size

trainset = datasets.VOCSegmentation(root='/', year='2007', image_set='trainval', download=True, transforms=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, worker_init_fn=np.random.seed(0))

testset = datasets.VOCSegmentation(root='/', year='2007', image_set='test', download=True, transforms=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, worker_init_fn=np.random.seed(0))

wandb.log({'Train samples': len(trainset), 'Test samples': len(testset)})

classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

############################### MODEL DEF

net = torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=True, num_classes=args.num_classes, aux_loss=None)

############################### VIS

palette = [(0, 0, 0),
           (128, 0, 0),
           (0, 128, 0),
           (128, 128, 0),
           (0, 0, 128),
           (128, 0, 128),
           (0, 128, 128),
           (128, 128, 128),
           (64, 0, 0),
           (192, 0, 0),
           (64, 128, 0),
           (192, 128, 0),
           (64, 0, 128),
           (192, 0, 128),
           (64, 128, 128),
           (192, 128, 128),
           (0, 64, 0),
           (128, 64, 0),
           (0, 192, 0),
           (128, 192, 0),
           (0, 64, 128)]

palette_dict = dict(zip([tuple([n] * 3) for n in range(args.num_classes)], palette))


def colour_map(output):
    output = output.unsqueeze(-1)
    cmap = torch.cat([output] * 3, dim=-1)
    cmap = cmap.cpu().numpy()

    for b in range(cmap.shape[0]):
        for x in range(cmap.shape[1]):
            for y in range(cmap.shape[2]):
                cmap[b, x, y, :] = np.array(palette_dict[tuple(cmap[b, x, y, :])])
    return cmap


def vis_activations(activations):
    l = 0
    imgs = {}
    for layer in activations:
        l += 1
        # first batch only
        fb_layer = layer[0].detach().cpu()

        all_filters = torch.cat([i for i in fb_layer], dim=1)
        imgs['Layer {}'.format(l)] = wandb.Image(all_filters)

    wandb.log(imgs)


############################### HELPERS


def exit_handler():
    print('Finishing run...')
    run.finish()
    print('Done!')


atexit.register(exit_handler)


def get_activations(m):
    # TODO
    return torch.zeros((1, 1, 1, 1))


def concat_with_pad(x_list):
    x_list = [x.reshape(x.shape[0], x.shape[1], -1) for x in x_list]
    max_dim = max([x.shape[1] for x in x_list])
    x_list = [torch.nn.functional.pad(x, (0, 0, max_dim - x.shape[1], 0)) for x in x_list]
    x_list = torch.cat(x_list, dim=-1)
    return x_list


def dist(actvs):
    actvs = concat_with_pad(actvs)
    penalty = torch.sum(torch.sigmoid(torch.sum(actvs, dim=2)))
    return penalty


def scores(o, t):
    score_arr = np.zeros((args.num_classes, 5))

    for cls_num in range(args.num_classes):
        output = o[:, cls_num]
        target = (t == cls_num).float()

        with torch.no_grad():
            tp = torch.sum(target * output)
            tn = torch.sum((1 - target) * (1 - output))
            fp = torch.sum((1 - target) * output)
            fn = torch.sum(target * (1 - output))

            p = tp / (tp + fp + 0.0001)
            r = tp / (tp + fn + 0.0001)
            f1 = 2 * p * r / (p + r + 0.0001)
            acc = (tp + tn) / (tp + tn + fp + fn)
            iou = tp / ((torch.sum(output + target) - tp) + 0.0001)

        score_arr[cls_num] = np.array([p.item(), r.item(), f1.item(), acc.item(), iou.item()])
    return score_arr


############################### TRAIN & TEST FNS

def run_epoch(epoch, model, data_loader, apl=False, vis=False, train=False):
    if train:
        model.train()
        mode = 'Train'
    else:
        model.eval()
        mode = 'Val'
    scores_list = ['prec', 'rec', 'f1', 'acc', 'iou']

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)['out']

        activations = get_activations(model)
        loss = criterion(output, target)
        if apl:
            aploss = dist(activations) / data.shape[0] * args.apl_factor
            total_loss = loss + aploss
        else:
            aploss = 0.0
            total_loss = loss
        total_loss.backward()
        optimizer.step()

        batch_scores = scores(torch.round(output), target)
        print(scores_list)
        print(np.mean(batch_scores, axis=1))
        results = {'{}/Epoch'.format(mode): epoch, '{}/Loss'.format(mode): loss, '{}/APL'.format(mode): aploss, '{}/Total Loss'.format(mode): total_loss}
        results.update({'{}/avg_iou'.format(mode): np.mean(np.nonzero(batch_scores[:, 4]))})
        results.update({'{}/avg_acc'.format(mode): np.mean(np.nonzero(batch_scores[:, 3]))})

        if batch_idx % args.save_freq == 0:
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tAPL: {:.8f}'.format(mode,
                                                                                     epoch, batch_idx * len(data), len(train_loader.dataset),
                                                                                     100. * batch_idx / len(train_loader), loss.data.item(), aploss))

            results['Inputs'] = [wandb.Image(d) for d in data[:min(10, args.batch_size)]]
            results['Outputs'] = [wandb.Image(o) for o in colour_map(output[:min(10, args.batch_size)].argmax(dim=1))]
            results['Targets'] = [wandb.Image(t) for t in colour_map(target[:min(10, args.batch_size)])]

            if vis:
                vis_activations(activations)
        wandb.log(results)

    if train:
        torch.save(model.state_dict(), 'params/' + params_id + '.pth')
    return model


############################### TRAIN & TEST MODEL

run = wandb.init(project=proj, entity="jessicamarycooper", config=args)
args = wandb.config
params_id = wandb.run.name

model = net.to(device)

if args.load is not '':
    state_dict = torch.load('params/' + args.load + ".pth")
    model.load_state_dict(state_dict)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.CrossEntropyLoss()

if args.train:
    for epoch in range(1, args['epochs'] + 1):
        model = run_epoch(epoch, model, train_loader, train=True, apl=args.apl, vis=False)

model = run_epoch(1, model, test_loader, train=False, apl=args.apl, vis=False)

run.finish()
