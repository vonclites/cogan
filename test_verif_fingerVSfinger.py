from cogan_demo.dataset import get_dataset
import argparse
# from utils import *
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from cogan_demo.model import Mapper
from cogan_demo.utils import *


#################################################################################################
# class Mapper(nn.Module):
#     def __init__(self, prenet='resnet18', outdim=128):
#         super(Mapper, self).__init__()
#
#         model = getattr(models, prenet)(pretrained=False)
#
#         self.model = list(model.children())[:-1]
#
#         self.backbone = nn.Sequential(*self.model)
#
#         if prenet == 'resnet18':
#             nfc = 512
#         elif prenet == 'resnet50':
#             nfc = 2048
#
#         self.fc1 = nn.Linear(nfc, outdim)
#         # self.fc2 = nn.Linear(1024, outdim)
#
#     def forward(self, x):
#         bs = x.size(0)
#         y = self.backbone(x)
#         y = y.view(bs, -1)
#         output = self.fc1(y)
#         return output
#
#
# def comp_bnfeatures(self):
#     # extract batchnorm features
#     feature = []
#     for i in range(len(self.bn_hook)):
#         feature.append(self.bn_hook[i].delta_mean)
#         feature.append(self.bn_hook[i].delta_var)
#
#     feature = torch.cat(feature, 1)
#     return feature
#
#
# def parameters_fc(self):
#     return [self.fc1.weight] + [self.fc1.bias] + [self.fc2.weight] + [self.fc2.bias]
#
#
# def parameters_net(self):
#     return self.basemodel.parameters()


#################################################################################################


parser = argparse.ArgumentParser(description='Contrastive view')
parser.add_argument('--batch_size', default=96, type=int, help='batch size')
parser.add_argument('--margin', default=5, type=int, help='batch size')
parser.add_argument('--delta_1', default=1, type=float, help='Delta 1 HyperParameter')
parser.add_argument('--delta_2', default=1, type=float, help='Delta 2 HyperParameter')
parser.add_argument('--photo_folder', type=str,
                    default='../clean_13_as_test/photo/',
                    help='path to data')
parser.add_argument('--print_folder', type=str,
                    default='../clean_13_as_test/print/',
                    help='path to data')

parser.add_argument('--save_folder', type=str,
                    default='./checkpoint/',
                    help='path to save the data')

# model setup
parser.add_argument('--basenet', default='resnet18', type=str,
                    help='e.g., resnet50, resnext50, resnext101'
                         'and their wider variants, resnet50x4')
parser.add_argument('-d', '--feat_dim', default=128, type=int,
                    help='feature dimension for contrastive loss')

args = parser.parse_args()

device = torch.device('cuda:0')

state = torch.load('./checkpoint/model_resnet18_100_1.0_1.0.pt')

net_print = Mapper(prenet='resnet18')
net_print.to(device)
net_print.load_state_dict(state['net_print'])
net_print.eval()

train_loader = get_dataset(args)

print(len(train_loader))

dist_l = []
lbl_l = []
for iter, (img_photo, img_print, lbl) in enumerate(train_loader):
    # plot_tensor([img_photo[0], img_print[0]])
    print(iter)
    bs = img_photo.size(0)
    lbl = lbl.type(torch.float)

    img_photo, img_print, lbl = img_photo.to(device), img_print.to(device), lbl.to(device)

    _, y_photo = net_print(img_photo)
    _, y_print = net_print(img_print)

    dist = ((y_photo - y_print) ** 2).sum(1)
    dist_l.append(dist.data)
    lbl_l.append((1 - lbl).data)

dist = torch.cat(dist_l, 0)
lbl = torch.cat(lbl_l, 0)
dist = dist.cpu().detach().numpy()
lbl = lbl.cpu().detach().numpy()

import sklearn.metrics as metrics


import numpy as np

np.save('./data/pr_pr_pix2pix_verifier3_lbl.npy', lbl)
np.save('./data/pr_pr_pix2pix_verifier3_dist.npy', dist)

fpr, tpr, threshold = metrics.roc_curve(lbl, dist)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='Train, AUC = %0.6f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
