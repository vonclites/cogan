import os
import argparse
import torchvision
import numpy as np
import random as python_random
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from cogan_demo.dataset import get_dataset
from cogan_demo.model import *


def extract_bn_features(self):
    feature = []
    for i in range(len(self.bn_hook)):
        feature.append(self.bn_hook[i].delta_mean)
        feature.append(self.bn_hook[i].delta_var)

    feature = torch.cat(feature, 1)
    return feature


def parameters_fc(self):
    return [self.fc1.weight] + [self.fc1.bias] + [self.fc2.weight] + [self.fc2.bias]


def parameters_net(self):
    return self.basemodel.parameters()


parser = argparse.ArgumentParser(description='Contrastive view')
parser.add_argument('--batch_size', default=96, type=int, help='batch size')
parser.add_argument('--margin', default=100, type=int, help='batch size')
parser.add_argument('--delta_1', default=1, type=float, help='Delta 1 HyperParameter')
parser.add_argument('--delta_2', default=1, type=float, help='Delta 2 HyperParameter')
parser.add_argument('--nir_dir', type=str,
                    default='/home/hulk2/data/periocular/hk/images/dev/NIR',
                    help='path to data')
parser.add_argument('--vis_dir', type=str,
                    default='/home/hulk2/data/periocular/hk/images/dev/VIS',
                    help='path to data')
parser.add_argument('--valid_classes_filepath', type=str,
                    help='text file of class labels to include in dataset')
parser.add_argument('--ckpt_dir', type=str,
                    help='path to save the data')
parser.add_argument('--nir_mean_fp', type=str,
                    default='/home/hulk2/data/periocular/hk/images/dev/nir_mean.txt',
                    help='Path to file containing channelwise image statistic')
parser.add_argument('--nir_std_fp', type=str,
                    default='/home/hulk2/data/periocular/hk/images/dev/nir_std.txt',
                    help='Path to file containing channelwise image statistic')
parser.add_argument('--vis_mean_fp', type=str,
                    default='/home/hulk2/data/periocular/hk/images/dev/vis_mean.txt',
                    help='Path to file containing channelwise image statistic')
parser.add_argument('--vis_std_fp', type=str,
                    default='/home/hulk2/data/periocular/hk/images/dev/vis_std.txt',
                    help='Path to file containing channelwise image statistic')
parser.add_argument('--results_folder', type=str,
                    help='path to save the data')

parser.add_argument('--basenet', default='resnet18', type=str,
                    help='e.g., resnet50, resnext50, resnext101'
                         'and their wider variants, resnet50x4')
parser.add_argument('--net', default='unetV1', type=str)
parser.add_argument('-d', '--feat_dim', default=128, type=int,
                    help='feature dimension for contrastive loss')


args = parser.parse_args()

python_random.seed(62484)
np.random.seed(62484)
torch.manual_seed(62484)

gpu0 = torch.device('cuda:0')
gpu1 = torch.device('cuda:1')
cpu = torch.device('cpu')

output_dir = os.path.join(args.ckpt_dir, args.results_folder)
os.makedirs(output_dir, exist_ok=True)

state = torch.load(os.path.join(args.ckpt_dir, 'checkpoint.pt'))

net_photo = Mapper(prenet='resnet18', outdim=args.feat_dim)

net_photo.to(gpu0)
net_photo.load_state_dict(state['net_photo'])
net_photo.eval()

net_print = Mapper(prenet='resnet18', outdim=args.feat_dim)

net_print.to(gpu1)
net_print.load_state_dict(state['net_print'])
net_print.eval()

train_loader = get_dataset(args)

print(len(train_loader))

dist_l = []
lbl_l = []
for step, (img_photo, img_print, lbl) in enumerate(train_loader):
    # plot_tensor([img_photo[0], img_print[0]])
    print(step)
    bs = img_photo.size(0)
    lbl = lbl.type(torch.float)

    img_photo = img_photo.to(gpu0)
    img_print = img_print.to(gpu1)
    # lbl = lbl.to(gpu0)

    fake_photo, y_photo = net_photo(img_photo)
    fake_print, y_print = net_print(img_print)

    y_photo = y_photo.to(cpu)
    y_print = y_print.to(cpu)
    dist = ((y_photo - y_print) ** 2).sum(1)
    dist_l.append(dist.data)
    lbl_l.append((1-lbl).data)

dist = torch.cat(dist_l, 0)
lbl = torch.cat(lbl_l, 0)
dist = dist.cpu().detach().numpy()
lbl = lbl.cpu().detach().numpy()

fpr, tpr, threshold = metrics.roc_curve(lbl, dist)
roc_auc = metrics.auc(fpr, tpr)

np.save(os.path.join(output_dir, 'pr_ph_lbl_test.npy'), lbl)
np.save(os.path.join(output_dir, 'pr_ph_dist_test.npy'), dist)

torchvision.utils.save_image(img_photo[:6], os.path.join(output_dir, 'real_vis.png'))
torchvision.utils.save_image(img_print[:6], os.path.join(output_dir, 'real_nir.png'))
torchvision.utils.save_image(fake_photo[:6], os.path.join(output_dir, 'fake_vis.png'))
torchvision.utils.save_image(fake_print[:6], os.path.join(output_dir, 'fake_nir.png'))

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(os.path.join(output_dir, 'roc.png'))
plt.show()
