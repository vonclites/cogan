import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.utils import save_image

from cogan_demo.dataset import get_dataset
from cogan_demo.utils import *
from cogan_demo.model import *


#################################################################################################
def comp_bnfeatures(self):
    # extract batchnorm features
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
parser.add_argument('--net', default='unetV1', type=str)
parser.add_argument('-d', '--feat_dim', default=128, type=int,
                    help='feature dimension for contrastive loss')

args = parser.parse_args()

device = torch.device('cuda:0')

output_dir = "data/" + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2) + "_" + str(args.feat_dim)
# output_dir = "data/" + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2) + "_" + args.net
os.makedirs("%s" % (output_dir), exist_ok=True)
state = torch.load('./%s/model_resnet18_%s_%s_%s_%s.pt' % (args.save_folder, str(args.margin), str(args.delta_1), str(args.delta_2), str(args.feat_dim)))
# state = torch.load('./%s/model_unetV2_%s_%s_%s.pt' % (args.save_folder, str(args.margin), str(args.delta_1), str(args.delta_2)))
# state = torch.load('./%s/model_unetV3_%s_%s_%s.pt' % (args.save_folder, str(args.margin), str(args.delta_1), str(args.delta_2)))
# state = torch.load('./%s/model_unetV4_%s_%s_%s.pt' % (args.save_folder, str(args.margin), str(args.delta_1), str(args.delta_2)))
# state = torch.load('./%s/model_unetV5_%s_%s_%s.pt' % (args.save_folder, str(args.margin), str(args.delta_1), str(args.delta_2)))
# state = torch.load('./%s/model_unetV6_%s_%s_%s.pt' % (args.save_folder, str(args.margin), str(args.delta_1), str(args.delta_2)))
# state = torch.load('./%s/model_%s_%s_%s_%s.pt' % (args.save_folder, args.net, str(args.margin), str(args.delta_1), str(args.delta_2)))

net_photo = Mapper(prenet='resnet18', outdim=args.feat_dim)
# net_photo = UNetV6(feat_dim=args.feat_dim)
# if args.net == "unetV1":
#     net_photo = UNet(feat_dim=args.feat_dim)
# if args.net == "unetV2":
#     net_photo = UNetV2(feat_dim=args.feat_dim)
# if args.net == "unetV3":
#     net_photo = UNetV3(feat_dim=args.feat_dim)
# if args.net == "unetV4":
#     net_photo = UNetV4(feat_dim=args.feat_dim)
# if args.net == "unetV5":
#     net_photo = UNetV5(feat_dim=args.feat_dim)
# if args.net == "unetV6":
#     net_photo = UNetV6(feat_dim=args.feat_dim)
net_photo.to(device)
net_photo.load_state_dict(state['net_photo'])
net_photo.eval()



net_print = Mapper(prenet='resnet18', outdim=args.feat_dim)
# net_print = UNetV6(feat_dim=args.feat_dim)
# if args.net == "unetV1":
#     net_print = UNet(feat_dim=args.feat_dim)
# if args.net == "unetV2":
#     net_print = UNetV2(feat_dim=args.feat_dim)
# if args.net == "unetV3":
#     net_print = UNetV3(feat_dim=args.feat_dim)
# if args.net == "unetV4":
#     net_print = UNetV4(feat_dim=args.feat_dim)
# if args.net == "unetV5":
#     net_print = UNetV5(feat_dim=args.feat_dim)
# if args.net == "unetV6":
#     net_print = UNetV6(feat_dim=args.feat_dim)
net_print.to(device)
net_print.load_state_dict(state['net_print'])
net_print.eval()


train_loader = get_dataset(args)

print(len(train_loader))

dist_l = []
lbl_l = []
imagesDir = "Images/" + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2) + "_" + str(args.feat_dim)
os.makedirs("%s" % (imagesDir), exist_ok=True)
for iter, (img_photo, img_print, lbl) in enumerate(train_loader):
    # plot_tensor([img_photo[0], img_print[0]])
    print(iter)
    bs = img_photo.size(0)
    lbl = lbl.type(torch.float)

    img_photo, img_print, lbl = img_photo.to(device), img_print.to(device), lbl.to(device)

    temp1 = net_photo.EncodeImage(img_photo)
    fake_img_photo = net_print.DecodeImage(temp1)
    # fake_img_photo, y_photo = net_photo(img_photo)
    # fake_img_print, y_print = net_print(img_print)

    # img_sample = torch.cat((img_photo.data, fake_img_photo.data, img_print.data), -2)
    img_sample = torch.cat((img_photo.data, fake_img_photo.data), -2)
    save_image(img_sample, "%s/%s.png" % (imagesDir, iter), nrow=bs, normalize=True)
