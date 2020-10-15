import os
import argparse

from cogan_demo.dataset import get_dataset
from cogan_demo.utils import *
from cogan_demo.model import *


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
                    default='./checkpoint/',
                    help='path to save the data')

# model setup
parser.add_argument('--basenet', default='resnet18', type=str,
                    help='e.g., resnet50, resnext50, resnext101'
                         'and their wider variants, resnet50x4')
parser.add_argument('-d', '--feat_dim', default=128, type=int,
                    help='feature dimension for contrastive loss')

args = parser.parse_args()

os.makedirs(args.ckpt_dir, exist_ok=True)

device = torch.device('cuda:0')

net_photo = Mapper(prenet='resnet18', outdim=args.feat_dim)
# net_photo = UNet(feat_dim=args.feat_dim)
net_photo.to(device)
net_photo.train()

net_print = Mapper(prenet='resnet18', outdim=args.feat_dim)
# net_print = UNet(feat_dim=args.feat_dim)
net_print.to(device)
net_print.train()

disc_photo = Discriminator(in_channels=3)
disc_photo.to(device)
disc_photo.train()

disc_print = Discriminator(in_channels=3)
disc_print.to(device)
disc_print.train()

# for i, p in net_print.named_parameters():

#     print(i, p.size())
# exit()

optimizer_G = torch.optim.Adam(
    params=list(net_photo.parameters()) + list(net_print.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)
optimizer_D = torch.optim.Adam(
    params=list(disc_photo.parameters()) + list(disc_print.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

adversarial_loss = torch.nn.MSELoss().to(device)
L2_Norm_loss = torch.nn.MSELoss().to(device)

train_loader = get_dataset(args)

print(len(train_loader))

Tensor = torch.cuda.FloatTensor
patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)

# output_dir = str(args.ckpt_dir) + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2)
# os.makedirs("%s" % (output_dir), exist_ok=True)

for epoch in range(500):
    print(epoch)

    loss_m_d = AverageMeter()
    loss_m_g = AverageMeter()
    acc_m = AverageMeter()

    for step, (img_photo, img_print, lbl) in enumerate(train_loader):
        # plot_tensor([img_photo[0], img_print[0]])
        bs = img_photo.size(0)
        lbl = lbl.type(torch.float)

        img_photo = img_photo.to(device)
        img_print = img_print.to(device)
        lbl = lbl.to(device)

        # This work is for my first disc
        # valid = Variable(Tensor(np.ones((img_photo.size(0), *patch))), requires_grad=False)
        # fake = Variable(Tensor(np.zeros((img_photo.size(0), *patch))), requires_grad=False)
        valid = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(bs, 1).fill_(0.0), requires_grad=False)

        # """"""""""""""""""
        # "   Generator    "
        # """"""""""""""""""

        fake_photo, y_photo = net_photo(img_photo)
        fake_print, y_print = net_print(img_print)

        # This work is for my first disc
        # pred_fake_photo = disc_photo(fake_photo, img_photo)
        # pred_fake_print = disc_print(fake_print, img_print)
        pred_fake_photo = disc_photo(fake_photo)
        pred_fake_print = disc_print(fake_print)

        loss_GAN = (adversarial_loss(pred_fake_photo, valid) +
                    adversarial_loss(pred_fake_print, valid)) / 2
        loss_L2 = (L2_Norm_loss(fake_photo, img_photo) +
                   L2_Norm_loss(fake_print, img_print)) / 2

        dist = ((y_photo - y_print) ** 2).sum(1)

        margin = torch.ones_like(dist, device=device) * args.margin

        loss = lbl * dist + (1 - lbl) * F.relu(margin - dist)
        loss = loss.mean() + loss_GAN * args.delta_1 + loss_L2 * args.delta_2

        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()

        acc = (dist < args.margin).type(torch.float)
        acc = (acc == lbl).type(torch.float)
        acc = acc.mean()
        acc_m.update(acc)

        loss_m_g.update(loss.item())

        # """"""""""""""""""
        # " Discriminator  "
        # """"""""""""""""""
        # This work is for my first disc
        # pred_real_photo = disc_photo(img_photo, img_photo)
        # pred_fake_photo = disc_photo(fake_photo.detach(), img_photo)
        #
        # pred_real_print = disc_print(img_print, img_print)
        # pred_fake_print = disc_print(fake_print.detach(), img_print)

        # This work is for my second disc
        pred_real_photo = disc_photo(img_photo)
        pred_fake_photo = disc_photo(fake_photo.detach())

        pred_real_print = disc_print(img_print)
        pred_fake_print = disc_print(fake_print.detach())

        d_loss = (
            adversarial_loss(pred_real_print, valid)
            + adversarial_loss(pred_real_photo, valid)
            + adversarial_loss(pred_fake_print, fake)
            + adversarial_loss(pred_fake_photo, fake)
        ) / 4

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        loss_m_d.update(d_loss.item())

        if step % 10 == 0:
            print(
                'epoch: %02d, iter: %02d/%02d, D loss: %.4f, G loss: %.4f, acc: %.4f'
                % (epoch, step, len(train_loader), loss_m_d.avg, loss_m_g.avg, acc_m.avg)
            )

    state = {
        'net_photo': net_photo.state_dict(),
        'net_print': net_print.state_dict(),
        'disc_photo': disc_photo.state_dict(),
        'disc_print': disc_print.state_dict(),
        'optimizer': optimizer_G.state_dict()
    }

    split = os.path.splitext(os.path.basename(args.valid_classes_filepath))[0]
    model_name = '{}_{}_{}_{}_{}_{}'.format(
        split, args.basenet, args.margin, args.delta_1, args.delta_2, args.feat_dim
    )
    torch.save(state, os.path.join(args.ckpt_dir, model_name + '.pt'))
    print('\nmodel saved!\n')
