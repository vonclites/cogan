import os
import argparse
import torchvision
import numpy as np
import random as python_random
from torch.utils.tensorboard import SummaryWriter

from cogan.dataset import get_dataset
from cogan.utils import *
from cogan.model import *


parser = argparse.ArgumentParser(description='Contrastive view')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--margin', default=100, type=int, help='batch size')
parser.add_argument('--delta_1', default=1, type=float, help='Adversarial Coefficient')
parser.add_argument('--delta_2', default=1, type=float, help='L2 Coefficient')
parser.add_argument('--nir_dir', type=str,
                    default='/home/hulk1/data/periocular/hk/images/dev/NIR',
                    help='path to data')
parser.add_argument('--vis_dir', type=str,
                    default='/home/hulk1/data/periocular/hk/images/dev/VIS',
                    help='path to data')
parser.add_argument('--valid_classes_filepath', type=str,
                    help='text file of class labels to include in dataset')
parser.add_argument('--model_dir', type=str,
                    help='base directory path in which individual runs will be saved')
parser.add_argument('--nir_mean_fp', type=str,
                    default='/home/hulk1/data/periocular/hk/stats/nir_mean.txt',
                    help='Path to file containing channel-wise image statistic')
parser.add_argument('--nir_std_fp', type=str,
                    default='/home/hulk1/data/periocular/hk/stats/nir_std.txt',
                    help='Path to file containing channel-wise image statistic')
parser.add_argument('--vis_mean_fp', type=str,
                    default='/home/hulk1/data/periocular/hk/stats/vis_mean.txt',
                    help='Path to file containing channel-wise image statistic')
parser.add_argument('--vis_std_fp', type=str,
                    default='/home/hulk1/data/periocular/hk/stats/vis_std.txt',
                    help='Path to file containing channel-wise image statistic')

# model setup
parser.add_argument('--basenet', default='resnet18', type=str,
                    help='e.g., resnet50, resnext50, resnext101'
                         'and their wider variants, resnet50x4')
parser.add_argument('-d', '--feat_dim', default=128, type=int,
                    help='feature dimension for contrastive loss')

args = parser.parse_args()

python_random.seed(62484)
np.random.seed(62484)
torch.manual_seed(62484)

gpu0 = torch.device('cuda:0')
gpu1 = torch.device('cuda:1')
cpu = torch.device('cpu')

model_name = '{}_{}_{}_{}_{}'.format(
        args.basenet, args.margin, args.delta_1, args.delta_2, args.feat_dim
    )
model_dir = os.path.join(args.model_dir, model_name)
data_split = os.path.splitext(os.path.basename(args.valid_classes_filepath))[0]
ckpt_dir = os.path.join(model_dir, data_split)
os.makedirs(ckpt_dir, exist_ok=True)

writer = SummaryWriter(log_dir=ckpt_dir)

net_photo = Mapper(prenet='resnet18', outdim=args.feat_dim)
# net_photo = UNet(feat_dim=args.feat_dim)
net_photo.to(gpu0)
net_photo.train()

net_print = Mapper(prenet='resnet18', outdim=args.feat_dim)
# net_print = UNet(feat_dim=args.feat_dim)
net_print.to(gpu1)
net_print.train()

disc_photo = Discriminator(in_channels=3)
disc_photo.to(gpu0)
disc_photo.train()

disc_print = Discriminator(in_channels=3)
disc_print.to(gpu1)
disc_print.train()

# for i, p in net_print.named_parameters():

#     print(i, p.size())
# exit()

optimizer_g = torch.optim.Adam(
    params=list(net_photo.parameters()) + list(net_print.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)
optimizer_d = torch.optim.Adam(
    params=list(disc_photo.parameters()) + list(disc_print.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

adversarial_photo_loss = torch.nn.MSELoss().to(gpu0)
adversarial_print_loss = torch.nn.MSELoss().to(gpu1)
l2_photo_loss = torch.nn.MSELoss().to(gpu0)
l2_print_loss = torch.nn.MSELoss().to(gpu1)

train_loader = get_dataset(args)

print(len(train_loader))

Tensor = torch.cuda.FloatTensor
patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)

g_loss_meter = AverageMeter()
g_id_loss_meter = AverageMeter()
g_adv_photo_loss_meter = AverageMeter()
g_adv_print_loss_meter = AverageMeter()
g_adv_loss_raw_meter = AverageMeter()
g_adv_loss_meter = AverageMeter()
g_l2_photo_loss_meter = AverageMeter()
g_l2_print_loss_meter = AverageMeter()
g_l2_loss_raw_meter = AverageMeter()
g_l2_loss_meter = AverageMeter()

d_loss_meter = AverageMeter()
d_real_photo_loss_meter = AverageMeter()
d_real_print_loss_meter = AverageMeter()
d_fake_photo_loss_meter = AverageMeter()
d_fake_print_loss_meter = AverageMeter()

accuracy_meter = AverageMeter()

for epoch in range(500):
    print(epoch)
    for step, (img_photo, img_print, lbl) in enumerate(train_loader):
        # plot_tensor([img_photo[0], img_print[0]])
        bs = img_photo.size(0)
        lbl = lbl.type(torch.float)

        img_photo = img_photo.to(gpu0)
        img_print = img_print.to(gpu1)
        lbl = lbl.to(cpu)

        valid = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(bs, 1).fill_(0.0), requires_grad=False)

        # """"""""""""""""""
        # "   Generator    "
        # """"""""""""""""""

        fake_photo, y_photo = net_photo(img_photo)
        fake_print, y_print = net_print(img_print)

        pred_fake_photo = disc_photo(fake_photo)
        pred_fake_print = disc_print(fake_print)

        g_adv_photo_loss = adversarial_photo_loss(pred_fake_photo,
                                                  valid.to(gpu0)).to(cpu)
        g_adv_print_loss = adversarial_print_loss(pred_fake_print,
                                                  valid.to(gpu1)).to(cpu)
        g_adv_photo_loss_meter.update(g_adv_photo_loss.item())
        g_adv_print_loss_meter.update(g_adv_print_loss.item())

        g_adv_loss_raw = (
            g_adv_photo_loss +
            g_adv_print_loss
        ) / 2
        g_adv_loss_raw_meter.update(g_adv_loss_raw.item())

        g_adv_loss = g_adv_loss_raw * args.delta_1
        g_adv_loss_meter.update(g_adv_loss.item())

        g_l2_photo_loss = l2_photo_loss(fake_photo, img_photo).to(cpu)
        g_l2_print_loss = l2_print_loss(fake_print, img_print).to(cpu)

        g_l2_photo_loss_meter.update(g_l2_photo_loss.item())
        g_l2_print_loss_meter.update(g_l2_print_loss.item())

        g_l2_loss_raw = (
            g_l2_photo_loss +
            g_l2_print_loss
        ) / 2
        g_l2_loss_raw_meter.update(g_l2_loss_raw.item())

        g_l2_loss = g_l2_loss_raw * args.delta_2
        g_l2_loss_meter.update(g_l2_loss.item())

        dist = ((y_photo.to(cpu) - y_print.to(cpu)) ** 2).sum(1)

        margin = torch.ones_like(dist, device=cpu) * args.margin

        g_id_loss = (lbl * dist + (1 - lbl) * F.relu(margin - dist)).mean()
        g_id_loss_meter.update(g_id_loss.item())

        g_loss = g_id_loss + g_adv_loss + g_l2_loss
        g_loss_meter.update(g_loss.item())

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        acc = (dist < args.margin).type(torch.float)
        acc = (acc == lbl).type(torch.float)
        acc = acc.mean()
        accuracy_meter.update(acc)

        # """"""""""""""""""
        # " Discriminator  "
        # """"""""""""""""""
        pred_real_photo = disc_photo(img_photo)
        pred_fake_photo = disc_photo(fake_photo.detach())

        pred_real_print = disc_print(img_print)
        pred_fake_print = disc_print(fake_print.detach())

        d_real_photo_loss = adversarial_photo_loss(pred_real_photo, valid.to(gpu0)).to(cpu)
        d_real_print_loss = adversarial_print_loss(pred_real_print, valid.to(gpu1)).to(cpu)
        d_fake_photo_loss = adversarial_photo_loss(pred_fake_photo, fake.to(gpu0)).to(cpu)
        d_fake_print_loss = adversarial_print_loss(pred_fake_print, fake.to(gpu1)).to(cpu)

        d_real_photo_loss_meter.update(d_real_photo_loss.item())
        d_real_print_loss_meter.update(d_real_print_loss.item())
        d_fake_photo_loss_meter.update(d_fake_photo_loss.item())
        d_fake_print_loss_meter.update(d_fake_print_loss.item())

        d_loss = (
            d_real_photo_loss +
            d_real_print_loss +
            d_fake_photo_loss +
            d_fake_print_loss
        ) / 4
        d_loss_meter.update(d_loss.item())

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        if step % 10 == 0:
            print(
                'Epoch: %02d, iter: %02d/%02d, D loss: %.4f, G loss: %.4f, acc: %.4f'
                % (epoch, step, len(train_loader),
                   d_loss_meter.avg,
                   g_loss_meter.avg,
                   accuracy_meter.avg)
            )
        if (step + 1) % len(train_loader) == 0:
            global_step = epoch * len(train_loader) + step + 1
            writer.add_scalar('generator/adversarial_photo',
                              g_adv_photo_loss_meter.avg,
                              global_step)
            writer.add_scalar('generator/adversarial_print',
                              g_adv_print_loss_meter.avg,
                              global_step)
            writer.add_scalar('generator/adversarial_raw',
                              g_adv_loss_raw_meter.avg,
                              global_step)
            writer.add_scalar('generator/adversarial',
                              g_adv_loss_meter.avg,
                              global_step)
            writer.add_scalar('generator/l2_photo', g_l2_photo_loss_meter.avg, global_step)
            writer.add_scalar('generator/l2_print', g_l2_print_loss_meter.avg, global_step)
            writer.add_scalar('generator/l2_raw', g_l2_loss_raw_meter.avg, global_step)
            writer.add_scalar('generator/l2', g_l2_loss_meter.avg, global_step)
            writer.add_scalar('generator/identity', g_id_loss_meter.avg, global_step)
            writer.add_scalar('loss/generator', g_loss_meter.avg, global_step)

            writer.add_scalar('discriminator/real_photo',
                              d_real_photo_loss_meter.avg,
                              global_step)
            writer.add_scalar('discriminator/real_print',
                              d_real_print_loss_meter.avg,
                              global_step)
            writer.add_scalar('discriminator/fake_photo',
                              d_fake_photo_loss_meter.avg,
                              global_step)
            writer.add_scalar('discriminator/fake_print',
                              d_fake_print_loss_meter.avg,
                              global_step)
            writer.add_scalar('loss/discriminator', d_loss_meter.avg, global_step)

            writer.add_scalar('accuracy', acc, global_step)

            real_photo_grid = torchvision.utils.make_grid(img_photo[:25], nrow=5)
            real_print_grid = torchvision.utils.make_grid(img_print[:25], nrow=5)
            fake_photo_grid = torchvision.utils.make_grid(fake_photo[:25], nrow=5)
            fake_print_grid = torchvision.utils.make_grid(fake_print[:25], nrow=5)

            writer.add_image('real photo', real_photo_grid, global_step)
            writer.add_image('real print', real_print_grid, global_step)
            writer.add_image('fake photo', fake_photo_grid, global_step)
            writer.add_image('fake print', fake_print_grid, global_step)
            writer.flush()

            g_loss_meter.reset()
            g_id_loss_meter.reset()
            g_adv_photo_loss_meter.reset()
            g_adv_print_loss_meter.reset()
            g_adv_loss_raw_meter.reset()
            g_adv_loss_meter.reset()
            g_l2_photo_loss_meter.reset()
            g_l2_print_loss_meter.reset()
            g_l2_loss_raw_meter.reset()
            g_l2_loss_meter.reset()
            d_loss_meter.reset()
            d_real_photo_loss_meter.reset()
            d_real_print_loss_meter.reset()
            d_fake_photo_loss_meter.reset()
            d_fake_print_loss_meter.reset()
            accuracy_meter.reset()

    state = {
        'net_photo': net_photo.state_dict(),
        'net_print': net_print.state_dict(),
        'disc_photo': disc_photo.state_dict(),
        'disc_print': disc_print.state_dict(),
        'optimizer': optimizer_g.state_dict()
    }

    ckpt_fp = os.path.join(ckpt_dir, 'checkpoint.pt')
    torch.save(state, ckpt_fp)
    print('\nModel saved!\n')

writer.close()
