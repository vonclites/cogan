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
parser.add_argument('--logs_folder', default='logs', type=str,
                    help='folder within ckpt_dir to save the logs')
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

writer = SummaryWriter(log_dir=os.path.join(model_dir, args.logs_folder))

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

adversarial_photo_loss = torch.nn.MSELoss().to(gpu0)
adversarial_print_loss = torch.nn.MSELoss().to(gpu1)
l2_photo_loss = torch.nn.MSELoss().to(gpu0)
l2_print_loss = torch.nn.MSELoss().to(gpu1)

train_loader = get_dataset(args)

print(len(train_loader))

Tensor = torch.cuda.FloatTensor
patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)

for epoch in range(500):
    print(epoch)

    loss_m_d = AverageMeter()
    loss_m_g = AverageMeter()
    acc_m = AverageMeter()

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

        gen_adversarial_photo_loss = adversarial_photo_loss(pred_fake_photo,
                                                            valid.to(gpu0)).to(cpu)
        gen_adversarial_print_loss = adversarial_print_loss(pred_fake_print,
                                                            valid.to(gpu1)).to(cpu)

        gen_l2_photo_loss = l2_photo_loss(fake_photo, img_photo).to(cpu)
        gen_l2_print_loss = l2_print_loss(fake_print, img_print).to(cpu)

        adversarial_loss = (
            gen_adversarial_photo_loss +
            gen_adversarial_print_loss
        ) / 2

        l2_loss = (
            gen_l2_photo_loss +
            gen_l2_print_loss
        ) / 2

        dist = ((y_photo.to(cpu) - y_print.to(cpu)) ** 2).sum(1)

        margin = torch.ones_like(dist, device=cpu) * args.margin

        identity_loss = (lbl * dist + (1 - lbl) * F.relu(margin - dist)).mean()
        loss = (identity_loss +
                adversarial_loss * args.delta_1 +
                l2_loss * args.delta_2)

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
        pred_real_photo = disc_photo(img_photo)
        pred_fake_photo = disc_photo(fake_photo.detach())

        pred_real_print = disc_print(img_print)
        pred_fake_print = disc_print(fake_print.detach())

        discriminator_real_photo_loss = adversarial_photo_loss(pred_real_photo,
                                                               valid.to(gpu0)).to(cpu)
        discriminator_real_print_loss = adversarial_print_loss(pred_real_print,
                                                               valid.to(gpu1)).to(cpu)
        discriminator_fake_photo_loss = adversarial_photo_loss(pred_fake_photo,
                                                               fake.to(gpu0)).to(cpu)
        discriminator_fake_print_loss = adversarial_print_loss(pred_fake_print,
                                                               fake.to(gpu1)).to(cpu)

        d_loss = (
            discriminator_real_photo_loss +
            discriminator_real_print_loss +
            discriminator_fake_photo_loss +
            discriminator_fake_print_loss
        ) / 4

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        loss_m_d.update(d_loss.item())

        if step % 10 == 0:
            print(
                'Epoch: %02d, iter: %02d/%02d, D loss: %.4f, G loss: %.4f, acc: %.4f'
                % (epoch, step, len(train_loader), loss_m_d.avg, loss_m_g.avg, acc_m.avg)
            )
        if step % 100 == 0:
            writer.add_scalar('loss/generator/adversarial_photo',
                              gen_adversarial_photo_loss,
                              step)
            writer.add_scalar('loss/generator/adversarial_print',
                              gen_adversarial_print_loss,
                              step)
            writer.add_scalar('loss/generator/adversarial',
                              adversarial_loss,
                              step)
            writer.add_scalar('loss/generator/l2_photo', gen_l2_photo_loss, step)
            writer.add_scalar('loss/generator/l2_print', gen_l2_print_loss, step)
            writer.add_scalar('loss/generator/l2', l2_loss, step)
            writer.add_scalar('loss/generator/identity', identity_loss, step)
            writer.add_scalar('loss/generator/total', loss, step)
            writer.add_scalar('loss/discriminator/real_photo',
                              discriminator_real_photo_loss,
                              step)
            writer.add_scalar('loss/discriminator/real_print',
                              discriminator_real_print_loss,
                              step)
            writer.add_scalar('loss/discriminator/fake_photo',
                              discriminator_fake_photo_loss,
                              step)
            writer.add_scalar('loss/discriminator/fake_print',
                              discriminator_fake_print_loss,
                              step)
            writer.add_scalar('loss/discriminator/total', d_loss, step)
            writer.add_scalar('accuracy', acc, step)

            real_photo_grid = torchvision.utils.make_grid(img_photo[:25], nrow=5)
            real_print_grid = torchvision.utils.make_grid(img_print[:25], nrow=5)
            fake_photo_grid = torchvision.utils.make_grid(fake_photo[:25], nrow=5)
            fake_print_grid = torchvision.utils.make_grid(fake_print[:25], nrow=5)

            writer.add_image('real photo', real_photo_grid, step)
            writer.add_image('real print', real_print_grid, step)
            writer.add_image('fake photo', fake_photo_grid, step)
            writer.add_image('fake print', fake_print_grid, step)

    state = {
        'net_photo': net_photo.state_dict(),
        'net_print': net_print.state_dict(),
        'disc_photo': disc_photo.state_dict(),
        'disc_print': disc_print.state_dict(),
        'optimizer': optimizer_G.state_dict()
    }

    data_split = os.path.splitext(os.path.basename(args.valid_classes_filepath))[0]
    ckpt_dir = os.path.join(model_dir, data_split)
    ckpt_fp = os.path.join(ckpt_dir, 'checkpoint.pt')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(state, ckpt_fp)
    print('\nModel saved!\n')
