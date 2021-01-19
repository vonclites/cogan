import os
import argparse
import torchvision
import numpy as np
import random as python_random
from torch.utils.tensorboard import SummaryWriter

from cogan.dataset import get_dataset
from cogan.utils import *
from cogan.model import *

GPU0 = torch.device('cuda:0')
GPU1 = torch.device('cuda:1')
CPU = torch.device('cpu')

Tensor = torch.cuda.FloatTensor


class Model(object):
    def __init__(self, ckpt_dir, feat_dim):
        self.adversarial_photo_loss = torch.nn.MSELoss().to(GPU0)
        self.adversarial_print_loss = torch.nn.MSELoss().to(GPU1)
        self.l2_photo_loss = torch.nn.MSELoss().to(GPU0)
        self.l2_print_loss = torch.nn.MSELoss().to(GPU1)

        self.writer = SummaryWriter(log_dir=ckpt_dir)

        self.net_photo = Mapper(prenet='resnet18', outdim=feat_dim)
        self.net_photo.to(GPU0)

        self.net_print = Mapper(prenet='resnet18', outdim=feat_dim)
        self.net_print.to(GPU1)

        self.disc_photo = Discriminator(in_channels=3)
        self.disc_photo.to(GPU0)

        self.disc_print = Discriminator(in_channels=3)
        self.disc_print.to(GPU1)

        self.optimizer_g = torch.optim.Adam(
            params=list(self.net_photo.parameters()) + list(self.net_print.parameters()),
            lr=1e-4,
            weight_decay=1e-4
        )
        self.optimizer_d = torch.optim.Adam(
            params=list(self.disc_photo.parameters()) + list(self.disc_print.parameters()),
            lr=1e-4,
            weight_decay=1e-4
        )

        self.g_loss_meter = AverageMeter()
        self.g_id_loss_meter = AverageMeter()
        self.g_adv_photo_loss_meter = AverageMeter()
        self.g_adv_print_loss_meter = AverageMeter()
        self.g_adv_loss_raw_meter = AverageMeter()
        self.g_adv_loss_meter = AverageMeter()
        self.g_l2_photo_loss_meter = AverageMeter()
        self.g_l2_print_loss_meter = AverageMeter()
        self.g_l2_loss_raw_meter = AverageMeter()
        self.g_l2_loss_meter = AverageMeter()

        self.d_loss_meter = AverageMeter()
        self.d_real_photo_loss_meter = AverageMeter()
        self.d_real_print_loss_meter = AverageMeter()
        self.d_fake_photo_loss_meter = AverageMeter()
        self.d_fake_print_loss_meter = AverageMeter()

        self.accuracy_meter = AverageMeter()

    def train_epoch(self, train_loader, delta_1, delta_2, margin, epoch):
        self.net_photo.train()
        self.net_print.train()
        self.disc_photo.train()
        self.disc_print.train()

        img_photo, img_print = None, None
        fake_photo, fake_print = None, None
        global_step = None

        for step, (img_photo, img_print, lbl) in enumerate(train_loader):
            global_step = epoch * len(train_loader) + step + 1
            fake_photo, fake_print = self._train_step(
                img_photo=img_photo,
                img_print=img_print,
                lbl=lbl,
                delta_1=delta_1,
                delta_2=delta_2,
                margin=margin)
            if step % 10 == 0:
                print(
                    'Epoch: %02d, iter: %02d/%02d, D loss: %.4f, G loss: %.4f, acc: %.4f'
                    % (epoch, step, len(train_loader),
                       self.d_loss_meter.avg,
                       self.g_loss_meter.avg,
                       self.accuracy_meter.avg)
                )
        real_photo_grid = torchvision.utils.make_grid(img_photo[:16], nrow=4)
        real_print_grid = torchvision.utils.make_grid(img_print[:16], nrow=4)
        fake_photo_grid = torchvision.utils.make_grid(fake_photo[:16], nrow=4)
        fake_print_grid = torchvision.utils.make_grid(fake_print[:16], nrow=4)

        self.writer.add_image('real photo', real_photo_grid, global_step)
        self.writer.add_image('real print', real_print_grid, global_step)
        self.writer.add_image('fake photo', fake_photo_grid, global_step)
        self.writer.add_image('fake print', fake_print_grid, global_step)

        self.writer.add_scalar('generator/adversarial_photo',
                               self.g_adv_photo_loss_meter.avg,
                               global_step)
        self.writer.add_scalar('generator/adversarial_print',
                               self.g_adv_print_loss_meter.avg,
                               global_step)
        self.writer.add_scalar('generator/adversarial_raw',
                               self.g_adv_loss_raw_meter.avg,
                               global_step)
        self.writer.add_scalar('generator/adversarial',
                               self.g_adv_loss_meter.avg,
                               global_step)
        self.writer.add_scalar('generator/l2_photo', self.g_l2_photo_loss_meter.avg, global_step)
        self.writer.add_scalar('generator/l2_print', self.g_l2_print_loss_meter.avg, global_step)
        self.writer.add_scalar('generator/l2_raw', self.g_l2_loss_raw_meter.avg, global_step)
        self.writer.add_scalar('generator/l2', self.g_l2_loss_meter.avg, global_step)
        self.writer.add_scalar('generator/identity', self.g_id_loss_meter.avg, global_step)
        self.writer.add_scalar('loss/generator', self.g_loss_meter.avg, global_step)

        self.writer.add_scalar('discriminator/real_photo',
                               self.d_real_photo_loss_meter.avg,
                               global_step)
        self.writer.add_scalar('discriminator/real_print',
                               self.d_real_print_loss_meter.avg,
                               global_step)
        self.writer.add_scalar('discriminator/fake_photo',
                               self.d_fake_photo_loss_meter.avg,
                               global_step)
        self.writer.add_scalar('discriminator/fake_print',
                               self.d_fake_print_loss_meter.avg,
                               global_step)
        self.writer.add_scalar('loss/discriminator', self.d_loss_meter.avg, global_step)

        self.writer.add_scalar('accuracy', self.accuracy_meter.avg, global_step)
        self.writer.flush()

    def _train_step(self, img_photo, img_print, lbl, delta_1, delta_2, margin):
        bs = img_photo.size(0)
        lbl = lbl.type(torch.float)
        lbl = lbl.to(CPU)
        img_photo = img_photo.to(GPU0)
        img_print = img_print.to(GPU1)

        valid = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(bs, 1).fill_(0.0), requires_grad=False)

        fake_photo, y_photo = self.net_photo(img_photo)
        fake_print, y_print = self.net_print(img_print)

        pred_fake_photo = self.disc_photo(fake_photo)
        pred_fake_print = self.disc_print(fake_print)

        # """"""""""""""""""
        # "   Generator    "
        # """"""""""""""""""
        g_adv_photo_loss = self.adversarial_photo_loss(pred_fake_photo,
                                                       valid.to(GPU0)).to(CPU)
        g_adv_print_loss = self.adversarial_print_loss(pred_fake_print,
                                                       valid.to(GPU1)).to(CPU)
        self.g_adv_photo_loss_meter.update(g_adv_photo_loss.item())
        self.g_adv_print_loss_meter.update(g_adv_print_loss.item())

        g_adv_loss_raw = (
                                 g_adv_photo_loss +
                                 g_adv_print_loss
                         ) / 2
        self.g_adv_loss_raw_meter.update(g_adv_loss_raw.item())

        g_adv_loss = g_adv_loss_raw * delta_1
        self.g_adv_loss_meter.update(g_adv_loss.item())

        g_l2_photo_loss = self.l2_photo_loss(fake_photo, img_photo).to(CPU)
        g_l2_print_loss = self.l2_print_loss(fake_print, img_print).to(CPU)

        self.g_l2_photo_loss_meter.update(g_l2_photo_loss.item())
        self.g_l2_print_loss_meter.update(g_l2_print_loss.item())

        g_l2_loss_raw = (
                                g_l2_photo_loss +
                                g_l2_print_loss
                        ) / 2
        self.g_l2_loss_raw_meter.update(g_l2_loss_raw.item())

        g_l2_loss = g_l2_loss_raw * delta_2
        self.g_l2_loss_meter.update(g_l2_loss.item())

        dist = ((y_photo.to(CPU) - y_print.to(CPU)) ** 2).sum(1)

        margin = torch.ones_like(dist, device=CPU) * margin

        g_id_loss = (lbl * dist + (1 - lbl) * F.relu(margin - dist)).mean()
        self.g_id_loss_meter.update(g_id_loss.item())

        g_loss = g_id_loss + g_adv_loss + g_l2_loss
        self.g_loss_meter.update(g_loss.item())

        self.optimizer_g.zero_grad()
        g_loss.backward()
        self.optimizer_g.step()

        acc = (dist < margin).type(torch.float)
        acc = (acc == lbl).type(torch.float)
        acc = acc.mean()
        self.accuracy_meter.update(acc)

        # """"""""""""""""""
        # " Discriminator  "
        # """"""""""""""""""
        pred_real_photo = self.disc_photo(img_photo)
        pred_fake_photo = self.disc_photo(fake_photo.detach())

        pred_real_print = self.disc_print(img_print)
        pred_fake_print = self.disc_print(fake_print.detach())

        d_real_photo_loss = self.adversarial_photo_loss(pred_real_photo, valid.to(GPU0)).to(CPU)
        d_real_print_loss = self.adversarial_print_loss(pred_real_print, valid.to(GPU1)).to(CPU)
        d_fake_photo_loss = self.adversarial_photo_loss(pred_fake_photo, fake.to(GPU0)).to(CPU)
        d_fake_print_loss = self.adversarial_print_loss(pred_fake_print, fake.to(GPU1)).to(CPU)

        self.d_real_photo_loss_meter.update(d_real_photo_loss.item())
        self.d_real_print_loss_meter.update(d_real_print_loss.item())
        self.d_fake_photo_loss_meter.update(d_fake_photo_loss.item())
        self.d_fake_print_loss_meter.update(d_fake_print_loss.item())

        d_loss = (
                         d_real_photo_loss +
                         d_real_print_loss +
                         d_fake_photo_loss +
                         d_fake_print_loss
                 ) / 4
        self.d_loss_meter.update(d_loss.item())

        self.optimizer_d.zero_grad()
        d_loss.backward()
        self.optimizer_d.step()
        return fake_photo, fake_print

    def _reset_meters(self):
        self.g_loss_meter.reset()
        self.g_id_loss_meter.reset()
        self.g_adv_photo_loss_meter.reset()
        self.g_adv_print_loss_meter.reset()
        self.g_adv_loss_raw_meter.reset()
        self.g_adv_loss_meter.reset()
        self.g_l2_photo_loss_meter.reset()
        self.g_l2_print_loss_meter.reset()
        self.g_l2_loss_raw_meter.reset()
        self.g_l2_loss_meter.reset()
        self.d_loss_meter.reset()
        self.d_real_photo_loss_meter.reset()
        self.d_real_print_loss_meter.reset()
        self.d_fake_photo_loss_meter.reset()
        self.d_fake_print_loss_meter.reset()
        self.accuracy_meter.reset()


def parse_args():
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
    parser.add_argument('--valid_train_classes_fp', type=str,
                        help='text file of class labels to include in training dataset')
    parser.add_argument('--valid_test_classes_fp', type=str,
                        help='text file of class labels to include in test (validation) dataset')
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
    parser.add_argument('--basenet', default='resnet18', type=str,
                        help='e.g., resnet50, resnext50, resnext101'
                             'and their wider variants, resnet50x4')
    parser.add_argument('-d', '--feat_dim', default=128, type=int,
                        help='feature dimension for contrastive loss')
    return parser.parse_args()


def run():
    args = parse_args()

    model_name = '{}_{}_{}_{}_{}'.format(
        args.basenet, args.margin, args.delta_1, args.delta_2, args.feat_dim
    )
    model_dir = os.path.join(args.model_dir, model_name)
    data_split = os.path.splitext(os.path.basename(args.valid_train_classes_fp))[0]
    ckpt_dir = os.path.join(model_dir, data_split)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader = get_dataset(args)
    print(len(train_loader))

    patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)

    model = Model(ckpt_dir, args.feat_dim)

    for epoch in range(500):
        model.train_epoch(train_loader, args.delta_1, args.delta_2, args.margin, epoch)

        state = {
            'net_photo': model.net_photo.state_dict(),
            'net_print': model.net_print.state_dict(),
            'disc_photo': model.disc_photo.state_dict(),
            'disc_print': model.disc_print.state_dict(),
            'optimizer': model.optimizer_g.state_dict()
        }

        ckpt_fp = os.path.join(ckpt_dir, 'checkpoint.pt')
        torch.save(state, ckpt_fp)
        print('\nModel saved!\n')

    model.writer.close()


if __name__ == '__main__':
    python_random.seed(62484)
    np.random.seed(62484)
    torch.manual_seed(62484)

    run()
