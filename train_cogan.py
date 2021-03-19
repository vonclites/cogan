import os
import torch
import argparse
import torchvision
import numpy as np
import random as python_random
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as func
from torch.utils.tensorboard import SummaryWriter

import backboned_unet
from cogan import utils
from cogan.dataset import get_dev_dataset
from cogan.model import Discriminator

GPU0 = torch.device('cuda:0')
GPU1 = torch.device('cuda:1')
CPU = torch.device('cpu')

Tensor = torch.cuda.FloatTensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--num_epochs', type=int, help='batch size')
    parser.add_argument('--margin', type=int, help='batch size')
    parser.add_argument('--backbone', type=str,
                        help='resnet18, resnet34, resnet50,'
                             'and their wider variants, resnet50x4')
    parser.add_argument('--adversarial_coeff', type=float,
                        help='Adversarial Loss Coefficient')
    parser.add_argument('--pixel_coeff', type=float,
                        help='Pixel-to-Pixel Loss Coefficient')
    parser.add_argument('--perceptual_coeff', type=float,
                        help='Perceptual Loss Coefficient')
    parser.add_argument('-d', '--feat_dim',type=int,
                        help='feature dimension for contrastive loss')
    parser.add_argument('--model_dir', type=str,
                        help='base directory path in which individual runs will be saved')
    parser.add_argument('--nir_dir', type=str,
                        help='path to data')
    parser.add_argument('--vis_dir', type=str,
                        help='path to data')
    parser.add_argument('--valid_train_classes_fp', type=str,
                        help='text file of class labels to include in training dataset')
    parser.add_argument('--valid_test_classes_fp', type=str,
                        help='text file of class labels to include in test (validation) dataset')
    parser.add_argument('--nir_mean_fp', type=str,
                        help='Path to file containing channel-wise image statistic')
    parser.add_argument('--nir_std_fp', type=str,
                        help='Path to file containing channel-wise image statistic')
    parser.add_argument('--vis_mean_fp', type=str,
                        help='Path to file containing channel-wise image statistic')
    parser.add_argument('--vis_std_fp', type=str,
                        help='Path to file containing channel-wise image statistic')
    return parser.parse_args()


class Model(object):
    def __init__(self, backbone, ckpt_dir, feat_dim):
        self.adversarial_photo_loss = torch.nn.MSELoss().to(GPU0)
        self.adversarial_print_loss = torch.nn.MSELoss().to(GPU1)
        self.pixel_photo_loss = torch.nn.MSELoss().to(GPU0)
        self.pixel_print_loss = torch.nn.MSELoss().to(GPU1)

        self.writer = SummaryWriter(log_dir=ckpt_dir)
        self.eval_writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'eval'))

        self.net_photo = backboned_unet.Unet(backbone_name=backbone, classes=feat_dim)
        # self.net_photo = UNet()
        # self.net_photo = Mapper()
        self.net_photo.to(GPU0)

        self.net_print = backboned_unet.Unet(backbone_name=backbone, classes=feat_dim)
        # self.net_print = UNet()
        # self.net_print = Mapper()
        self.net_print.to(GPU1)

        self.disc_photo = Discriminator(in_channels=3)
        self.disc_photo.to(GPU0)

        self.disc_print = Discriminator(in_channels=3)
        self.disc_print.to(GPU1)

        self.perceptual_net_photo = models.vgg16(pretrained=True).features.to(GPU0)
        self.perceptual_net_print = models.vgg16(pretrained=True).features.to(GPU1)
        self.perceptual_net_photo = self.perceptual_net_photo.eval()
        self.perceptual_net_print = self.perceptual_net_print.eval()
        self.perceptual_photo_loss = torch.nn.MSELoss().to(GPU0)
        self.perceptual_print_loss = torch.nn.MSELoss().to(GPU1)

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

        self.g_loss_meter = utils.AverageMeter()
        self.g_id_loss_meter = utils.AverageMeter()
        self.g_adv_photo_loss_meter = utils.AverageMeter()
        self.g_adv_print_loss_meter = utils.AverageMeter()
        self.g_adv_loss_raw_meter = utils.AverageMeter()
        self.g_adv_loss_meter = utils.AverageMeter()
        self.g_pixel_photo_loss_meter = utils.AverageMeter()
        self.g_pixel_print_loss_meter = utils.AverageMeter()
        self.g_pixel_loss_raw_meter = utils.AverageMeter()
        self.g_pixel_loss_meter = utils.AverageMeter()
        self.g_perceptual_photo_loss_meter = utils.AverageMeter()
        self.g_perceptual_print_loss_meter = utils.AverageMeter()
        self.g_perceptual_loss_raw_meter = utils.AverageMeter()
        self.g_perceptual_loss_meter = utils.AverageMeter()

        self.d_loss_meter = utils.AverageMeter()
        self.d_real_photo_loss_meter = utils.AverageMeter()
        self.d_real_print_loss_meter = utils.AverageMeter()
        self.d_fake_photo_loss_meter = utils.AverageMeter()
        self.d_fake_print_loss_meter = utils.AverageMeter()

        self.accuracy_meter = utils.AverageMeter()
        self.auc_meter = utils.AverageMeter()
        self.eer_meter = utils.AverageMeter()
        self.frr_far_01_meter = utils.AverageMeter()
        self.frr_far_10_meter = utils.AverageMeter()

    def _reset_meters(self):
        self.g_loss_meter.reset()
        self.g_id_loss_meter.reset()
        self.g_adv_photo_loss_meter.reset()
        self.g_adv_print_loss_meter.reset()
        self.g_adv_loss_raw_meter.reset()
        self.g_adv_loss_meter.reset()
        self.g_pixel_photo_loss_meter.reset()
        self.g_pixel_print_loss_meter.reset()
        self.g_pixel_loss_raw_meter.reset()
        self.g_pixel_loss_meter.reset()
        self.g_perceptual_photo_loss_meter.reset()
        self.g_perceptual_print_loss_meter.reset()
        self.g_perceptual_loss_raw_meter.reset()
        self.g_perceptual_loss_meter.reset()
        self.d_loss_meter.reset()
        self.d_real_photo_loss_meter.reset()
        self.d_real_print_loss_meter.reset()
        self.d_fake_photo_loss_meter.reset()
        self.d_fake_print_loss_meter.reset()
        self.accuracy_meter.reset()
        self.auc_meter.reset()
        self.eer_meter.reset()
        self.frr_far_01_meter.reset()
        self.frr_far_10_meter.reset()

    def train_epoch(self,
                    train_loader,
                    adversarial_coeff,
                    pixel_coeff,
                    perceptual_coeff,
                    margin,
                    epoch):
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
                adversarial_coeff=adversarial_coeff,
                pixel_coeff=pixel_coeff,
                perceptual_coeff=perceptual_coeff,
                margin=margin)
            if step % 10 == 0:
                print(
                    'Epoch: %02d, iter: %02d/%02d, D loss: %.4f, G loss: %.4f, acc: %.4f'
                    % (epoch, step, len(train_loader),
                       self.d_loss_meter.avg,
                       self.g_loss_meter.avg,
                       self.accuracy_meter.avg)
                )
        real_photo_grid = torchvision.utils.make_grid(
            tensor=img_photo[:16],
            nrow=4,
            normalize=True,
            range=(-1, 1)
        )
        real_print_grid = torchvision.utils.make_grid(
            tensor=img_print[:16],
            nrow=4,
            normalize=True,
            range=(-1, 1)
        )
        fake_photo_grid = torchvision.utils.make_grid(
            tensor=fake_photo[:16],
            nrow=4,
            normalize=True,
            range=(-1, 1)
        )
        fake_print_grid = torchvision.utils.make_grid(
            tensor=fake_print[:16],
            nrow=4,
            normalize=True,
            range=(-1, 1)
        )

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
        self.writer.add_scalar('generator/pixel_photo', self.g_pixel_photo_loss_meter.avg, global_step)
        self.writer.add_scalar('generator/pixel_print', self.g_pixel_print_loss_meter.avg, global_step)
        self.writer.add_scalar('generator/pixel_raw', self.g_pixel_loss_raw_meter.avg, global_step)
        self.writer.add_scalar('generator/pixel', self.g_pixel_loss_meter.avg, global_step)
        self.writer.add_scalar('generator/perceptual_photo', self.g_perceptual_photo_loss_meter.avg, global_step)
        self.writer.add_scalar('generator/perceptual_print', self.g_perceptual_print_loss_meter.avg, global_step)
        self.writer.add_scalar('generator/perceptual_raw', self.g_perceptual_loss_raw_meter.avg, global_step)
        self.writer.add_scalar('generator/perceptual', self.g_perceptual_loss_meter.avg, global_step)
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
        self._reset_meters()

    def _train_step(self,
                    img_photo,
                    img_print,
                    lbl,
                    adversarial_coeff,
                    pixel_coeff,
                    perceptual_coeff,
                    margin):
        bs = img_photo.size(0)
        lbl = lbl.type(torch.float)
        lbl = lbl.to(CPU)
        img_photo = img_photo.to(GPU0)
        img_print = img_print.to(GPU1)

        valid = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False)
        # valid = np.random.uniform(0.9, 1.0, size=(bs, 1))
        fake = Variable(Tensor(bs, 1).fill_(0.0), requires_grad=False)

        fake_photo, y_photo = self.net_photo(img_photo)
        fake_print, y_print = self.net_print(img_print)

        pred_fake_photo = self.disc_photo(fake_photo)
        pred_fake_print = self.disc_print(fake_print)

        photo_features = self.perceptual_net_photo(img_photo)
        fake_photo_features = self.perceptual_net_photo(fake_photo)

        print_features = self.perceptual_net_print(img_print)
        fake_print_features = self.perceptual_net_print(fake_print)

        # """"""""""""""""""
        # "   Generator    "
        # """"""""""""""""""

        # Adversarial Loss
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

        g_adv_loss = g_adv_loss_raw * adversarial_coeff
        self.g_adv_loss_meter.update(g_adv_loss.item())

        # Pixel Loss
        g_pixel_photo_loss = self.pixel_photo_loss(fake_photo, img_photo).to(CPU)
        g_pixel_print_loss = self.pixel_print_loss(fake_print, img_print).to(CPU)

        self.g_pixel_photo_loss_meter.update(g_pixel_photo_loss.item())
        self.g_pixel_print_loss_meter.update(g_pixel_print_loss.item())

        g_pixel_loss_raw = (g_pixel_photo_loss + g_pixel_print_loss) / 2
        self.g_pixel_loss_raw_meter.update(g_pixel_loss_raw.item())

        g_pixel_loss = g_pixel_loss_raw * pixel_coeff
        self.g_pixel_loss_meter.update(g_pixel_loss.item())

        # Perceptual Loss
        g_perceptual_photo_loss = self.perceptual_photo_loss(
            input=fake_photo_features,
            target=photo_features
        ).to(CPU)
        g_perceptual_print_loss = self.perceptual_print_loss(
            input=fake_print_features,
            target=print_features
        ).to(CPU)

        self.g_perceptual_photo_loss_meter.update(g_perceptual_photo_loss.item())
        self.g_perceptual_print_loss_meter.update(g_perceptual_print_loss.item())

        g_perceptual_loss_raw = (g_perceptual_photo_loss + g_perceptual_print_loss) / 2
        self.g_perceptual_loss_raw_meter.update(g_perceptual_loss_raw.item())

        g_perceptual_loss = g_perceptual_loss_raw * perceptual_coeff
        self.g_perceptual_loss_meter.update(g_perceptual_loss.item())

        # Identity Loss
        dist = ((y_photo.to(CPU) - y_print.to(CPU)) ** 2).sum(1)
        margin = torch.ones_like(dist, device=CPU) * margin

        g_id_loss = (lbl * dist + (1 - lbl) * func.relu(margin - dist)).mean()
        self.g_id_loss_meter.update(g_id_loss.item())

        # Total Loss
        g_loss = g_id_loss + g_adv_loss + g_pixel_loss + g_perceptual_loss
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

    def eval(self, test_loader, global_step):
        self.net_photo.eval()
        self.net_print.eval()
        self.disc_photo.eval()
        self.disc_print.eval()

        with torch.no_grad():
            dist_l = []
            lbl_l = []
            for img_photo, img_print, lbl in test_loader:
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

                y_photo = y_photo.to(CPU)
                y_print = y_print.to(CPU)
                dist = ((y_photo - y_print) ** 2).sum(1)
                dist_l.append(dist.data)
                lbl_l.append((1 - lbl).data)

            dist = torch.cat(dist_l, 0)
            lbl = torch.cat(lbl_l, 0)
            dist = dist.cpu().detach().numpy()
            lbl = lbl.cpu().detach().numpy()

            fpr, tpr, threshold = metrics.roc_curve(lbl, dist)
            auc = metrics.auc(fpr, tpr)
            eer = utils.eer(fpr, tpr)
            frr = 1 - tpr

            self.auc_meter.update(auc)
            self.eer_meter.update(eer)
            self.frr_far_10_meter.update(frr[np.searchsorted(fpr, .1)])
            self.frr_far_01_meter.update(frr[np.searchsorted(fpr, .01)])

            self.eval_writer.add_scalar('auc', self.auc_meter.avg, global_step)
            self.eval_writer.add_scalar('eer', self.eer_meter.avg, global_step)
            self.eval_writer.add_scalar('frr@far=10%', self.frr_far_10_meter.avg, global_step)
            self.eval_writer.add_scalar('frr@far=1%', self.frr_far_01_meter.avg, global_step)

            fig = plt.figure()
            ax: plt.Axes = fig.add_subplot()
            ax.plot(fpr, tpr, 'b', label='AUC = {:.2f} | EER = {:.2f}'.format(auc, eer))
            ax.legend(loc='lower right')
            ax.plot([0, 1], [0, 1], 'r--')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_ylabel('True Positive Rate')
            ax.set_xlabel('False Positive Rate')
            self.eval_writer.add_figure('ROC Curve', fig, global_step=global_step)
            self.eval_writer.flush()
            self._reset_meters()


def run(args):
    model_name = '{}_{}_{}_{}_{}'.format(
        args.backbone,
        args.margin,
        args.adversarial_coeff,
        args.pixel_coeff,
        args.perceptual_coeff,
        args.feat_dim
    )
    model_dir = os.path.join(args.model_dir, model_name)
    data_split = os.path.splitext(os.path.basename(args.valid_train_classes_fp))[0]
    ckpt_dir = os.path.join(model_dir, data_split)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader = get_dev_dataset(
        batch_size=args.batch_size,
        vis_dir=args.vis_dir,
        nir_dir=args.nir_dir,
        valid_classes_fp=args.valid_train_classes_fp,
        vis_mean_fp=args.vis_mean_fp,
        vis_std_fp=args.vis_std_fp,
        nir_mean_fp=args.nir_mean_fp,
        nir_std_fp=args.nir_std_fp
    )
    test_loader = get_dev_dataset(
        batch_size=args.batch_size,
        vis_dir=args.vis_dir,
        nir_dir=args.nir_dir,
        valid_classes_fp=args.valid_test_classes_fp,
        vis_mean_fp=args.vis_mean_fp,
        vis_std_fp=args.vis_std_fp,
        nir_mean_fp=args.nir_mean_fp,
        nir_std_fp=args.nir_std_fp
    )
    steps_per_epoch = len(train_loader)

    model = Model(args.backbone, ckpt_dir, args.feat_dim)

    for epoch in range(args.num_epochs):
        model.train_epoch(
            train_loader=train_loader,
            adversarial_coeff=args.adversarial_coeff,
            pixel_coeff=args.pixel_coeff,
            perceptual_coeff=args.perceptual_coeff,
            margin=args.margin,
            epoch=epoch
        )
        model.eval(test_loader,
                   global_step=epoch * steps_per_epoch + steps_per_epoch)
        state = {
            'epoch': epoch,
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
    model.eval_writer.close()


if __name__ == '__main__':
    python_random.seed(62484)
    np.random.seed(62484)
    torch.manual_seed(62484)
    run(parse_args())
