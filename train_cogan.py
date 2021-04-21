import os
import torch
import argparse
import itertools
import torchvision
import numpy as np
import random as python_random
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from torchvision import models
from torch.nn import functional as func
from torch.utils.tensorboard import SummaryWriter

import backboned_unet
from cogan import utils
from cogan.dataset import get_dev_dataset, get_test_dataset
from cogan.model import Discriminator
from cogan.ddp.ddp_functions import ddp_setup, add_ddp_args


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--num_epochs', type=int, help='number of epochs to train')
    parser.add_argument('--margin', type=int, help='batch size')
    parser.add_argument('--backbone', type=str,
                        help='resnet18, resnet34, resnet50,'
                             'and their wider variants, resnet50x4')
    parser.add_argument('--identity_coeff', type=float,
                        help='Identity Loss Coefficient')
    parser.add_argument('--adversarial_coeff', type=float,
                        help='Adversarial Loss Coefficient')
    parser.add_argument('--pixel_coeff', type=float,
                        help='Pixel-to-Pixel Loss Coefficient')
    parser.add_argument('--perceptual_coeff', type=float,
                        help='Perceptual Loss Coefficient')
    parser.add_argument('--feat_dim', type=int,
                        help='feature dimension for contrastive loss')
    parser.add_argument('--dist_measure', type=str,
                        help='Either "cos" or "l2"')
    parser.add_argument('--positive_prob', type=float,
                        help='Chance to generate a genuine training pair')
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
    parser.add_argument('--fixed_seed', type=int, default=62484)
    return parser


def set_seeds(fixed_seed):
    python_random.seed(fixed_seed)
    np.random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)


class Model(object):
    def __init__(self, backbone, ckpt_dir, feat_dim):
        self.adversarial_photo_loss = torch.nn.MSELoss()
        self.adversarial_print_loss = torch.nn.MSELoss()
        self.pixel_photo_loss = torch.nn.MSELoss()
        self.pixel_print_loss = torch.nn.MSELoss()

        self.writer = SummaryWriter(log_dir=ckpt_dir)
        self.eval_writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'eval'))

        self.net_vis = backboned_unet.Unet(backbone_name=backbone, classes=feat_dim)
        self.net_nir = backboned_unet.Unet(backbone_name=backbone, classes=feat_dim)

        self.disc_photo = Discriminator(in_channels=3)
        self.disc_print = Discriminator(in_channels=3)

        self.perceptual_net_photo = models.vgg16(pretrained=True).features
        self.perceptual_net_print = models.vgg16(pretrained=True).features
        self.perceptual_net_photo = self.perceptual_net_photo.eval()
        self.perceptual_net_print = self.perceptual_net_print.eval()
        self.perceptual_photo_loss = torch.nn.MSELoss()
        self.perceptual_print_loss = torch.nn.MSELoss()

        self.optimizer_g = torch.optim.Adam(
            params=list(self.net_vis.parameters()) + list(self.net_nir.parameters()),
            lr=1e-4,
            weight_decay=1e-4
        )
        self.optimizer_d = torch.optim.Adam(
            params=list(self.disc_photo.parameters()) + list(self.disc_print.parameters()),
            lr=1e-4,
            weight_decay=1e-4
        )

        self.g_loss_meter = utils.AverageMeter()
        self.g_id_loss_raw_meter = utils.AverageMeter()
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
        self.g_id_loss_raw_meter.reset()
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
                    identity_coeff,
                    adversarial_coeff,
                    pixel_coeff,
                    perceptual_coeff,
                    margin,
                    dist_measure,
                    epoch):
        self.net_vis.train()
        self.net_nir.train()
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
                identity_coeff=identity_coeff,
                adversarial_coeff=adversarial_coeff,
                pixel_coeff=pixel_coeff,
                perceptual_coeff=perceptual_coeff,
                margin=margin,
                dist_measure=dist_measure
            )
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
                    identity_coeff,
                    adversarial_coeff,
                    pixel_coeff,
                    perceptual_coeff,
                    margin,
                    dist_measure):
        bs = img_photo.size(0)
        lbl = lbl.type(torch.float)

        valid = torch.ones(bs, requires_grad=False)
        # valid = np.random.uniform(0.9, 1.0, size=(bs, 1))
        fake = torch.zeros(bs, requires_grad=False)

        fake_photo, y_vis = self.net_vis(img_photo)
        fake_print, y_nir = self.net_nir(img_print)

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
        g_adv_photo_loss = self.adversarial_photo_loss(pred_fake_photo, valid)
        g_adv_print_loss = self.adversarial_print_loss(pred_fake_print, valid)
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
        g_pixel_photo_loss = self.pixel_photo_loss(fake_photo, img_photo)
        g_pixel_print_loss = self.pixel_print_loss(fake_print, img_print)

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
        )
        g_perceptual_print_loss = self.perceptual_print_loss(
            input=fake_print_features,
            target=print_features
        )

        self.g_perceptual_photo_loss_meter.update(g_perceptual_photo_loss.item())
        self.g_perceptual_print_loss_meter.update(g_perceptual_print_loss.item())

        g_perceptual_loss_raw = (g_perceptual_photo_loss + g_perceptual_print_loss) / 2
        self.g_perceptual_loss_raw_meter.update(g_perceptual_loss_raw.item())

        g_perceptual_loss = g_perceptual_loss_raw * perceptual_coeff
        self.g_perceptual_loss_meter.update(g_perceptual_loss.item())

        # dist = None
        # Identity Loss
        if dist_measure == 'l2':
            distance = ((y_vis - y_nir) ** 2).sum(1)
        elif dist_measure == 'cos':
            cos = torch.nn.CosineSimilarity(dim=1)
            distance = cos(y_vis, y_nir)
        else:
            raise ValueError('dist_measure must be either "l2" or "cos".')

        margin = torch.ones_like(distance) * margin

        g_id_loss_raw = (lbl * distance + (1 - lbl) * func.relu(margin - distance)).mean()
        self.g_id_loss_raw_meter.update(g_id_loss_raw.item())
        g_id_loss = g_id_loss_raw * identity_coeff
        self.g_id_loss_meter.update(g_id_loss.item())

        # Total Loss
        g_loss = g_id_loss + g_adv_loss + g_pixel_loss + g_perceptual_loss
        self.g_loss_meter.update(g_loss.item())

        self.optimizer_g.zero_grad()
        g_loss.backward()
        self.optimizer_g.step()

        acc = torch.less(distance, margin).float()
        acc = torch.equal(acc, lbl).float().mean()
        self.accuracy_meter.update(acc)

        # """"""""""""""""""
        # " Discriminator  "
        # """"""""""""""""""
        pred_real_photo = self.disc_photo(img_photo)
        pred_fake_photo = self.disc_photo(fake_photo.detach())

        pred_real_print = self.disc_print(img_print)
        pred_fake_print = self.disc_print(fake_print.detach())

        d_real_photo_loss = self.adversarial_photo_loss(pred_real_photo, valid)
        d_real_print_loss = self.adversarial_print_loss(pred_real_print, valid)
        d_fake_photo_loss = self.adversarial_photo_loss(pred_fake_photo, fake)
        d_fake_print_loss = self.adversarial_print_loss(pred_fake_print, fake)

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

    def eval(self, vis_loader, nir_loader, dist_measure, global_step):
        self.net_vis.eval()
        self.net_nir.eval()
        self.disc_photo.eval()
        self.disc_print.eval()

        with torch.no_grad():
            vis_labels = []
            vis_features = []
            nir_labels = []
            nir_features = []

            for images, labels in vis_loader:
                labels = labels.type(torch.float)

                _, features = self.net_vis(images)
                vis_features.append(features)
                vis_labels.append(labels)

            for images, labels in nir_loader:
                labels = labels.type(torch.float)

                _, features = self.net_nir(images)
                nir_features.append(features)
                nir_labels.append(labels)

            vis_features = torch.cat(vis_features, 0)
            vis_labels = torch.cat(vis_labels, 0)
            nir_features = torch.cat(nir_features, 0)
            nir_labels = torch.cat(nir_labels, 0)

            classes = set(vis_labels.numpy())
            pairs = []
            for class_id in classes:
                vis_labels_idx = [i
                                  for i, sample in enumerate(vis_labels)
                                  if sample == class_id]
                nir_labels_idx = [i
                                  for i, sample in enumerate(nir_labels)
                                  if sample == class_id]
                combinations = itertools.combinations(range(len(vis_labels_idx)), r=2)
                pairs += [(vis_labels_idx[vis_idx], nir_labels_idx[nir_idx], 1)
                          for vis_idx, nir_idx in combinations]
                vis_labels_idx = [i
                                  for i, sample in enumerate(vis_labels)
                                  if sample != class_id]
                pairs += [(vis_idx, nir_idx, 0)
                          for nir_idx in nir_labels_idx
                          for vis_idx in vis_labels_idx]

            pairs = np.array(pairs)
            vis_features = vis_features[pairs[:, 0]]
            nir_features = nir_features[pairs[:, 1]]

            # Identity Loss
            if dist_measure == 'l2':
                dist = ((vis_features - nir_features) ** 2).sum(1)
            elif dist_measure == 'cos':
                cos = torch.nn.CosineSimilarity(dim=1)
                dist = cos(vis_features, nir_features)
            else:
                raise ValueError('dist_measure must be either "l2" or "cos".')

            labels = 1 - pairs[:, 2]

            fpr, tpr, threshold = metrics.roc_curve(labels, dist)
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


def main(gpu, args):
    set_seeds(args.fixed_seed)
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    model_name = '{}_m{}_id{}_ad{}_pi{}_pe{}_f{}'.format(
        args.backbone,
        args.margin,
        args.identity_coeff,
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
        nir_std_fp=args.nir_std_fp,
        positive_prob=args.positive_prob
    )
    vis_test_loader = None
    nir_test_loader = None
    if args.valid_test_classes_fp:
        vis_test_loader, nir_test_loader = get_test_dataset(
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
            identity_coeff=args.identity_coeff,
            adversarial_coeff=args.adversarial_coeff,
            pixel_coeff=args.pixel_coeff,
            perceptual_coeff=args.perceptual_coeff,
            margin=args.margin,
            dist_measure=args.dist_measure,
            epoch=epoch
        )
        if args.valid_test_classes_fp and epoch != 0:
            model.eval(
                vis_loader=vis_test_loader,
                nir_loader=nir_test_loader,
                dist_measure=args.dist_measure,
                global_step=epoch * steps_per_epoch + steps_per_epoch
            )
        state = {
            'epoch': epoch,
            'net_photo': model.net_vis.state_dict(),
            'net_print': model.net_nir.state_dict(),
            'disc_photo': model.disc_photo.state_dict(),
            'disc_print': model.disc_print.state_dict(),
            'optimizer': model.optimizer_g.state_dict()
        }

        ckpt_fp = os.path.join(ckpt_dir, 'checkpoint.pt')
        torch.save(state, ckpt_fp)
        print('\nModel saved!\n')

    model.writer.close()
    model.eval_writer.close()


def distributed_run():
    parser = create_argparser()
    parser = add_ddp_args(parser)
    args = parser.parse_args()
    ddp_setup(main, args)


if __name__ == '__main__':
    distributed_run()
