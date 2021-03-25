import os
import json
import torch
import argparse
import torchvision
import numpy as np
import random as python_random
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tqdm import tqdm

import backboned_unet
from cogan import utils
from cogan.dataset import get_test_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Contrastive view')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--margin', type=int, help='batch size')
    parser.add_argument('--backbone', type=str,
                        help='resnet18, resnet34, resnet50'
                        )
    parser.add_argument('--nir_dir', type=str,
                        help='path to data')
    parser.add_argument('--vis_dir', type=str,
                        help='path to data')
    parser.add_argument('--valid_classes_fp', type=str,
                        help='text file of class labels to include in dataset')
    parser.add_argument('--ckpt_dir', type=str,
                        help='path to save the data')
    parser.add_argument('--sync_capture', dest='sync_capture', action='store_true')
    parser.add_argument('--no_sync_capture', dest='sync_capture', action='store_false')
    parser.set_defaults(sync_capture=True)
    parser.add_argument('--nir_mean_fp', type=str,
                        help='Path to file containing channelwise image statistic')
    parser.add_argument('--nir_std_fp', type=str,
                        help='Path to file containing channelwise image statistic')
    parser.add_argument('--vis_mean_fp', type=str,
                        help='Path to file containing channelwise image statistic')
    parser.add_argument('--vis_std_fp', type=str,
                        help='Path to file containing channelwise image statistic')
    parser.add_argument('--results_folder', type=str,
                        help='path to save the data')
    parser.add_argument('--feat_dim', type=int,
                        help='feature dimension for contrastive loss')
    return parser.parse_args()


def run(args):
    python_random.seed(62484)
    np.random.seed(62484)
    torch.manual_seed(62484)

    gpu0 = torch.device('cuda:0')
    gpu1 = torch.device('cuda:1')
    cpu = torch.device('cpu')

    state = torch.load(os.path.join(args.ckpt_dir, 'checkpoint.pt'))

    net_vis = backboned_unet.Unet(backbone_name=args.backbone, classes=args.feat_dim)
    net_nir = backboned_unet.Unet(backbone_name=args.backbone, classes=args.feat_dim)

    net_vis.to(gpu0)
    net_vis.load_state_dict(state['net_photo'])
    net_vis.eval()

    net_nir.to(gpu1)
    net_nir.load_state_dict(state['net_print'])
    net_nir.eval()

    output_dir = os.path.join(args.ckpt_dir, args.results_folder)
    os.makedirs(output_dir, exist_ok=True)

    test_loader = get_test_dataset(
        batch_size=args.batch_size,
        vis_dir=args.vis_dir,
        nir_dir=args.nir_dir,
        valid_classes_fp=args.valid_classes_fp,
        vis_mean_fp=args.vis_mean_fp,
        vis_std_fp=args.vis_std_fp,
        nir_mean_fp=args.nir_mean_fp,
        nir_std_fp=args.nir_std_fp,
        sync_capture=args.sync_capture
    )

    max_steps = len(test_loader)

    dist_l = []
    lbl_l = []
    for step, (img_vis, img_nir, lbl) in enumerate(test_loader):
        # plot_tensor([img_photo[0], img_print[0]])
        batch_size = img_vis.size(0)
        print('Step: {} / {}    Batch Size: {}'.format(step, max_steps, batch_size))
        lbl = lbl.type(torch.float)

        img_vis = img_vis.to(gpu0)
        img_nir = img_nir.to(gpu1)
        # lbl = lbl.to(gpu0)

        fake_vis, y_vis = net_vis(img_vis)
        fake_nir, y_nir = net_nir(img_nir)

        y_vis = y_vis.to(cpu)
        y_nir = y_nir.to(cpu)
        dist = ((y_vis - y_nir) ** 2).sum(1)
        dist_l.append(dist.data)
        lbl_l.append((1 - lbl).data)

    dist = torch.cat(dist_l, 0)
    lbl = torch.cat(lbl_l, 0)
    dist = dist.cpu().detach().numpy()
    lbl = lbl.cpu().detach().numpy()

    fpr, tpr, threshold = metrics.roc_curve(lbl, dist)
    roc_auc = metrics.auc(fpr, tpr)
    eer = utils.eer(fpr, tpr)
    frr = 1 - tpr
    frr_10 = frr[np.searchsorted(fpr, .1)]
    frr_01 = frr[np.searchsorted(fpr, .01)]

    results = {
        'auc': roc_auc,
        'eer': eer,
        'frr10': frr_10,
        'frr01': frr_01
    }
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        json.dump(results, f, indent=4)

    np.save(os.path.join(output_dir, 'pr_ph_lbl_test.npy'), lbl)
    np.save(os.path.join(output_dir, 'pr_ph_dist_test.npy'), dist)

    torchvision.utils.save_image(img_vis[:6], os.path.join(output_dir, 'real_vis.png'))
    torchvision.utils.save_image(img_nir[:6], os.path.join(output_dir, 'real_nir.png'))
    torchvision.utils.save_image(fake_vis[:6], os.path.join(output_dir, 'fake_vis.png'))
    torchvision.utils.save_image(fake_nir[:6], os.path.join(output_dir, 'fake_nir.png'))

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


if __name__ == '__main__':
    run(parse_args())
