import argparse

from cogan import train_cogan


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=400, type=int, help='batch size')
    parser.add_argument('--margin', default=100, type=int, help='batch size')
    parser.add_argument('--adversarial_coeff', default=1, type=float,
                        help='Adversarial Loss Coefficient')
    parser.add_argument('--pixel_coeff', default=1, type=float,
                        help='Pixel-to-Pixel Loss Coefficient')
    parser.add_argument('--perceptual_coeff', default=1, type=float,
                        help='Perceptual Loss Coefficient')
    parser.add_argument('--nir_dir', type=str,
                        default='/home/hulk1/data/periocular/hk/images/dev/NIR',
                        help='path to data')
    parser.add_argument('--vis_dir', type=str,
                        default='/home/hulk1/data/periocular/hk/images/dev/VIS',
                        help='path to data')
    parser.add_argument('--valid_train_classes_fps', type=str, nargs='+',
                        help='text file of class labels to include in training dataset')
    parser.add_argument('--valid_test_classes_fps', type=str, nargs='+',
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
    parser.add_argument('--backbone', default='resnet18', type=str,
                        help='resnet18, resnet34, resnet50,'
                             'and their wider variants, resnet50x4')
    parser.add_argument('-d', '--feat_dim', default=128, type=int,
                        help='feature dimension for contrastive loss')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert len(args.valid_train_classes_fps) == len(args.valid_test_classes_fps)
    for train, test in zip(args.valid_train_classes_fps, args.valid_test_classes_fps):
        args.valid_train_classes_fp = train
        args.valid_test_classes_fp = test
        train_cogan.run(args)
