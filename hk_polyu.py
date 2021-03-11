import os
import math
import shutil
import argparse
import numpy as np
import random as python_random
from skimage.io import imread
from sklearn.model_selection import KFold

NUM_SUBJECTS = 209


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session1_dir', type=str,
                        help='path to data')
    parser.add_argument('--output_dir', type=str,
                        help='path to data')
    return parser.parse_args()


def create_image_directories(subjects, input_dir, output_dir):
    for subject in subjects:
        subject_dir = os.path.join(input_dir, subject)
        for eye in ['L', 'R']:
            eye_dir = os.path.join(subject_dir, eye)
            for domain in ['NIR', 'VIS']:
                domain_dir = os.path.join(eye_dir, domain)
                target_dir = os.path.join(output_dir, domain)
                target_dir = os.path.join(target_dir, subject) + eye
                os.makedirs(target_dir, exist_ok=True)
                filenames = os.listdir(domain_dir)
                for filename in filenames:
                    source_filepath = os.path.join(domain_dir, filename)
                    target_filepath = os.path.join(target_dir, filename)
                    shutil.copy(source_filepath, target_filepath)


def write_dev_class_list(dev_splits, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for split, (train_classes, validation_classes) in enumerate(dev_splits):
        filepath = os.path.join(output_dir, 'train{}.txt'.format(split))
        with open(filepath, 'w') as f:
            f.write('\n'.join(train_classes))
        filepath = os.path.join(output_dir, 'val{}.txt'.format(split))
        with open(filepath, 'w') as f:
            f.write('\n'.join(validation_classes))
    filepath = os.path.join(output_dir, 'dev.txt')
    with open(filepath, 'w') as f:
        f.write('\n'.join(np.concatenate(dev_splits[0])))


def calculate_image_stats(domain_dir):
    images = []
    for subject in os.listdir(domain_dir):
        subject_dir = os.path.join(domain_dir, subject)
        for filename in os.listdir(subject_dir):
            image = imread(os.path.join(subject_dir, filename))
            images.append(image)
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    return mean, std


def write_image_stats(nir_mean, nir_std, vis_mean, vis_std, stats_dir):
    os.makedirs(stats_dir, exist_ok=True)
    with open('/home/hulk2/data/hk/stats/nir_mean.txt', 'w') as f:
        np.savetxt(f, [nir_mean])
    with open('/home/hulk2/data/hk/stats/nir_std.txt', 'w') as f:
        np.savetxt(f, [nir_std])
    with open('/home/hulk2/data/hk/stats/vis_mean.txt', 'w') as f:
        np.savetxt(f, [vis_mean])
    with open('/home/hulk2/data/hk/stats/vis_std.txt', 'w') as f:
        np.savetxt(f, [vis_std])


def create_open_world_protocol(session1_dir, output_dir,):
    python_random.seed(62484)
    np.random.seed(62484)

    session1_subjects = sorted(os.listdir(session1_dir))

    assert len(session1_subjects) == NUM_SUBJECTS

    output_protocol_dir = os.path.join(output_dir, 'protocols')
    output_image_dir = os.path.join(output_dir, 'images')
    output_stats_dir = os.path.join(output_dir, 'stats')

    dev_subjects = session1_subjects[:math.ceil(NUM_SUBJECTS / 2)]
    test_subjects = session1_subjects[-math.ceil(NUM_SUBJECTS / 2):]

    dev_subjects = np.array(dev_subjects)
    kf = KFold(n_splits=5, shuffle=True)
    dev_splits = [(dev_subjects[train_indices], dev_subjects[val_indices])
                  for train_indices, val_indices in kf.split(dev_subjects)]

    dev_splits = [(subjects_to_classes(train_split, is_dev=True),
                   subjects_to_classes(val_split, is_dev=True))
                  for train_split, val_split in dev_splits]
    test_classes = subjects_to_classes(test_subjects, is_dev=False)

    output_dev_dir = os.path.join(output_image_dir, 'dev')
    output_test_dir = os.path.join(output_image_dir, 'test')
    create_image_directories(
        subjects=dev_subjects,
        input_dir=session1_dir,
        output_dir=output_dev_dir
    )
    create_image_directories(
        subjects=test_subjects,
        input_dir=session1_dir,
        output_dir=output_test_dir
    )
    write_dev_class_list(dev_splits, output_protocol_dir)
    filepath = os.path.join(output_protocol_dir, 'test.txt')
    with open(filepath, 'w') as f:
        f.write('\n'.join(test_classes))

    nir_mean, nir_std = calculate_image_stats(os.path.join(output_dev_dir, 'NIR'))
    vis_mean, vis_std = calculate_image_stats(os.path.join(output_dev_dir, 'VIS'))
    write_image_stats(
        nir_mean=nir_mean,
        nir_std=nir_std,
        vis_mean=vis_mean,
        vis_std=vis_std,
        stats_dir=output_stats_dir
    )

def subjects_to_classes(subjects, is_dev):
    classes = []
    for subject in subjects:
        if subject == '105':
            classes.append(subject + 'L') if is_dev else classes.append(subject + 'R')
        else:
            classes.append(subject + 'L')
            classes.append(subject + 'R')
    return classes


def class_filter(valid_classes_list):
    def is_valid_class(filepath):
        class_id = os.path.split(os.path.split(filepath)[0])[1]
        return class_id in valid_classes_list
    return is_valid_class


def run():
    args = parse_args()
    create_open_world_protocol(args.session1_dir, args.output_dir)


if __name__ == '__main__':
    run()
