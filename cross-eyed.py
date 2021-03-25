import os
import math
import shutil
import pathlib
import numpy as np
import random as python_random
from skimage.io import imread, imsave
from sklearn.model_selection import KFold

INPUT_IMAGE_DIR = pathlib.Path('/home/hulk1/data/periocular/ce/PeriCrossEyed')
OUTPUT_IMAGE_DIR = pathlib.Path('/home/hulk1/data/periocular/ce/images')

NIR_FOLDER = 'NIR'
VIS_FOLDER = 'VW'

PROTOCOL_DIR = '/home/hulk1/data/periocular/ce/protocols'


def create_open_world_protocol():
    python_random.seed(62484)
    np.random.seed(62484)

    subjects = sorted(os.listdir(INPUT_IMAGE_DIR / 'L'))
    dev_subjects = subjects[:int(len(subjects) / 2)]
    test_subjects = subjects[int(len(subjects) / 2):]

    dev_subjects = np.array(dev_subjects)
    kf = KFold(n_splits=5, shuffle=True)
    dev_splits = [(dev_subjects[train_indices], dev_subjects[val_indices])
                  for train_indices, val_indices in kf.split(dev_subjects)]
    dev_splits = [([subject + eye for subject in train_subjects for eye in ['L', 'R']],
                   [subject + eye for subject in val_subjects for eye in ['L', 'R']])
                  for train_subjects, val_subjects in dev_splits]
    test_classes = [subject + eye for subject in test_subjects for eye in ['L', 'R']]

    protocol_dir = os.path.join(PROTOCOL_DIR, 'ow')
    os.makedirs(protocol_dir, exist_ok=True)
    for i, (train_subjects, val_subjects) in enumerate(dev_splits):
        with open(os.path.join(protocol_dir, 'train{}.txt'.format(i)), 'w') as f:
            f.write('\n'.join(train_subjects))
        with open(os.path.join(protocol_dir, 'val{}.txt'.format(i)), 'w') as f:
            f.write('\n'.join(val_subjects))
    with open(os.path.join(protocol_dir, 'dev.txt'), 'w') as f:
        f.write('\n'.join(np.concatenate(dev_splits[0])))
    with open(os.path.join(protocol_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_classes))

    os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, 'dev'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, 'test'), exist_ok=True)

    create_image_directories(dev_subjects, OUTPUT_IMAGE_DIR / 'dev')
    create_image_directories(test_subjects, OUTPUT_IMAGE_DIR / 'test')

    write_image_stats(
        image_dir=OUTPUT_IMAGE_DIR / 'dev',
        stats_dir=os.path.join(os.path.dirname(OUTPUT_IMAGE_DIR), 'stats')
    )


def create_image_directories(subjects, output_dir):
    output_dir = pathlib.Path(output_dir)
    for subject in subjects:
        for eye in ['L', 'R']:
            for domain in [NIR_FOLDER, VIS_FOLDER]:
                output_class_dir = output_dir / domain / (subject + eye)
                input_dir = INPUT_IMAGE_DIR / eye / subject / domain
                filenames = sorted(os.listdir(input_dir))
                os.makedirs(output_class_dir, exist_ok=True)
                for filename in filenames:
                    basename, ext = os.path.splitext(filename)
                    output_filename = basename + '_{}'.format(eye) + ext
                    image = imread(input_dir / filename)
                    if image.shape[:2] != (800, 900):
                        missing = 900 - image.shape[1]
                        left = int(missing / 2)
                        right = math.ceil(missing / 2)
                        padding = [(0, 0), (left, right), (0, 0)]
                        image = np.pad(image, padding[:image.ndim], mode='edge')
                        assert image.shape[:2] == (800, 900)
                        imsave(output_class_dir / output_filename, image)
                    else:
                        shutil.copy(input_dir / filename,
                                    output_class_dir / output_filename)


# noinspection PyTypeChecker
def write_image_stats(image_dir, stats_dir):
    os.makedirs(stats_dir, exist_ok=True)
    for domain in [NIR_FOLDER, VIS_FOLDER]:
        mean, std = calculate_image_stats(image_dir / domain)
        np.savetxt(os.path.join(stats_dir, '{}_mean.txt'.format(domain)), [mean])
        np.savetxt(os.path.join(stats_dir, '{}_std.txt'.format(domain)), [std])


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


if __name__ == '__main__':
    create_open_world_protocol()
