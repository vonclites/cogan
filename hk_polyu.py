import os
import math
import shutil
import numpy as np
import random as python_random
from skimage.io import imread
from sklearn.model_selection import KFold

SESSION1_DIR = '/home/hulk1/data/periocular/hk/PolyU_Cross_Session_1/PolyU_Cross_Iris'
SESSION2_DIR = '/home/hulk1/data/periocular/hk/PolyU_Cross_Session_2/PolyU_Cross_Iris'
IMAGE_DIR = '/home/hulk1/data/periocular/hk/images'
PROTOCOL_DIR = '/home/hulk1/data/periocular/hk/protocols'
NUM_SUBJECTS = 209


def create_image_directories(subjects, output_dir):
    for subject in subjects:
        subject_dir = os.path.join(SESSION1_DIR, subject)
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


def create_open_world_protocol():
    python_random.seed(62484)
    np.random.seed(62484)

    session1_subjects = sorted(os.listdir(SESSION1_DIR))

    assert len(session1_subjects) == NUM_SUBJECTS

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

    create_image_directories(
        subjects=dev_subjects,
        output_dir=os.path.join(IMAGE_DIR, 'dev')
    )
    create_image_directories(
        subjects=test_subjects,
        output_dir=os.path.join(IMAGE_DIR, 'test')
    )
    write_dev_class_list(dev_splits, PROTOCOL_DIR)
    filepath = os.path.join(PROTOCOL_DIR, 'test.txt')
    with open(filepath, 'w') as f:
        f.write('\n'.join(test_classes))


def run():
    create_open_world_protocol()


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


if __name__ == '__main__':
    run()
