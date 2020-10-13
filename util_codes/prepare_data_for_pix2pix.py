from os import listdir
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shutil import copyfile
import numpy as np
import cv2

# train data
# photo_dir = '/media/aldb2/0274E2F866ED37FC/BioCop fingerphoto/cleaned2012_trainingset/photo/'
# print_dir = '/media/aldb2/0274E2F866ED37FC/BioCop fingerphoto/cleaned2012_trainingset/print/'

# test data
photo_dir = '/media/aldb2/0274E2F866ED37FC/BioCop fingerphoto/clean_13_as_test/photo/'
print_dir = '/media/aldb2/0274E2F866ED37FC/BioCop fingerphoto/clean_13_as_test/print/'

out_dir = '/media/aldb2/0274E2F866ED37FC/BioCop fingerphoto/cleaned2012_trainingset/photo2print_dataset/test/'

allids = [f for f in listdir(photo_dir)]


for i, id in enumerate(allids):
    ph_dir = join(photo_dir, id)
    pr_dir = join(print_dir, id)
    fphotos_add = [f for f in listdir(ph_dir) if isfile(join(ph_dir, f))]
    fprints_add = [f for f in listdir(pr_dir) if isfile(join(pr_dir, f))]

    c = 0
    for fph in fphotos_add:
        for fpr in fprints_add:
            img1 = cv2.imread(join(ph_dir, fph))
            img2 = cv2.imread(join(pr_dir, fpr))
            img1 = cv2.resize(img1, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(img2, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)



            outimg = np.concatenate((img1, img2), 1)
            cv2.imwrite(out_dir + id + '_' + str(c)+'.png', outimg)
            c+=1

    if i%100==0:
        print(i)
            # if len(fprints_add) > 2:
    #     img1 = mpimg.imread(join(pr_dir, fprints_add[0]))
    #     img2 = mpimg.imread(join(pr_dir, fprints_add[1]))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(img1)
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(img2)
    #     plt.show()

    # print(fprints_add)
    # print(fphotos_add)
    # print('------------------')

exit()

exit()

for a in allfiles:
    print(a)
exit()

print(allfiles)
c = 0
for f in allfiles:
    # get the fingerprint
    if '300LC' in f:
        # get id of the person
        fsplit = f.split('_')
        id = fsplit[2]
        finger = fsplit[4]

        photo_dir = target_dir + 'photo/' + id + '_' + finger
        print_dir = target_dir + 'print/' + id + '_' + finger
        if not os.path.exists(photo_dir):
            os.makedirs(photo_dir)
        # else:
        #     exit('!!!!!!!!!!!!!!!!')

        if not os.path.exists(print_dir):
            os.makedirs(print_dir)
        # else:
        #     exit('!!!!!!!!!!!!!!!!')

        # copy fingerprint to the dir
        copyfile(join(source_dir, f), join(print_dir, f))

        # find the corresponding fphoto
        for f1 in allfiles:

            finger_name, hand_name = get_finger_type(int(finger))

            if id in f1 and finger_name in f1 and hand_name in f1:
                copyfile(join(source_dir, f1), join(photo_dir, f))
    c += 1
    print(c, len(allfiles))
print('done!')
