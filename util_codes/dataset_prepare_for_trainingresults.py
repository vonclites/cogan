from os import listdir
import os
from os.path import isfile, join

from shutil import copyfile

# if 'index' in f:
#     a = 2
# elif 'middle' in f:
#     a = 3
# elif 'ring' in f:
#     a = 4
# else:
#     exit('!!!!!!!!!')
#
# if 'left' in f:
#     a = a + 5


def get_finger_type(number):
    if number == 2:
        return 'index', 'right'
    elif number == 3:
        return 'middle', 'right'
    elif number == 4:
        return 'ring', 'right'
    elif number == 7:
        return 'index', 'left'
    elif number == 8:
        return 'middle', 'left'
    elif number == 9:
        return 'ring', 'left'
    else:
        raise ValueError('finger index is not valid', str(number))


source_dir = '/media/aldb2/0274E2F866ED37FC/BioCop fingerphoto/pix2pix_results_complete/photo2print_verifier3_pix2pix/test_latest/images/'

target_dir = '/media/aldb2/0274E2F866ED37FC/BioCop fingerphoto/pix2pix_results_complete/photo2print_verifier3_pix2pix/test_latest/prints_fake/'

allfiles = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]
print(allfiles)
c = 0
for f in allfiles:
    # get the fingerprint

    if 'fake_B' in f:

        id = f.split('_')
        id = id[0] + '_' + id[1]
        print_dir = target_dir + id
        if not os.path.exists(print_dir):
            os.makedirs(print_dir)

        copyfile(join(source_dir, f), join(print_dir, id+'.png'))


        # else:
        #     exit('!!!!!!!!!!!!!!!!')

        # copy fingerprint to the dir
        c+=1
        print(c, len(allfiles))
print('done!')
