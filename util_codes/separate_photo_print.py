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


# source_dir = '../../cleaned2012_seconddrive/'
#
# target_dir = '../../clean_12_as_train/'

# Still needs plenty of work
source_dir = '/home/hulk2/dev/research/cogan_demo/test/cleaned_13'

target_dir = '/home/hulk2/dev/research/cogan_demo/test/separated'

os.makedirs("%s" % (target_dir), exist_ok=True)

allfiles = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]
print(allfiles)
c = 0
for f in allfiles:
    # get the fingerprint
    if '300LC' in f:
        # get id of the person
        fsplit = f.split('_')
        if fsplit[0] == "horizontal" or fsplit[0] == "vertical" or fsplit[0] == "rotated":
            id = fsplit[3]
            finger = fsplit[5]
        else:
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
    c+=1
    print(c, len(allfiles))
print('done!')
