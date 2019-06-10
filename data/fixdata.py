import os
from shutil import copyfile
import numpy as np
IMG_DIR = 'imgs100'
DIR_FIXED = 'imgs_fixed_100'
DIR_FIXED_TRAIN = 'imgs_fixed_train_100'
DIR_FIXED_TEST= 'imgs_fixed_test_100'
# os.mkdir(DIR_FIXED)
os.mkdir(DIR_FIXED_TRAIN)
os.mkdir(DIR_FIXED_TEST)

#
# seen = []
# for root, dirs, files in os.walk(IMG_DIR):
#     files = sorted(files)
#     print(files)
#     for i,file in enumerate(files):
#         if file.endswith(".jpg"):
#             orig = file
#             file = file.replace('_','-')
#             file = file.replace('oe','-')
#             file = file.replace('ae','-')
#             file = file[:-4]
#             remove = 0
#             for i in range(len(file) - 1,0,-1):
#                 if file[i].isdigit():
#                     remove += 1
#                 else:
#                     break
#             file = file[:-remove]
#             folder = np.random.choice(2, 1, p=[0.75, 0.25])[0]
#             if file not in seen:
#                 # os.mkdir(DIR_FIXED_TEST + '/'  + file)
#                 # os.mkdir(DIR_FIXED_TRAIN + '/'  + file)
#                 os.mkdir(DIR_FIXED + '/'  + file)
#                 seen.append(file)
#             copyfile(IMG_DIR + '/' + orig, DIR_FIXED+ '/' + file + '/' + orig)
seen = []
for root, dirs, files in os.walk(DIR_FIXED):
    files = sorted(files)
    if len(files) >= 50:
        perm = files[0][:-4]
        remove = 0
        for i in range(len(perm) -  1,0,-1):
            if perm[i].isdigit():
                remove += 1
            else:
                break
        print(perm)
        print(remove)
        perm = perm[:-remove]
        print(perm)
        if perm not in seen:
            os.mkdir(DIR_FIXED_TEST + '/'  + perm)
            os.mkdir(DIR_FIXED_TRAIN + '/'  + perm)
            seen.append(perm)
        for j,file in enumerate(files):
            if file.endswith(".jpg"):
                orig = file
                file = file[:-4]
                folder = np.random.choice(2, 1, p=[0.75, 0.25])[0]
                if folder == 0:
                    copyfile(IMG_DIR + '/' + orig, DIR_FIXED_TRAIN + '/' + perm + '/' + str(j) + '.jpg')
                else:
                    copyfile(IMG_DIR + '/' + orig, DIR_FIXED_TEST + '/' + perm + '/' + str(j) + '.jpg')

            # if file not in seen:
            #     os.mkdir(DIR_FIXED + '/'  + file)
            #     seen.append(file)
