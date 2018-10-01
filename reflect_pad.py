import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from skimage import io, util

import warnings
warnings.filterwarnings("ignore")

PATH = Path("/home/bread/data/all")
IMAGES = PATH/'train'
MASKS = PATH/'masks'
TEST = PATH/'test'

def batch_reflect_pad(fns, path, are_masks=0):
    path128 = Path(str(path) + '128')
    ctr = 0
    print("Padding all images in {0}".format(path))
    for filename in fns:
        ctr += 1
        if(ctr % 1000 == 0):
            print(ctr)
        abs_path = path/filename
        im = io.imread(str(abs_path))
        if(are_masks):
            im_128 = util.pad(im, ((0,27), (0,27)), mode='reflect')
            # im_128 = im_128[:,:,0]
        else:
            im_128 = util.pad(im, ((0,27), (0,27), (0,0)), mode='reflect')
            # im_128 = im_128[:,:,:3]
        io.imsave(str(path128/filename), im_128)

def main():
    fns = os.listdir(IMAGES)
    test_fns = os.listdir(TEST)

    batch_reflect_pad(fns, IMAGES)
    batch_reflect_pad(fns, MASKS, 1)
    batch_reflect_pad(test_fns, TEST)

if __name__ == '__main__':
    main()