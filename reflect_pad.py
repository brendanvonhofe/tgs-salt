import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

PATH = Path("/home/bread/data/salt")
IMAGES = PATH/'images'
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
        im = plt.imread(str(abs_path))
        if(are_masks):
            im_128 = np.pad(im, ((0,27), (0,27)), mode='reflect')
        else:
            im_128 = np.pad(im, ((0,27), (0,27), (0,0)), mode='reflect')
        plt.imsave(str(path128/filename), im_128)

def main():
    fns = os.listdir(IMAGES)
    test_fns = os.listdir(TEST)

    batch_reflect_pad(fns, IMAGES)
    batch_reflect_pad(fns, MASKS, 1)
    batch_reflect_pad(test_fns, TEST)

if __name__ == '__main__':
    main()