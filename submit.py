import sys
import pathconfig
sys.path.append(pathconfig.sys_path)

from fastai.conv_learner import *
from model import get_learner
from data import get_model_data

PATH = Path(pathconfig.PATH)

from skimage.transform import resize

img_size_ori = 101
img_size_target = 128

# # Script to upsample train/test images

# dirs = [('test128', TEST_DN), ('train128', TRAIN_DN)]
# for (d128, d) in dirs:
#     for fn in os.listdir(PATH/d):
#         im = imageio.imread(PATH/d/fn)
#         up_im = upsample(im)
#         imageio.imwrite(PATH/d128/fn, up_im)

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True, anti_aliasing=True)

def myfunc(x):
    if(x > 0):
        return 1
    else:
        return 0

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def main():

    preds = np.load(PATH/'preds.npy')
    ids = np.load(PATH/'test_ids.npy')

    preds101 = np.zeros((18000,101,101))
    for i in range(len(preds)):
        preds101[i] = downsample(preds[i])

    vfunc = np.vectorize(myfunc)
    predictions = vfunc(preds101)

    rle_preds = []
    for m in predictions:
        rle_preds.append(rle_encoding(m))

    rle_str = [' '.join(str(e) for e in rl) for rl in rle_preds]

    sub = pd.DataFrame(np.zeros((18000, 2)), columns=['id', 'rle_mask'])

    sub['id'] = ids
    sub['rle_mask'] = rle_str

    sub.to_csv(PATH/'bulshit.csv', index=False)

    print("submission finished")

if __name__ == '__main__':
    main()
