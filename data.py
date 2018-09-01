import sys
import pandas as pd

sys.path.append("/Users/brendan/Desktop/fastai/")

from fastai.dataset import *

PATH = Path('data')
TRAIN_DN = 'images128'
MASKS_DN = 'masks128'
TEST_DN = 'test128'

class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): return 0

def train_test_fnames(val_size=400):
    train_csv = pd.read_csv(PATH/'train.csv')

    x_names = np.array([Path(TRAIN_DN)/(o+'.png') for o in train_csv['id']])
    y_names = np.array([Path(MASKS_DN)/(o+'.png') for o in train_csv['id']])
    test_names = np.array([Path(TEST_DN)/o for o in os.listdir(PATH/TEST_DN)])

    val_idxs = list(range(val_size))
    ((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, x_names, y_names)
    test=(test_names, test_names)

    return {'trn': (trn_x, trn_y), 'val': (val_x, val_y), 'test': test}

def transforms():
    return [RandomRotate(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
#             RandomDihedral(tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)]

def get_model_data(batch_size=32):
    im_size = 128

    sets = train_test_fnames()
    trn = sets['trn']
    val = sets['val']
    test = sets['test']

    aug_tfms = transforms()

    tfms = tfms_from_model(resnet34, im_size, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
    datasets = ImageData.get_ds(MatchedFilesDataset, trn, val, tfms, test=test, path=PATH)
    md = ImageData(PATH, datasets, batch_size, num_workers=16, classes=None)

    return md