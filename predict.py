import sys
import hpconfig as cfg
import pathconfig
sys.path.append(pathconfig.sys_path)

from fastai.conv_learner import *
from model import get_learner
from data import get_model_data

PATH = Path(pathconfig.PATH)


def main():
    md = get_model_data()

    learn = get_learner(cfg.arch)
    preds = np.empty((0, 128, 128))
    for i in range(cfg.ensemble):

        learn.load(cfg.arch + str(2) + '-' + str(i))
        # Make predictions
        learn.TTA() # Test time augmentation
        preds = np.append(preds, [learn.predict(is_test=True)], axis = 0)




    preds = np.mean(preds, axis = 0)
    np.save(PATH/'preds.npy)', preds)

    ids = np.array([str(a)[8:-4] for a in md.test_ds.fnames])
    np.save(PATH/'test_ids.npy', ids)
if __name__ == '__main__':
    main()
