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
    learn.load(cfg.arch + str(2))

    # Make predictions
    learn.TTA() # Test time augmentation
    preds = learn.predict(is_test=True)
    ids = np.array([str(a)[-14:-4] for a in md.test_ds.fnames])

    np.save(PATH/'preds.npy', preds)
    np.save(PATH/'test_ids.npy', ids)


if __name__ == '__main__':
    main()
