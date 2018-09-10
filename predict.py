import sys
sys.path.append("/home/bread/fastai/")

from fastai.conv_learner import *
from model import get_learner
from data import get_model_data

PATH = Path('/home/bread/data/salt')


def main():
    md = get_model_data()

    learn = get_learner('densenet121')
    learn.load('dn2')
    
    # Make predictions
    learn.TTA() # Test time augmentation
    preds = learn.predict(is_test=True)
    ids = np.array([str(a)[8:-4] for a in md.test_ds.fnames])

    np.save(PATH/'preds.npy', preds)
    np.save(PATH/'test_ids.npy', ids)


if __name__ == '__main__':
    main()