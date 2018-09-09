import sys
sys.path.append("/home/bread/fastai/")

from fastai.conv_learner import *
from model import get_learner


def main():
    learn = get_learner()

    learn.freeze_to(1)

    lr=4e-2
    wd=1e-7

    lrs = np.array([lr/100,lr/10,lr])

    learn.fit(lrs,1,wds=wd,cycle_len=8,use_clr=(5,8))

    learn.unfreeze()
    learn.bn_freeze(True)

    learn.fit(lrs/4, 1, wds=wd, cycle_len=20, use_clr=(20,10))

    lr=2e-4
    wd=1e-7

    lrs = np.array([lr/100,lr/10,lr])
    learn.fit(lrs, 1, wds=wd, cycle_len=5)

if __name__ == '__main__':
    main()
