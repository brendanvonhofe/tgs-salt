import sys
sys.path.append("/home/bread/fastai/")

from fastai.conv_learner import *
from model import get_learner

class EarlyStopping(Callback):
    def __init__(self, savename, learn):
        self.best_dice = 0
        self.savename = savename
        self.learn = learn

    def on_epoch_end(self, metrics):
        if(metrics[2] > self.best_dice):
            self.best_dice = metrics[2]
            print(f'\nNew highest dice achieved: {metrics[2]}, saving to {self.savename}')
            self.learn.save(self.savename)


def main():
    # Currently support "resnet34" and "densenet121"
    learn = get_learner('resnet34')

    learn.freeze_to(1)

    lr=4e-2
    wd=1e-7

    lrs = np.array([lr/16,lr/4,lr])

    learn.fit(lrs,1,wds=wd,cycle_len=20,use_clr=(5,8), callbacks=[EarlyStopping('rn0', learn)])

    learn.load('rn0')

    learn.unfreeze()
    learn.bn_freeze(True)

    learn.fit(lrs/4, 1, wds=wd, cycle_len=50, use_clr=(20,10), callbacks=[EarlyStopping('rn1', learn)])

    learn.load('rn1')

    lr=2e-4
    wd=1e-7

    lrs = np.array([lr/16,lr/4,lr])
    learn.fit(lrs, 1, wds=wd, cycle_len=20, callbacks=[EarlyStopping('rn2', learn)])

if __name__ == '__main__':
    main()
