import sys
import hpconfig as cfg
import pathconfig
sys.path.append(pathconfig.sys_path)

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
    cfg.print_hps()
    # Currently support "resnet34" and "densenet121"
    learn = get_learner(cfg.arch)

    learn.freeze_to(1)

    lr=cfg.seq_lrs[0]
    wd=cfg.seq_wds[0]

    lrs = np.array([lr/(cfg.lrs_scalings[0] ** 2),lr/cfg.lrs_scalings[0],lr])

    learn.fit(lrs,1,wds=cfg.seq_wds[0],cycle_len=cfg.cycle_lens[0],use_clr=cfg.clrs[0], callbacks=[EarlyStopping(cfg.arch + str(0), learn)])

    learn.load('rn0')

    learn.unfreeze()
    learn.bn_freeze(True)

    learn.fit(lrs/4, 1, wds=wd, cycle_len=cfg.cycle_lens[1], use_clr=cfg.clrs[1], callbacks=[EarlyStopping(cfg.arch + str(1), learn)])

    learn.load('rn1')

    lr=cfg.seq_lrs[1]
    wd=cfg.seq_wds[1]

    lrs = np.array([lr/(cfg.lrs_scalings[1] ** 2),lr/cfg.lrs_scalings[1],lr])
    learn.fit(lrs, 1, wds=wd, cycle_len=cfg.cycle_lens[2], callbacks=[EarlyStopping(cfg.arch + str(2), learn)])

if __name__ == '__main__':
    main()
