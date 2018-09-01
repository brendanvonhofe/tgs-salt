import sys
sys.path.append("/Users/brendan/Desktop/fastai/")

from fastai.conv_learner import *
from model import get_learner


def main():
    learn = get_learner()

    learn.freeze_to(1)

    learn.lr_find()
    learn.sched.plot()

if __name__ == '__main__':
    main()