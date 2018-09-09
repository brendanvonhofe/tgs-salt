# tgs-salt Kaggle competition

Install fastai from https://github.com/fastai/fastai

Initial up-scaling of images for UNet and downscaling for submission currently requires skimage which breaks the fastai
environment. Not the best solution but I have a separate environment to use skimage.

todo ~
	- Set up SSH access to local machine
	- Rewrite ipynb notebook as a more organized Python package.
	- Reflection padding and other ways to deal with image size
	- cross fold validation
	- early stopping
	- Combat overfitting
		- Dropout
		- Additional data augmentation
			- Shearing, zooming, distortions, etc.
	- Define hyperparameters and write scripts to tune em
		- Dilated conv
		- Train model from scratch?
	- Implement additional models (see below)
	- Ensemble
	- *** Architecture search


Resources:
	- TernausNet: https://arxiv.org/abs/1801.05746
	- Linknet: https://arxiv.org/pdf/1707.03718.pdf
	- Pyramid Scene Parsing Network: https://arxiv.org/pdf/1612.01105.pdf
	- Fully Convolutional Networks for Semantic Segmentation: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
	- Segnet: https://arxiv.org/pdf/1511.00561.pdf
	- DeepUNet: https://arxiv.org/pdf/1709.00201.pdf
	- Dilated Convs for small objects [?]: https://arxiv.org/pdf/1709.00179.pdf
	- CRFs: https://arxiv.org/pdf/1711.04483.pdf
		- As post-processing?
	- DeepLab: https://arxiv.org/pdf/1606.00915.pdf

https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
