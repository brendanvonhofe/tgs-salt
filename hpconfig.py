seq_lrs = [4e-2, 2e-4]
lrs_scalings = [4, 4]
#lrs_scalings = [10, 10]
seq_wds = [1e-7, 1e-7]
cycle_lens = [20, 50, 20]
clrs = [(5, 8), (20, 10)]
kernels = [256, 256, 256, 256]
#kernels = [256, 128, 64, 32]
arch = "resnet34"
# arch = "densenet121"
p = [0.5, 0.5, 0.5, 0.0]

def print_hps():
    print("learning rates at each set of epochs: ", seq_lrs)
    print("scaling of learning rates through layer groups: ", lrs_scalings)
    print("weight decays at each epoch set: ", seq_wds)
    print("cycle lengths: ", cycle_lens)
    print("learning rate cycle parameters: ", clrs)
    print("#kernels at each layer: ", kernels)
    print("architecture: ", arch)
    print("Probability of dropout in each layer: ", p)
