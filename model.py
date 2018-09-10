import sys
sys.path.append("/Users/brendan/Desktop/fastai/")

from fastai.conv_learner import *
from data import get_model_data

# Instantiate model
def get_learner(arch='resnet34'):
    # Instantiate fastai learner
    md = get_model_data()

    if(arch == 'resnet34'):
        f = resnet34
        cut,lr_cut = model_meta[f]
        layers = cut_model(f(True), cut)
        m_base = nn.Sequential(*layers)
        m = to_gpu(Unet34(m_base, arch='resnet34'))
    elif(arch == 'densenet121'):
        f = dn121
        cut,lr_cut = model_meta[f]
        layers = cut_model(f(True), cut)
        m_base = nn.Sequential(*layers)[0]
        m = to_gpu(Unet34(m_base, arch='densenet121'))

    models = UnetModel(m, lr_cut=lr_cut)
    learn = ConvLearner(md, models)
    learn.opt_fn=optim.Adam
    learn.crit=nn.BCEWithLogitsLoss()
    learn.metrics=[accuracy_thresh(0.5),dice]

    return learn

# Similar to IoU metric
def dice(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

# Save activations from contracting path to concatenate to expansive path in Unet
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

# Neural net module for expansive path in Unet
class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p) # Expansive part
        x_p = self.x_conv(x_p) # Further convolution on the activations from contracting part
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))


# Expansive path of Unet
class Unet34(nn.Module):
    def __init__(self, rn, arch='resnet34', p = [0.8, 0.8, 0.8, 0.8]):
        super().__init__()
        # Number of channels (filters) at key layers
        rn34_ch = [(512, 256), (256, 128), (256, 64), (256, 64)]
        dn121_ch = [(1024, 1024), (256, 512), (256, 256), (256, 64)]
        # Layer groups to save features of
        rn34_l = [2,4,5,6]
        dn121_l = [2,4,6,8]


        self.rn = rn # Resnet base

        if(arch == 'resnet34'):
            ch = rn34_ch
            ls = rn34_l
        elif(arch == 'densenet121'):
            ch = dn121_ch
            ls = dn121_l

        self.sfs = [SaveFeatures(rn[i]) for i in ls] # Saved activations from contracting/resnet part
#         self.sfs = [SaveFeatures(rn[i]) for i in [2,5,12,22]] # for VGG16

        self.up1 = UnetBlock(ch[0][0],ch[0][1],256)
        self.drop1 = nn.Dropout2d(p[0])
        self.up2 = UnetBlock(ch[1][0],ch[1][1],256)
        self.drop2 = nn.Dropout2d(p[1])
        self.up3 = UnetBlock(ch[2][0],ch[2][1],256)
        self.drop3 = nn.Dropout2d(p[2])
        self.up4 = UnetBlock(ch[3][0],ch[3][1],256)
        self.drop4 = nn.Dropout2d(p[3])
        # self.up1 = UnetBlock(512,256,256)
        # self.up2 = UnetBlock(256,128,256)
        # self.up3 = UnetBlock(256,64,256)
        # self.up4 = UnetBlock(256,64,256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)
        
    def forward(self,x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.drop1(x)
        x = self.up2(x, self.sfs[2].features)
        x = self.drop2(x)
        x = self.up3(x, self.sfs[1].features)
        x = self.drop3(x)
        x = self.up4(x, self.sfs[0].features)
        x = self.drop4(x)
        x = self.up5(x)
        return x[:,0]
    
    def close(self):
        for sf in self.sfs: sf.remove()

class UnetModel():
    def __init__(self, model, lr_cut, name='unet'):
        self.model,self.name,self.lr_cut = model,name, lr_cut

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [self.lr_cut]))
        return lgs + [children(self.model)[1:]]

