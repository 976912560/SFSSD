import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco, voc512
import os




class tinySSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base tiny network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: tiny16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(tinySSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        #self.cfg = (coco, voc)[num_classes == 21]
        self.cfg = voc512
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.tiny = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        #self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.5)#select overlap=0.45(may 0.4 0.5)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        trans = list()
        loc = list()
        conf = list()

        # apply tiny up to fire4 relu
        for k in [0,1,2,3,4,5,6,7,15,23,24,32]:
            if k in [7,15,24,32]:
                x = self.tiny[k](x)
                x1 = self.tiny[k+1](x)
                x1 = self.tiny[k+2](x1)
                x2 = self.tiny[k+4](x)
                x2 = self.tiny[k+5](x2)
                x3 = self.tiny[k+6](x)
                x3 = self.tiny[k+7](x3)
                x = torch.cat((x1,x2),1 )+x3
            else:
                x = self.tiny[k](x)
        s1 = x.clone()

        

        # apply tiny up to fire8
        for k in [40,41,49,57,65]:
            if k in [41,49,57,65]:
                x = self.tiny[k](x)
                x1 = self.tiny[k+1](x)
                x1 = self.tiny[k+2](x1)
                x2 = self.tiny[k+4](x)
                x2 = self.tiny[k+5](x2)
                x3 = self.tiny[k+6](x)
                x3 = self.tiny[k+7](x3)
                x = torch.cat((x1,x2),1 )+x3
            else:
                x = self.tiny[k](x)
        s12 = x.clone()
        s2=self.tiny[141](s12)
        s2=self.tiny[142](s2)
        s2=self.tiny[143](s2)
        s2=self.tiny[144](s2)

        
        # apply tiny up to fire9
        for k in [73,74]:
            if k in [74]:
                x = self.tiny[k](x)
                x1 = self.tiny[k+1](x)
                x1 = self.tiny[k+2](x1)
                x2 = self.tiny[k+4](x)
                x2 = self.tiny[k+5](x2)
                x3 = self.tiny[k+6](x)
                x3 = self.tiny[k+7](x3)
                x = torch.cat((x1,x2),1 )+x3
            else:
                x = self.tiny[k](x)
        s3 = x.clone()


        
        # apply tiny up to fire10
        for k in [82,83]:
            if k in [83]:
                x = self.tiny[k](x)
                x1 = self.tiny[k+1](x)
                x1 = self.tiny[k+2](x1)
                x2 = self.tiny[k+4](x)
                x2 = self.tiny[k+5](x2)
                x3 = self.tiny[k+6](x)
                x3 = self.tiny[k+7](x3)
                x = torch.cat((x1,x2),1 )+x3
            else:
                x = self.tiny[k](x)
        s4 = x.clone()


        for k in [91]:
            if k in [91]:
                a = self.tiny[k](s3)
                a1 = self.tiny[k+1](a)
                a1 = self.tiny[k+2](a1)
                a2 = self.tiny[k+4](a)
                a2 = self.tiny[k+5](a2)
                a3 = self.tiny[k+6](a)
                a3 = self.tiny[k+7](a3)
                a = torch.cat((a1,a2),1 )+a3
            else:
                a = self.tiny[k](a)
                
        for k in [99,101]:
            if k in [101]:
                a = self.tiny[k](a)
                a1 = self.tiny[k+1](a)
                a1 = self.tiny[k+2](a1)
                a2 = self.tiny[k+4](a)
                a2 = self.tiny[k+5](a2)
                a3 = self.tiny[k+6](a)
                a3 = self.tiny[k+7](a3)
                a = torch.cat((a1,a2),1 )+a3
            else:
                a = self.tiny[k](a)
                a = self.tiny[k+1](a)
                s5 = a.clone()
                
        for k in [109,111]:
            if k in [111]:
                b = self.tiny[k](s4)
                b1 = self.tiny[k+1](b)
                b1 = self.tiny[k+2](b1)
                b2 = self.tiny[k+4](b)
                b2 = self.tiny[k+5](b2)
                b3 = self.tiny[k+6](b)
                b3 = self.tiny[k+7](b3)
                b = torch.cat((b1,b2),1 )+b3
            else:
                a = self.tiny[k](a)
                a = self.tiny[k+1](a)
                s6 = a.clone()
        
        for k in [119,121]:
            if k in [121]:
                b = self.tiny[k](b)
                b1 = self.tiny[k+1](b)
                b1 = self.tiny[k+2](b1)
                b2 = self.tiny[k+4](b)
                b2 = self.tiny[k+5](b2)
                b3 = self.tiny[k+6](b)
                b3 = self.tiny[k+7](b3)
                b = torch.cat((b1,b2),1 )+b3
            else:
                b = self.tiny[k](b)
                b = self.tiny[k+1](b)
                s7 = b.clone()
                
        for k in [129,131]:
            if k in [131]:
                b = self.tiny[k](b)
                b1 = self.tiny[k+1](b)
                b1 = self.tiny[k+2](b1)
                b2 = self.tiny[k+4](b)
                b2 = self.tiny[k+5](b2)
                b3 = self.tiny[k+6](b)
                b3 = self.tiny[k+7](b3)
                b = torch.cat((b1,b2),1 )+b3
            else:
                b = self.tiny[k](b)
                b = self.tiny[k+1](b)
                s8 = b.clone()
                
        b = self.tiny[139](b) 
        s9 = self.tiny[140](b) 
        s = torch.cat((s1,s2,s6,s9),1 )
        s = F.relu(s)
        sources.append(s)
        s10 = torch.cat((s12,s5,s8),1 )
        s10 = F.relu(s10)
        sources.append(s10)
        s11 = torch.cat((s3,s7),1 )
        s11 = F.relu(s11)
        sources.append(s11)
        s4 = F.relu(s4)
        sources.append(s4)


                
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                #self.priors.type(type(x.data)).cuda()                  # default boxes
                self.priors.type(type(x.data))
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                #use train_l2t_ww#
                #self.priors.cuda()
                self.priors
            )
        # eval
        return output 
        #select and train
        #return output,[sources[0],sources[1],sources[2]]
    def forward_with_features(self, x):       
        return self.forward(x)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision tiny make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/tiny.py
def tiny(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            conv2d = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
            #conv2d = nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2)#最新
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(64), nn.ReLU(inplace=True)]#64
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = 64#64      
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        elif v == 'E':    
            convt2d = nn.Conv2d(83, 24, kernel_size=3, padding=1)
            layers += [convt2d, nn.ReLU(inplace=True)]
        elif v == 'F':    
            convt2d = nn.ConvTranspose2d(24, 48, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)
            layers += [convt2d, nn.ReLU(inplace=True)]
        elif v == 'F1':
            convt2d = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)
            layers += [convt2d, nn.ReLU(inplace=True)]
        elif type(v)== list:
            if v[0]=='s1':
                in_channels = 85
                squeeze2d = nn.Conv2d(in_channels, v[1], kernel_size=1)
                conv2d_1= nn.Conv2d(v[1], v[2], kernel_size=1)
                conv2d_2= nn.Conv2d(v[1], v[3], kernel_size=3, padding=1)
                conv2d_3= nn.Conv2d(v[1], v[4], kernel_size=1)
            else:
                squeeze2d = nn.Conv2d(in_channels, v[0], kernel_size=1)
                conv2d_1= nn.Conv2d(v[0], v[1], kernel_size=1)
                conv2d_2= nn.Conv2d(v[0], v[2], kernel_size=3, padding=1)
                conv2d_3= nn.Conv2d(v[0], v[3], kernel_size=1)
            if batch_norm:
                layers += [squeeze2d,conv2d_1, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                layers += [squeeze2d,conv2d_2, nn.BatchNorm2d(v[2]), nn.ReLU(inplace=True),conv2d_3, nn.ReLU(inplace=True)]
            else:
                layers += [squeeze2d,conv2d_1, nn.ReLU(inplace=True)]
                layers += [squeeze2d,conv2d_2, nn.ReLU(inplace=True),conv2d_3, nn.ReLU(inplace=True)]
            if v[0]=='s1':
                in_channels = v[2]+v[3]
            else:
                in_channels = v[1]+v[2]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
    return layers

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to tiny for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels not in ['S','S1']:
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=3, stride=2, padding=1)]
            elif v == 'S1':
                #layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           #kernel_size=3, stride=2)]300
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=2, stride=2)]#512
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1)]
            flag = not flag
        in_channels = v
    return layers


def multibox(tiny, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []    
    tiny_source = [633,315,170,85]#[173,334,101,85]
    for k, v in enumerate(tiny_source):
        loc_layers += [nn.Conv2d(v,cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v,cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 4):   
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return tiny, extra_layers, (loc_layers, conf_layers)

#[25,29,54,83]
base = {
    '320': ['M', 64, 64, 'C', [15,49,53,102], [15,54,52,106],'C', [29,92,94,186], [29,90,83,173],'C', [49,163,171,334], [44,166,161,327], [45,155,146,301],[45,83,73,156],
			'C',[37,45,56,101], 'C', [38,41,44,85] ,['s1',38,41,44,85],'F1',[37,41,44,85],'F1',[37,45,56,101],'F1',[37,45,56,101],'F1',[38,41,44,85],'F1', 'E','F'],
    '512': ['M', 64, 64, 'C', [29,90,83,173], [29,92,94,186],'C', [49,163,171,334], [44,166,171,337] ,'C', [49,163,171,334], [44,166,161,327], [45,155,146,301],[25,29,54,83],
			'C',[37,39,46,85], 'C', [38,41,44,85] ,['s1',38,41,44,85],'F1',[37,45,56,101],'F1',['s1',37,41,44,85],'F1',[23,73,74,147],'F1',[37,73,74,147],'F1', 'E','F'],
}
extras = {
    '320': ['S', 51, 46, 'S1', 55, 85],
    '512': ['S', 51, 46, 'S1', 55, 85, 'S1', 51, 46],
}
mbox = {
    '320': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_tinyssd(phase, size=512, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 512:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(tiny(base[str(size)], 3),
                                     add_extras(extras[str(size)], 85),
                                     mbox[str(size)], num_classes)
    return tinySSD(phase, size, base_, extras_, head_, num_classes)
