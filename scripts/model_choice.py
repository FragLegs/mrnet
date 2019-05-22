# -*- coding: utf-8 -*-
from model import *

MODELS = {
    'MRNet': MRNet,
    'MRNet-VGG': MRNetVGG,
    # 'MRNet-Dense': MRNetDense,
    'MRNet-Squeeze': MRNetSqueeze,
    'MRNet-Res': MRNetRes,
    'MRNet-VGG-Fixed': MRNetVGGFixed,
    'MRNet-Res-Fixed': MRNetResFixed,
}
