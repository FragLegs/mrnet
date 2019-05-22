# -*- coding: utf-8 -*-
import sys

from model import *

MODELS = {
    'MRNet': MRNet,
    'MRNet-VGG': MRNetVGG,
    # 'MRNet-Dense': MRNetDense,
    'MRNet-Squeeze': MRNetSqueeze,
    'MRNet-Res': MRNetRes,
    'MRNet-VGG-Fixed': MRNetVGGFixed,
    'MRNet-Res-Fixed': MRNetResFixed,
    'MRNet-Res-7': MRNetRes7,
    'MRNet-Res-7-1': MRNetRes7_1,
}


if __name__ == '__main__':
    model_name = sys.argv[1]

    model = MODELS[model_name]()
    print(model)
