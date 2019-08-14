# -*- coding: utf-8 -*-
import sys

from model import *

MODELS = {
    'MRNet': MRNet,
    'MRNet-Squeeze': MRNetSqueeze,
    'MRNet-Attend': MRNetAttention,
    'MRNet-SqueezeAttend': MRNetSqueezeAttention,
}


if __name__ == '__main__':
    model_name = sys.argv[1]

    model = MODELS[model_name]()

    print(model)
