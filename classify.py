import os
import sys
import torch
import pickle
from torch import nn
from math import exp
from PIL import Image
from tqdm import tqdm
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import LambdaLR
sys.path.append("../ocrd_typegroups_classifier")
from ocrd_typegroups_classifier.typegroups_classifier import TypegroupsClassifier
from ocrd_typegroups_classifier.network.densenet import densenet121
from ocrd_typegroups_classifier.data.binarization import Sauvola
from ocrd_typegroups_classifier.data.binarization import Otsu
from ocrd_typegroups_classifier.data.qloss import QLoss


if len(sys.argv)!=2:
    print('Syntax: python3 %s input-textline.jpg' % sys.argv[0])

img = Image.open(sys.argv[1])
tgc = TypegroupsClassifier.load(os.path.join('ocrd_typegroups_classifier', 'models', 'classifier.tgc'))

result = tgc.classify(img, 75, 64, False)
esum = 0
for key in result:
    esum += exp(result[key])
for key in result:
    result[key] = exp(result[key]) / esum
print(result)
