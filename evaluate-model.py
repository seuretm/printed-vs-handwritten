import os
import sys
import torch
import pickle
from torch import nn
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.append("../ocrd_typegroups_classifier")

from ocrd_typegroups_classifier.network.vraec import vraec101
from ocrd_typegroups_classifier.network.vraec import vraec50
from ocrd_typegroups_classifier.typegroups_classifier import TypegroupsClassifier
from ocrd_typegroups_classifier.data.qloss import QLoss
from ocrd_typegroups_classifier.data.binarization import Otsu
from ocrd_typegroups_classifier.data.binarization import Sauvola
from ocrd_typegroups_classifier.network.densenet import densenet121
from ocrd_typegroups_classifier.network.resnet import resnet18

# Loading and preparing the network
net = densenet121(num_classes=2)
#net = resnet18(num_classes=12)

# Some settings for the training
learning_rate = 0.1
#weight_decay = 0.0001
weight_decay = 0
lr_decay = lambda epoch: 0.97 ** epoch
reconstruction_loss = nn.MSELoss()
classification_loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = LambdaLR(optimizer, lr_lambda=[lr_decay])

# Creation of the typegroup classifier
tgc = TypegroupsClassifier(
    {
        'handwritten':0,
        'printed':1
    },
    net
)
if os.path.exists(os.path.join('ocrd_typegroups_classifier', 'models', 'classifier.tgc')):
    tgc = TypegroupsClassifier.load(os.path.join('ocrd_typegroups_classifier', 'models', 'classifier.tgc'))
else:
    print('Could not load a model to evaluate')
    quit(1)

validation = ImageFolder('lines/validation', transform=None)
validation.target_transform = tgc.classMap.get_target_transform(validation.class_to_idx)
good = 0
bad  = 0
with torch.no_grad():
    tgc.network.eval()
    for idx in tqdm(range(validation.__len__()), desc='Evaluation'):
        sample, target = validation.__getitem__(idx)
        path, _ = validation.samples[idx]
        if target==-1:
            continue
        result = tgc.classify(sample, 224, 64, True)
        highscore = max(result)
        label = tgc.classMap.cl2id[result[highscore]]
        if target==label:
            good += 1
        else:
            bad += 1

accuracy = 100*good/float(good+bad)

print('    Good:', good)
print('     Bad:', bad)
print('Accuracy:', accuracy)
