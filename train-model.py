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

# Data transformation & loading
# Note that due to the rotation, having several sequential shearing
# transforms sequentially is not the same as having only one with
# a larger range.
trans = transforms.Compose([
    transforms.RandomAffine(2, shear=2),
    transforms.RandomAffine(2, shear=2),
    transforms.RandomAffine(2, shear=2),
    #transforms.RandomCrop(224),
    transforms.RandomResizedCrop(150, scale=(0.25, 1.0), ratio=(0.9, 1.11), interpolation=2),
    transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.3, hue=0.02),
    transforms.RandomGrayscale(p=0.75),
    QLoss(min_q=2, max_q=60),
    transforms.RandomChoice([
        transforms.RandomApply((Otsu(),), p=0.1),
        transforms.RandomApply((Sauvola(2, 8),), p=0.05)
    ]),
    transforms.ToTensor()
])

training = ImageFolder('lines/training', transform=trans)
training.target_transform = tgc.classMap.get_target_transform(training.class_to_idx)

validation = ImageFolder('lines/validation', transform=None)
validation.target_transform = tgc.classMap.get_target_transform(validation.class_to_idx)
best_validation = 0

data_loader = torch.utils.data.DataLoader(training,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=4)

# Iterating over the data
print('Starting the training - grab a coffee and a good book!')
for epoch in range(30):
    # Modify learning rate
    scheduler.step()
    
    # Iterate over the data
    lossSum = 0
    good = 0
    known = 0
    tgc.network.train()
    for sample, label in tqdm(data_loader, desc='Training'):
        # Move data to device
        sample = sample.to(tgc.dev)
        label  = label.to(tgc.dev)
        
        # Training the classifier on samples with known labels
        sample, label = tgc.filter(sample, label)
        if len(label)==0: # no known labels
            continue
        out = tgc.network(sample)
        closs = classification_loss(out, label)
        optimizer.zero_grad()
        closs.backward()
        optimizer.step()
        lossSum += closs.item()
        
        # Computing accuracy
        _, p = torch.max(out, 1)
        good += torch.sum(p==label).item()
        known += len(label)
    print('Epoch %d, loss %.1f, %d/%d=%.1f%%' % (epoch, lossSum, good, known, (100.0*good)/known))
    
    targets = list()
    results = list()
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
            targets.append(target)
            results.append(label)
            if target==label:
                good += 1
            else:
                bad += 1
    with open('results.dat', 'wb') as f:
        pickle.dump(targets, f)
        pickle.dump(results, f)
    
    accuracy = 100*good/float(good+bad)
    
    print('    Good:', good)
    print('     Bad:', bad)
    print('Accuracy:', accuracy)
    
    if accuracy>best_validation:
        tgc.save(os.path.join('ocrd_typegroups_classifier', 'models', 'classifier.tgc'))
        best_validation = accuracy
        print('Network saved')
    else:
        tgc.save(os.path.join('ocrd_typegroups_classifier', 'models', 'classifier-last.tgc'))


































