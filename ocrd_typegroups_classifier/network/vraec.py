"""
    Model definition adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    The differences are:
        - insertion of variational convolutions in the layers
        - size adaptation by replacing the last avg pooling by an adaptive
          one
    - methods turning this class into an auto-encoder
"""
import logging
import math

import torch.nn as nn
from ocrd_typegroups_classifier.network.var_conv2d import VarConv2d

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def varConv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return VarConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _VariationalBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_VariationalBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU()
        self.conv2 = varConv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        
        self.varloss = 0

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out, self.varloss = self.conv2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class _VariationalBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_VariationalBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = VarConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        
        self.varloss = 0

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out, self.varloss = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class _VRAEC(nn.Module):

    def __init__(self, block, layers, layer_size=64, output_channels=256):
        self.inplanes = 64
        super(_VRAEC, self).__init__()
        
        if isinstance(layer_size, int):
            layer_size = (layer_size, layer_size, layer_size, layer_size)

        self.expected_input_size = (224, 224)
        self.layer_size = layer_size
        self.block = block

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.maxpool = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.layer1, self.vl1 = self._make_layer(block, layer_size[0], layers[0])
        self.layer2, self.vl2 = self._make_layer(block, layer_size[1], layers[1], stride=2)
        self.layer3, self.vl3 = self._make_layer(block, layer_size[2], layers[2], stride=2)
        self.layer4, self.vl4 = self._make_layer(block, layer_size[3], layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(layer_size[3] * block.expansion, output_channels)

        
        self.deconv1    = nn.ConvTranspose2d(64, 3, kernel_size=8, stride=2, padding=3)
        self.unpool     = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)
        self.unlayer1   = nn.ConvTranspose2d(block.expansion * layer_size[0], 64, kernel_size=3, padding=1)
        self.unlayer2   = nn.ConvTranspose2d(block.expansion * layer_size[1], block.expansion * layer_size[0], kernel_size=4, stride=2, padding=1)
        self.unlayer3   = nn.ConvTranspose2d(block.expansion * layer_size[2], block.expansion * layer_size[1], kernel_size=4, stride=2, padding=1)
        self.unlayer4   = nn.ConvTranspose2d(block.expansion * layer_size[3], block.expansion * layer_size[2], kernel_size=4, stride=2, padding=1)
        
        
        self.ae_layers = {
            0: (self.conv1, self.deconv1),
            1: (self.unpool, ),
            2: (self.layer1, self.unlayer1),
            3: (self.layer2, self.unlayer2),
            4: (self.layer3, self.unlayer3),
            5: (self.layer4, self.unlayer4),
            6: (self.avgpool,),
            7: (self.fc,)
        }
        
        self.var_layers = {
            0: None,
            1: None,
            2: self.vl1,
            3: self.vl2,
            4: self.vl3,
            5: self.vl4,
            6: None # sentinel
        }

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        layers[0].conv2.is_variational = False
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            b = block(self.inplanes, planes)
            b.conv2.is_variational = False
            layers.append(b)

        return (nn.Sequential(*layers), b)

    def forward(self, x):
        # input size: torch.Size([2, 3, 224, 224])
        
        x = self.conv1(x)
        x = self.relu(x)
        # after convo1: torch.Size([2, 64, 112, 112])
        
        x = self.maxpool(x)
        # after maxpool: torch.Size([2, 64, 56, 56])

        x = self.layer1(x)
        vl = self.vl1.varloss
        # after layer 1: torch.Size([2, 96, 56, 56])
        
        x = self.layer2(x)
        vl = vl + self.vl2.varloss
        # after layer 2: torch.Size([2, 96, 28, 28])
        
        x = self.layer3(x)
        vl = vl + self.vl3.varloss
        # after layer 3: torch.Size([2, 96, 14, 14])
        
        x = self.layer4(x)
        vl = vl + self.vl4.varloss
        # after layer 4: torch.Size([2, 96, 7, 7])
        
        ap = self.avgpool(x)
        # after avgpool: torch.Size([2, 96, 1, 1])
        
        x = ap.view(ap.size(0), -1)
        x = self.fc(x)

        return x, vl, ap.detach()
    
    def encode(self, x, nb_layers):
        px = x
        x = self.relu(self.conv1(x))
        
        poolpos = None
        if nb_layers>=1:
            px = x
            x = self.maxpool(x)
        
        layers = {
            2: (self.layer1, 'l1', self.vl1),
            3: (self.layer2, 'l2', self.vl2),
            4: (self.layer3, 'l3', self.vl3),
            5: (self.layer4, 'l4', self.vl4),
            6: None
        }
        vl = 0
        for l in range(2, nb_layers+1):
            if layers[l] is None:
                break
            px = x
            x = layers[l][0](x)
            vl = vl + layers[l][2].varloss
        
        return x, px, vl
    
    def set_variational(self, lnum, status):
        layers = {
            2: self.vl1,
            3: self.vl2,
            4: self.vl3,
            5: self.vl4
        }
        if lnum in layers:
            l = layers[lnum]
            l.conv2.is_variational = status
        
    
    def decode(self, x, layers):
        if 5 in layers:
            x = self.relu(self.unlayer4(x))
        
        if 4 in layers:
            x = self.relu(self.unlayer3(x))
        
        if 3 in layers:
            x = self.relu(self.unlayer2(x))
        
        if 2 in layers:
            x = self.relu(self.unlayer1(x))
        
        if 1 in layers:
            x = self.unpool(x)
        
        if 0 in layers:
            x = self.tanh(self.deconv1(x))
        return x
    
    def train_ae(self, x, optimizer, loss_function, layer_num=6):
        enc, penc, vl = self.encode(x, layer_num)
        dec = self.decode(enc, (layer_num,))
        optimizer.zero_grad()
        loss = loss_function(dec, penc.detach())
        
        if not vl is 0:
            loss = loss + vl / (vl.item()/loss.item())
        loss.backward()
        optimizer.step()
        return loss.item()
        
    
    def finetune(self, x, optimizer, loss_function, layer_num=6):
        layers = range(layer_num+1)
        enc, _, vl = self.encode(x, layer_num)
        dec = self.decode(enc, layers)
        optimizer.zero_grad()
        loss = loss_function(dec, x)
        if not vl is 0:
            loss = loss + vl / (vl.item()/loss.item())
        loss.backward()
        optimizer.step()
        return loss.item()

    def select_parameters(self, layers=range(8)):
        res = list()
        
        for l in layers:
            for layer in self.ae_layers[l]:
                for p in layer.parameters():
                    res.append(p)
        return res


def vraec18(pretrained=False, **kwargs):
    """Constructs a _ResAE-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _VRAEC(_VariationalBasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vraec34(pretrained=False, **kwargs):
    """Constructs a _ResAE-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _VRAEC(_VariationalBasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vraec50(pretrained=False, **kwargs):
    """Constructs a _ResAE-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _VRAEC(_VariationalBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vraec101(pretrained=False, **kwargs):
    """Constructs a _ResAE-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _VRAEC(_VariationalBottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model


def vraec152(pretrained=False, **kwargs):
    """Constructs a _ResAE-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _VRAEC(_VariationalBottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
        except Exception as exp:
            logging.warning(exp)
    return model
