"""Based on implementation by Yerlan Idelbayev: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, project=False, projection_dim=None):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)        
        self.dim = 64

        self.project = project
        if project:
            if projection_dim is not None:
                projection_dim = self.dim
            self.projection = nn.Sequential(
                nn.Linear(self.dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim)
            )
        self.classify = True

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        if self.project:
            out = self.projection(out)
        if self.classify:
            out = self.linear(out)
        return out

    def embs_and_logits(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        embs = out.view(out.size(0), -1)
        if self.project:
            embs = self.projection(embs)
        
        logits = self.linear(embs)
        
        return embs, logits

    def student_mode(self):
        for param in self.parameters():
            param.requires_grad = True
        for param in self.linear.parameters():
            param.requires_grad = False
        self.classify = False

    def teacher_mode(self, classify : bool):
        for param in self.parameters():
            param.requires_grad = False
        self.classify = classify

    def probing_mode(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = True
        self.classify = True

    def end_to_end_mode(self):
        for param in self.parameters():
            param.requires_grad = True
        self.classify = True

def resnet20_cifar(num_classes, project=False, projection_dim=None):
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes, project, projection_dim)


def resnet32_cifar(num_classes, project=False, projection_dim=None):
    return ResNetCIFAR(BasicBlock, [5, 5, 5], num_classes, project, projection_dim)


def resnet44_cifar(num_classes, project=False, projection_dim=None):
    return ResNetCIFAR(BasicBlock, [7, 7, 7], num_classes, project, projection_dim)


def resnet56_cifar(num_classes, project=False, projection_dim=None):
    return ResNetCIFAR(BasicBlock, [9, 9, 9], num_classes, project, projection_dim)


def resnet110_cifar(num_classes, project=False, projection_dim=None):
    return ResNetCIFAR(BasicBlock, [18, 18, 18], num_classes, project, projection_dim)


def resnet1202_cifar(num_classes, project=False, projection_dim=None):
    return ResNetCIFAR(BasicBlock, [200, 200, 200], num_classes, project, projection_dim)