import torch
from torch import nn
from torchvision.models import resnet18, resnet50

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, one_channel=False):
        super().__init__()
        self.model = resnet18(num_classes=num_classes)
        self.dim = 512
        if one_channel:
            #Set number of input channels to 1 (since MNIST images are greyscale)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    def forward(self, x):
        return self.model(x)

    def get_dim(self):
        return self.dim

class ResNet50(nn.Module):
    def __init__(self, num_classes=10, one_channel=False):
        super().__init__()
        self.model = resnet50(num_classes=num_classes)
        self.dim = 2048
        if one_channel:
            #Set number of input channels to 1 (since MNIST images are greyscale)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    def forward(self, x):
        return self.model(x)

    def get_dim(self):
        return self.dim

#Model without linear classification layer
class Embedder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.model.children())[:-1])
        self.dim = model.dim

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def get_dim(self):
        return self.dim

#Model which applies additional batch normalization to learned features
class NormalizedEmbedder(Embedder):
    def __init__(self, model):
        super().__init__(model)
        self.dim = model.dim
        self.final_normalization = nn.BatchNorm1d(512, affine=False)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.final_normalization(x) #Normalize the output
        return x

    def get_dim(self):
        return self.dim

class ConvNetEmbedder(nn.Module):
    def __init__(self, one_channel=False):
        super().__init__()
        if one_channel:
            self.conv1 = nn.Conv2d(1, 6, 5)
        else:
            self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dim = 16 * 6 * 5

    #Get the embedding
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        return x

    #Return embedding dimension
    def get_dim(self):
        return self.dim

#Class for a classifier on top of an embedder
class Classifier(nn.Module):
    def __init__(self, embedder, num_classes=10):
        super().__init__()
        self.embedder = embedder
        #Freeze the embedding layer
        for param in self.embedder.parameters():
            param.requires_grad = False
        #Classification layer
        self.linear_layer = nn.Linear(embedder.get_dim(), num_classes)

    def forward(self, x):
        x = self.embedder(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)

        return x