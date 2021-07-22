import torch
from torch import nn
from torch._C import autocast_decrement_nesting
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet152

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, one_channel=False, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = resnet18(pretrained=True)
        else:
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
    def __init__(self, num_classes=10, one_channel=False, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = resnet50(pretrained=True)
        else:
            self.model = resnet50(num_classes=num_classes)
        self.dim = 2048
        if one_channel:
            #Set number of input channels to 1 (since MNIST images are greyscale)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    def forward(self, x):
        return self.model(x)

    def get_dim(self):
        return self.dim

class ResNet152(nn.Module):
    def __init__(self, num_classes=10, one_channel=False):
        super().__init__()
        self.model = resnet152(num_classes=num_classes)
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
    def __init__(self, model, dim=None, batchnormalize=False):
        super().__init__()
        #Get the embedding layers from the given model
        #The attribute containing the model's layers may go by different names
        try:
            self.features = nn.Sequential(*list(model.model.children())[:-1])
        except AttributeError:
            self.features = nn.Sequential(*list(model.children())[:-1])
        except AttributeError:
            self.features = nn.Sequential(*list(model.backbone.children())[:-1])

        try:
            self.dim = model.dim
        except AttributeError:
            if dim is None:
                raise ValueError('Must specify the model embedding dimension.')
            else:
                self.dim = dim

        #Whether or not to batch norm the features at the end
        self.batchnormalize = batchnormalize
        if batchnormalize:
            self.final_normalization = nn.BatchNorm1d(self.dim, affine=False)
                
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        if self.batchnormalize:
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

class ResNetSimCLR(nn.Module):
    """Class for the SimCLR model.

    From PyTorch SimCLR repository.
    https://github.com/sthalles/SimCLR/blob/master/  

    """

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.dim = dim_mlp

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        return self.resnet_dict[model_name]

    def forward(self, x):
        return self.backbone(x)

def get_model(model_str: str, load: bool = False, load_path: str = None, 
    one_channel: bool = False, num_classes: int = 10, get_embedder: bool = False, 
    batchnormalize: bool = False):
    if load and load_path is None:
        raise ValueError('Must specify path to load model from.')

    if model_str == 'resnet18':
        model = ResNet18(one_channel=one_channel, num_classes=num_classes)
        if load:
            model.model.load_state_dict(torch.load(load_path))
        if get_embedder:
            model = Embedder(model)
    elif model_str == 'resnet50':
        model = ResNet50(one_channel=one_channel, num_classes=num_classes)
        if load:
            model.model.load_state_dict(torch.load(load_path))
        if get_embedder:
            model = Embedder(model)
    elif model_str == 'resnet152':
        model = ResNet152(one_channel=one_channel, num_classes=num_classes)
        if load:
            model.model.load_state_dict(torch.load(load_path))
        if get_embedder:
            model = Embedder(model)
    elif model_str == 'resnet18_embedder':
        model = Embedder(ResNet18(one_channel=one_channel, num_classes=num_classes), 
            batchnormalize=batchnormalize)
        if load:
            model.load_state_dict(torch.load(load_path))
    elif model_str == 'resnet50_embedder':
        model = Embedder(ResNet50(one_channel=one_channel, num_classes=num_classes), 
            batchnormalize=batchnormalize)
        if load:
            model.load_state_dict(torch.load(load_path))
    elif model_str == 'convnet_embedder':
        model = ConvNetEmbedder(one_channel=one_channel)
        if load:
            model.load_state_dict(torch.load(load_path))
    elif model_str == 'resnet18_pretrained':
        model = ResNet18(one_channel=one_channel, pretrained=True)
        if get_embedder:
            model = Embedder(model)
    elif model_str == 'resnet50_pretrained':
        model = ResNet50(one_channel=one_channel, pretrained=True)
        if get_embedder:
            model = Embedder(model)
    elif model_str == 'simclr_pretrained':
        assert load == True
        checkpoint = torch.load(load_path)
        model = ResNet50(num_classes=1000)
        model.model.load_state_dict(checkpoint['state_dict'])
        if get_embedder:
            model = Embedder(model, batchnormalize=batchnormalize)
    elif model_str == 'simclr_pretrained_cifar':
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
        model = Embedder(model, dim=64, batchnormalize=batchnormalize)
    elif model_str == 'simclr':
        assert load == True
        checkpoint = torch.load(load_path)
        model = ResNetSimCLR(base_model='resnet18', out_dim=128)
        model.load_state_dict(checkpoint['state_dict'])
        if get_embedder:
            model = Embedder(model, dim=128, batchnormalize=batchnormalize)
    else:
        raise ValueError('Model {} not defined.'.format(model_str))

    return model

