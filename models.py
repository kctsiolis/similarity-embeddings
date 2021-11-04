"""Collection of classes and functions for neural network vision models."""

from numpy.core.defchararray import mod
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet152
import cifar_models

class ResNet18(nn.Module):
    """Wrapper class for the ResNet18 model (imported from Torch).

    Attributes:
        model: The Torch model.
        dim: Dimension of the last embedding layer

    """
    def __init__(self, num_classes=10, one_channel=False, pretrained=False):
        """Instantiate object of class ResNet18.
        
        Args:
            num_classes: Number of classes (for applying model to classification task).
            one_channel: Whether or not input data has one colour channel (for MNIST).
            pretrained: Whether or not to get pretrained model from Torch.

        """
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
    """Wrapper class for the ResNet50 model (imported from Torch).

    Attributes:
        model: The Torch model.
        dim: Dimension of the last embedding layer

    """
    def __init__(self, num_classes=10, one_channel=False, pretrained=False):
        """Instantiate object of class ResNet50.
        
        Args:
            num_classes: Number of classes (for applying model to classification task).
            one_channel: Whether or not input data has one colour channel (for MNIST).
            pretrained: Whether or not to get pretrained model from Torch.

        """
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
    """Wrapper class for the ResNet152 model (imported from Torch).

    Attributes:
        model: The Torch model.
        dim: Dimension of the last embedding layer

    """
    def __init__(self, num_classes=10, one_channel=False, pretrained=False):
        """Instantiate object of class ResNet152.
        
        Args:
            num_classes: Number of classes (for applying model to classification task).
            one_channel: Whether or not input data has one colour channel (for MNIST).
            pretrained: Whether or not to get pretrained model from Torch.

        """
        super().__init__()
        if pretrained:
            self.model = resnet152(pretrained=True)
        else:
            self.model = resnet152(num_classes=num_classes)
        self.dim = 2048
        if one_channel:
            #Set number of input channels to 1 (since MNIST images are greyscale)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    def forward(self, x):
        return self.model(x)

    def get_dim(self):
        return self.dim

class Embedder(nn.Module):
    """Wrapper class for a feature embedder (model w/o classification layer).

    Attributes:
        features: The feature embedder.
        dim: Embedding dimension.

    """
    def __init__(self, model, dim=None, batchnormalize=False,
        track_running_stats=True):
        """Instantiate object of class Embedder.

        Args:
            model: A model with a classification layer (which will be removed).
            dim: The embedding dimension of the model.
            batchnormalize: Whether or not to add batch norm after feature embedding.
            track_running_stats: Which statistics to use for batch norm (if applicable).

        """
        super().__init__()
        #Get the embedding layers from the given model
        #The attribute containing the model's layers may go by different names
        try:
            self.features = nn.Sequential(*list(model.model.children())[:-1])
        except AttributeError:
            self.features = nn.Sequential(*list(model.children())[:-1])

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
            self.final_normalization = nn.BatchNorm1d(self.dim, affine=False,
                track_running_stats=track_running_stats)
                
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        if self.batchnormalize:
            x = self.final_normalization(x) #Normalize the output
        return x

    def get_dim(self):
        return self.dim


class TruncatedNet(nn.Module):
    """A wrapper class to truncate another model. (A slight reskin of the Embedder class)            

    Attributes:
        features: The feature embedder.
        dim: Embedding dimension.

    """
    def __init__(self, model,n, dim=None, batchnormalize=False,
        track_running_stats=True):
        """Instantiate object of class Embedder.

        Args:
            model: A model with a classification layer (which will be removed).
            n: number of layers to shave off
            dim: The embedding dimension of the model.
            batchnormalize: Whether or not to add batch norm after feature embedding.
            track_running_stats: Which statistics to use for batch norm (if applicable).
        
        """
        super().__init__()
        #Get the embedding layers from the given model
        #The attribute containing the model's layers may go by different names
        try:
            self.features = nn.Sequential(*list(model.model.children())[:-n])
        except AttributeError:
            self.features = nn.Sequential(*list(model.children())[:-n])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
            self.final_normalization = nn.BatchNorm1d(self.dim, affine=False,
                track_running_stats=track_running_stats)
                
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.batchnormalize:
            x = self.final_normalization(x) #Normalize the output        
        return x

    def get_dim(self):
        return self.dim        

class ConvNetEmbedder(nn.Module):
    """Wrapper class for simple CNN embedder.
    
    Attributes:
        conv1: First conv layer.
        conv2: Second conv layer.
        pool: Pooling layer.
        dim: Embedding dimension.

    """
    def __init__(self, one_channel=False):
        """Instantiate ConvNetEmbedder object.
        
        Args:
            one_channel: Whether or not input has one colour channel (for MNIST).

        """
        super().__init__()
        if one_channel:
            self.conv1 = nn.Conv2d(1, 6, 5)
        else:
            self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dim = 16 * 6 * 5

    def forward(self, x):
        """Get the embeddings."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        return x

    def get_dim(self):
        return self.dim

class Projection(nn.Module):
    def __init__(self,model: nn.Module,model_dim:int, projection_dim :int) -> None:
        super().__init__()
        self.model = model
        self.projection = torch.randn([model_dim,projection_dim]) / projection_dim
        self.projection.requires_grad = False

    def forward(self,x):
        x = torch.matmul(self.model(x),self.projection)
        return x


class Classifier(nn.Module):
    """Wrapper class for a classifier on top of an embedder.
    
    Attributes:
        embedder: The embedder (frozen).
        linear_layer: The new linear layer on top of the embedder.

    """
    def __init__(self, embedder, num_classes=10):
        """Instantiate the classifier object.
        
        Args:
            embedder: The embedder.
            num_classes: The number of classes.

        """
        super().__init__()
        self.embedder = embedder
        #Freeze the embedding layer
        for param in self.embedder.parameters():
            param.requires_grad = False
        #Classification layer
        # self.linear_layer = nn.Linear(embedder.get_dim(), num_classes)
        self.linear_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedder(x)
        x = self.linear_layer(x)

        return x

def get_model(model_str: str, load: bool = False, load_path: str = None, 
    one_channel: bool = False, num_classes: int = 10, get_embedder: bool = False, truncate_model : bool = False,truncation_level : int = -1,
    batchnormalize: bool = False, track_running_stats : bool =True, map_location = None) -> nn.Module:
    """Instantiate or load a specified model.
    
    Args:
        model_str: String specifying the model.
        load: Set to true to load existing model, otherwise instantiate a new one.
        load_path: Path to load model from (if we are loading).
        one_channel: Whether or not model sees only one colour channel.
        num_classes: Number of classes (if we are classifying).
        get_embedder: Whether or not to only get the feature embedding layers.
        batchnormalize: Whether or not to apply batch norm to the embeddings.
        track_running_stats: Which statistics to use for batch norm (if applicable).
        map_location: if 'load' the device to load the model onto

    Returns:
        The desired model.
    
    """
    checkpointing = False

    if get_embedder and truncate_model:
        raise ValueError('Cannot get embedder and truncate model')
    if truncate_model and truncation_level == -1:
        raise ValueError('Need to specify a truncation level if truncating the model')
    
    if model_str == 'resnet18':
        model = ResNet18(one_channel=one_channel, num_classes=num_classes)
        dim = 512    
    elif model_str == 'resnet50':
        model = ResNet50(one_channel=one_channel, num_classes=num_classes)
        dim = 2048
    elif model_str == 'resnet152':
        model = ResNet152(one_channel=one_channel, num_classes=num_classes)
        dim = 2048
    elif model_str == 'resnet18_embedder':
        model = Embedder(ResNet18(one_channel=one_channel, num_classes=num_classes), 
            batchnormalize=batchnormalize, track_running_stats=track_running_stats)
    elif model_str == 'resnet50_embedder':
        model = Embedder(ResNet50(one_channel=one_channel, num_classes=num_classes), 
            batchnormalize=batchnormalize, track_running_stats=track_running_stats)
    elif model_str == 'resnet50_classifier':
        model = Classifier(Embedder(ResNet50(one_channel=one_channel,
            num_classes=num_classes), batchnormalize=batchnormalize, 
            track_running_stats=track_running_stats),num_classes=num_classes)        
    elif model_str == 'resnet18_classifier':        
        model = Classifier(Embedder(ResNet18(one_channel=one_channel,
            num_classes=num_classes), batchnormalize=batchnormalize, 
            track_running_stats=track_running_stats),num_classes=num_classes)        
    elif model_str == 'convnet_embedder':
        model = ConvNetEmbedder(one_channel=one_channel)
    elif model_str == 'resnet18_pretrained':
        load = False
        model = ResNet18(one_channel=one_channel, pretrained=True)
        dim = 512
    elif model_str == 'resnet50_pretrained':
        load = False
        model = ResNet50(one_channel=one_channel, pretrained=True)
        dim = 2048
    elif model_str == 'resnet152_pretrained':
        load = False
        model = ResNet152(one_channel=one_channel, pretrained=True)
        dim = 2048
    elif model_str == 'simclr_pretrained':
        assert load == True
        checkpointing = True
        checkpoint = torch.load(load_path)
        model = ResNet50(num_classes=1000)
        dim = 2048        
    elif model_str == 'resnet_small_cifar':
        model = cifar_models.ResNet3Layer(num_classes=num_classes)
        dim = 256
    elif model_str == 'resnet_small_cifar_embedder':
        model = Embedder(cifar_models.ResNet3Layer(num_classes=num_classes), dim=256, batchnormalize=batchnormalize,
            track_running_stats=track_running_stats)   
    elif model_str == 'simple':
        model = cifar_models.SuperSimpleNet()             
        dim = 84
    elif model_str == 'simple_embedder':
        model = Embedder(cifar_models.SuperSimpleNet(), dim=84, batchnormalize=batchnormalize,
            track_running_stats=track_running_stats)                       
    elif model_str == 'resnet18_cifar':
        model = cifar_models.ResNet18(num_classes=num_classes)
        dim = 512
    elif model_str == 'resnet18_cifar_embedder':
        model = Embedder(
            cifar_models.ResNet18(num_classes=num_classes), dim=512,
            batchnormalize=batchnormalize, track_running_stats=track_running_stats)
    elif model_str == 'resnet18_cifar_classifier':
        model = Classifier(Embedder(
            cifar_models.ResNet18(num_classes=num_classes), dim=512,
            batchnormalize=batchnormalize, track_running_stats=track_running_stats),num_classes=num_classes) 
    elif model_str == 'resnet50_cifar':
        model = cifar_models.ResNet50(num_classes=num_classes)
        dim = 2048
    elif model_str == 'resnet50_cifar_embedder':
        model = Embedder(
            cifar_models.ResNet50(num_classes=num_classes), dim=2048,
            batchnormalize=batchnormalize, track_running_stats=track_running_stats)
    elif model_str == 'resnet50_cifar_classifier':
        model = Classifier(Embedder(
            cifar_models.ResNet50(num_classes=num_classes), dim=2048,
            batchnormalize=batchnormalize, track_running_stats=track_running_stats))                   
    else:
        raise ValueError('Model {} not defined.'.format(model_str))
        
    if load:
        if checkpointing:
            try:
                model.model.load_state_dict(checkpoint['state_dict'])
            except AttributeError:
                model.load_state_dict(checkpoint['state_dict'])
        else:                        
            model.load_state_dict(torch.load(load_path,map_location=map_location))
    
    if truncate_model:             
        model = TruncatedNet(model,n=truncation_level,dim=dim,batchnormalize=batchnormalize,track_running_stats=track_running_stats)

    if get_embedder:        
        model = Embedder(model, dim=dim, batchnormalize=batchnormalize,
            track_running_stats=track_running_stats)

    return model

class WrapWithProjection(nn.Module):
    def __init__(self,model:nn.Module,model_output_dim:int,projection_dim:int) -> None:
        super().__init__()
        self.model = model
        self.projection = nn.Parameter(torch.randn([model_output_dim,projection_dim]) / projection_dim)
        self.projection.requires_grad = False
    
    def forward(self,x):        
        x = torch.matmul(self.model(x), self.projection)
        return x


