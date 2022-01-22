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
    def __init__(self, num_classes, pretrained=False):
        """Instantiate object of class ResNet18.
        
        Args:
            num_classes: Number of classes (for applying model to classification task).
            pretrained: Whether or not to get pretrained model from Torch.

        """
        super().__init__()
        if pretrained:
            self.model = resnet18(pretrained=True)
        else:
            self.model = resnet18(num_classes=num_classes)
        self.dim = 512

    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module):
    """Wrapper class for the ResNet50 model (imported from Torch).

    Attributes:
        model: The Torch model.
        dim: Dimension of the last embedding layer

    """
    def __init__(self, num_classes, pretrained=False):
        """Instantiate object of class ResNet50.
        
        Args:
            num_classes: Number of classes (for applying model to classification task).
            pretrained: Whether or not to get pretrained model from Torch.

        """
        super().__init__()
        if pretrained:
            self.model = resnet50(pretrained=True)
        else:
            self.model = resnet50(num_classes=num_classes)
        self.dim = 2048

    def forward(self, x):
        return self.model(x)

class ResNet152(nn.Module):
    """Wrapper class for the ResNet152 model (imported from Torch).

    Attributes:
        model: The Torch model.
        dim: Dimension of the last embedding layer

    """
    def __init__(self, num_classes, pretrained=False):
        """Instantiate object of class ResNet152.
        
        Args:
            num_classes: Number of classes (for applying model to classification task).
            pretrained: Whether or not to get pretrained model from Torch.

        """
        super().__init__()
        if pretrained:
            self.model = resnet152(pretrained=True)
        else:
            self.model = resnet152(num_classes=num_classes)
        self.dim = 2048

    def forward(self, x):
        return self.model(x)

class Embedder(nn.Module):
    """Wrapper class for a feature embedder (model w/o classification layer).

    Attributes:
        features: The feature embedder.
        dim: Embedding dimension.

    """
    def __init__(self, model, dim=None, project=False):
        """Instantiate object of class Embedder.

        Args:
            model: A model with a classification layer (which will be removed).
            dim: The desired projected embedding dimension.

        """
        super().__init__()
        #Get the embedding layers from the given model
        #The attribute containing the model's layers may go by different names
            
        self.features = nn.Sequential(*list(model.children())[:-1])

        if dim is None:
            self.dim = model.dim
        else:
            self.dim = dim

        self.project = project
        if project:
            self.projection = nn.Sequential(
                nn.Linear(model.dim, self.dim),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim)
            )
                
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        if self.project:
            x = self.projection(x)
        return x

    def get_dim(self):
        return self.dim

    def student_mode(self):
        for param in self.features.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True

    def teacher_mode(self):
        for param in self.features.parameters():
            param.requires_grad = False
        self.project = False

class EmbedderAndLogits(nn.Module):
    """
    Wrapper for model to return both embeddings and logits
    """
    def __init__(self, model):
        """Instantiate object .

        Args:
            model: A model with a classification layer (which will be removed).
            dim: The embedding dimension of the model.

        """
        super().__init__()
        #Get the embedding layers from the given model
        #The attribute containing the model's layers may go by different names         
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.lc = list(model.children())[-1]            
            
        self.dim = model.dim
                
    def forward(self, x):
        x = self.features(x)
        features = torch.flatten(x, 1)

        logits = self.lc(features)

        return features, logits

    def get_dim(self):
        return self.dim        

    def get_features(self):
        return self.features

    def student_mode(self):
        for param in self.features.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True

    def teacher_mode(self):
        for param in self.features.parameters():
            param.requires_grad = False
        self.project = False



class TruncatedNet(nn.Module):
    """A wrapper class to truncate another model. (A slight reskin of the Embedder class)            

    Attributes:
        features: The feature embedder.
        dim: Embedding dimension.

    """
    def __init__(self, model, n):
        """Instantiate object of class Embedder.

        Args:
            model: A model with a classification layer (which will be removed).
            n: number of layers to shave off
        
        """
        super().__init__()
        #Get the embedding layers from the given model
        #The attribute containing the model's layers may go by different names
        self.features = nn.Sequential(*list(model.model.children())[:-n])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)    
        return x 

class MLP(nn.Module):
    ''' A basic 3 layer MLP '''

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32) -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x             

class ConvNetEmbedder(nn.Module):
    """Wrapper class for simple CNN embedder.
    
    Attributes:
        conv1: First conv layer.
        conv2: Second conv layer.
        pool: Pooling layer.
        dim: Embedding dimension.

    """
    def __init__(self):
        """Instantiate ConvNetEmbedder object.
        
        Args:

        """
        super().__init__()
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
    def __init__(self, embedder, num_classes):
        """Instantiate the classifier object.
        
        Args:
            embedder: The embedder.
            num_classes: The number of classes.

        """
        super().__init__()
        self.embedder = embedder
        self.features = embedder.get_features()
        #Freeze the embedding layer
        for param in self.embedder.parameters():
            param.requires_grad = False
        #Classification layer
        self.linear_layer = nn.Linear(embedder.get_dim(), num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.linear_layer(x)

        return x

class WrapWithProjection(nn.Module):
    def __init__(self,model:nn.Module,model_output_dim:int,projection_dim:int) -> None:
        super().__init__()
        self.model = model
        self.projection = nn.Parameter(torch.randn([model_output_dim,projection_dim]) / projection_dim)
        self.projection.requires_grad = False
    
    def forward(self,x):        
        x = torch.matmul(self.model(x), self.projection)
        return x

def get_model(model_str: str, load: bool = False, load_path: str = None, 
    num_classes: int = 10, get_embedder: bool = False, project_embedder: bool = False, 
    projection_dim: int = None, truncate_model : bool = False, truncation_level : int = -1) -> nn.Module:
    """Instantiate or load a specified model.
    
    
    Args:
        model_str: String specifying the model.
        load: Set to true to load existing model, otherwise instantiate a new one.
        load_path: Path to load model from (if we are loading).
        num_classes: Number of classes (if we are classifying).
        get_embedder: Whether or not to only get the feature embedding layers.

    Returns:
        The desired model.
    
    """    

    if get_embedder and truncate_model:
        raise ValueError('Cannot get embedder and truncate model')
    if truncate_model and truncation_level == -1:
        raise ValueError('Need to specify a truncation level if truncating the model')
    
    if model_str == 'mlp':
        model = MLP(input_dim=32*32*3, num_classes=num_classes,hidden_dim=256)
        dim = 256    
    elif model_str == 'resnet18':
        model = ResNet18(num_classes=num_classes)
        dim = 512    
    elif model_str == 'resnet50':
        model = ResNet50(num_classes=num_classes)
        dim = 2048
    elif model_str == 'resnet152':
        model = ResNet152(num_classes=num_classes)
        dim = 2048
    elif model_str == 'resnet18_embedder':
        model = Embedder(ResNet18(num_classes=num_classes),project=project_embedder, dim=projection_dim)
    elif model_str == 'resnet50_embedder':
        model = Embedder(ResNet50(num_classes=num_classes),project=project_embedder, dim=projection_dim)
    elif model_str == 'resnet50_classifier':
        model = Classifier(Embedder(ResNet50(num_classes=num_classes)),num_classes=num_classes)        
    elif model_str == 'resnet18_classifier':        
        model = Classifier(Embedder(ResNet18(num_classes=num_classes)),num_classes=num_classes)        
    elif model_str == 'convnet_embedder':
        model = ConvNetEmbedder()
    elif model_str == 'resnet18_pretrained':
        model = ResNet18(pretrained=True)
        dim = 512
    elif model_str == 'resnet18_pretrained_embedder':
        model = Embedder(ResNet18(pretrained=True),dim=512,project=project_embedder,dim=projection_dim)        
    elif model_str == 'resnet50_pretrained':
        model = ResNet50(pretrained=True)
        dim = 2048
    elif model_str == 'resnet152_pretrained':
        model = ResNet152(pretrained=True)
        dim = 2048
    elif model_str == 'simclr_pretrained':
        model = ResNet50(num_classes=1000)
        dim = 2048        
    elif model_str == 'resnet_small_cifar':
        model = cifar_models.ResNet3Layer(num_classes=num_classes)
        dim = 256
    elif model_str == 'resnet_very_small_cifar':
        model = cifar_models.ResNet2Layer(num_classes=num_classes)
        dim = 128
    elif model_str == 'resnet_small_cifar_embedder':
        model = Embedder(
            cifar_models.ResNet3Layer(num_classes=num_classes), dim=256, project=project_embedder)           
    elif model_str == 'resnet_very_small_cifar_embedder':
        model = Embedder(cifar_models.ResNet2Layer(num_classes=num_classes), dim=128,project=project_embedder)
    elif model_str == 'resnet_small_cifar_classifier':
        model = Classifier(Embedder(cifar_models.ResNet3Layer(num_classes=num_classes),project=project_embedder))
    elif model_str == 'resnet_very_small_cifar_classifier':
        model = Classifier(Embedder(cifar_models.ResNet2Layer(num_classes=num_classes),project=project_embedder))
    elif model_str == 'simple':
        model = cifar_models.SuperSimpleNet()             
        dim = 84
    elif model_str == 'simple_embedder':
        model = Embedder(cifar_models.SuperSimpleNet(), dim=84,project=project_embedder)                       
    elif model_str == 'resnet18_cifar':
        model = cifar_models.ResNet18(num_classes=num_classes)
        dim = 512
    elif model_str == 'resnet18_cifar_embedder':
        model = Embedder(
            cifar_models.ResNet18(num_classes=num_classes), dim=512,project=project_embedder)
    elif model_str == 'resnet18_cifar_classifier':
        model = Classifier(Embedder(
            cifar_models.ResNet18(num_classes=num_classes), dim=512,project=project_embedder),num_classes=num_classes) 
    elif model_str == 'resnet50_cifar':
        model = cifar_models.ResNet50(num_classes=num_classes)
        dim = 2048
    elif model_str == 'resnet50_cifar_embedder':
        model = Embedder(
            cifar_models.ResNet50(num_classes=num_classes), dim=2048,project=project_embedder)
    elif model_str == 'resnet50_cifar_classifier':
        model = Classifier(Embedder(
            cifar_models.ResNet50(num_classes=num_classes), dim=2048,project=project_embedder))                   
    else:
        raise ValueError('Model {} not defined.'.format(model_str))
        
    if load_path is not None:
        try:
            model.load_state_dict(torch.load(load_path)['state_dict'])
        except IndexError:
            model.load_state_dict(torch.load(load_path))
    
    if truncate_model:             
        model = TruncatedNet(model,n=truncation_level,dim=dim)

    if get_embedder:        
        model = Embedder(model, dim=dim,project=False)

    return model