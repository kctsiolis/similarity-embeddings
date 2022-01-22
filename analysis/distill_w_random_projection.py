import torch
import numpy
from sklearn import random_projection
from models.models import get_model, WrapWithProjection
from training.loaders import dataset_loader


# Start with a teacher model / preferably trained with sim. based embedding to be more robust to projections

teacher = get_model('resnet50_cifar_embedder',load = True,load_path = '/mnt/data/scratch/data/distilled_models/similarity_distillation_resnet50/model.pt',map_location='cpu')
student = get_model('resnet18_cifar_embedder',load = False)
proj_dim  =student.dim
print(f'Projecting teacher model from dimension {teacher.dim} to {student.dim} ')

teacher = WrapWithProjection(teacher,teacher.dim,proj_dim)


data = torch.randn([1,3,32,32])
target = torch.randn([1, proj_dim])

print(teacher(data).shape)



# Let the embeddings be given by E (n_samplexn_features) and P a random projection. Then goal is to learn PxE the projected embeddings. Using a much smaller model 