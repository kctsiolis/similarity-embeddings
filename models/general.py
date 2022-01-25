

from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class CLIPDistill(nn.Module):
    def __init__(self,
                 # teacher
                 teacher_dimension: int,
                 teacher_model: nn.Module,
                 # student
                 student_dimension: int,
                 student_model: nn.Module,
                 ):
        super().__init__()

        self.teacher = teacher_model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.student = student_model

        self.student_projection = nn.Sequential(nn.Linear(student_dimension, student_dimension, bias=False), nn.ReLU(
        ), nn.Linear(student_dimension, student_dimension, bias=False))

#         self.teacher_projection = nn.Parameter(torch.randn(teacher_dimension,embed_dim))

        if teacher_dimension == student_dimension:
            self.register_buffer('teacher_projection',
                                 torch.eye(teacher_dimension))
        else:
            self.register_buffer('teacher_projection', torch.randn(
                teacher_dimension, student_dimension))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def student_encoding(self, x):        
        return self.student_projection(self.student(x))

    def teacher_encoding(self, x):
        return self.teacher(x) @ self.teacher_projection

    def forward(self, x):
        student_features = self.student_encoding(x)
        
        with torch.no_grad():
            teacher_features = self.teacher_encoding(x)
            teacher_features = teacher_features / \
            teacher_features.norm(dim=-1, keepdim=True)

        # normalized features
        student_features = student_features / \
            student_features.norm(dim=-1, keepdim=True)
        

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        student_logits = logit_scale * student_features @ teacher_features.t()
        teacher_logits = student_logits.t()

        # shape = [global_batch_size, global_batch_size]
        return teacher_logits, student_logits

