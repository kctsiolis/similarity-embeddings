import torch
import torch.nn as nn
import torch.nn.functional as F
from training.data_augmentation import SimCLRTransform
import math

def get_similarity(model, data):
    embs = model(data)     
    embs = F.normalize(embs, p=2, dim=1)           
    return torch.matmul(embs, embs.transpose(0,1)) 

def get_margin_similarity(model, data, target, margin_value, margin_type):
    embs = model(data)        
    embs = F.normalize(embs, p=2, dim=1)           
    sims = torch.matmul(embs, embs.transpose(0,1))    
    
    #### Add margin to similarities
    # Using the identity cos(x +- m) = cos(x)cos(m) -+ sin(m)sin(x)                
    sine = torch.sqrt((1.0 - torch.pow(sims, 2)) + 1e-6).clamp(0,1) # add 1e-6 for numerical stability                                             

    # only apply margin to samples in different classes
    n = target.shape[0]
    device = target.get_device()    
    positive_pairs = target.unsqueeze(1).repeat(1,n) == target.repeat(n,1)    
    
    # ### Interclass margin only
    if margin_type  == 'inter':
        # Inter class angle margin
        cosm =  math.cos(margin_value) * torch.ones(n,n).to(device)        
        sinm = math.sin(margin_value) * torch.ones(n,n).to(device)                
        cosm[positive_pairs] = 1
        sinm[positive_pairs] = 0
        sims = (torch.mul(sims, cosm) -  torch.mul(sine, sinm)).fill_diagonal_(1.)  
    elif margin_type == 'intra':
        # Intra class angle margin
        cosm =  math.cos(margin_value) * torch.ones(n,n).to(device)        
        sinm = -math.sin(margin_value) * torch.ones(n,n).to(device)                
        cosm[~positive_pairs] = 1
        sinm[~positive_pairs] = 0
        sims = (torch.mul(sims, cosm) -  torch.mul(sine, sinm)).fill_diagonal_(1.)  
    elif margin_type == 'interintra':
        # Inter and intra class angle margin
        cosm =  math.cos(margin_value) * torch.ones(n,n).to(device)                
        cosm[positive_pairs] = math.cos(margin_value) # Inter-class margin should be smaller
        sinm = math.sin(margin_value) * torch.ones(n,n).to(device)                
        sinm[positive_pairs] = -math.sin(margin_value) # Inter-class margin should be smaller
        sims = (torch.mul(sims, cosm) -  torch.mul(sine, sinm)).fill_diagonal_(1.)  
    elif margin_type  == 'inter-scaled':
        # Inter class scaling margin
        sims[positive_pairs] = torch.minimum( (1 + margin_value) * sims[positive_pairs] , torch.tensor([1.],dtype  = torch.float32, device = device))        
    elif margin_type == 'intra-scaled':
        # Intra class scaling margin
        sims[~positive_pairs] = (1 - margin_value) * sims[~positive_pairs]
    elif margin_type == 'interintra-scaled':
        # Inter and intra class scaled margin
        sims[positive_pairs] = torch.minimum( (1 + margin_value) * sims[positive_pairs] , torch.tensor([1.],dtype  = torch.float32, device = device))
        sims[~positive_pairs] = (1 - margin_value) * sims[~positive_pairs]
    elif margin_type  == 'inter-sum':
        # Inter class scaling margin
        sims[positive_pairs] = torch.minimum( margin_value +  sims[positive_pairs] , torch.tensor([1.],dtype  = torch.float32, device = device))        
    elif margin_type == 'intra-sum':
        # Intra class scaling margin
        sims[~positive_pairs] = torch.maximum( -margin_value +  sims[~positive_pairs] , torch.tensor([0.],dtype  = torch.float32, device = device))        
    elif margin_type == 'interintra-sum':
        # Inter and intra class scaled margin
        sims[positive_pairs] = torch.minimum( margin_value +  sims[positive_pairs] , torch.tensor([1.],dtype  = torch.float32, device = device))        
        sims[~positive_pairs] = torch.maximum( -margin_value +  sims[~positive_pairs] , torch.tensor([0.],dtype  = torch.float32, device = device))        
    else:
        raise ValueError('Margin type unavailable')
 

    return sims

def get_weighted_similarity(model, data, teacher_temp):    
    
    embs, logits = model.embs_and_logits(data)    
    if teacher_temp is not None:
        logits = logits / teacher_temp
    probits = torch.max(F.softmax(logits,dim = 1),dim=1).values        
    confidence = probits * probits[:,None]        
    embs = F.normalize(embs, p=2, dim=1)
    sims = torch.matmul(embs, embs.transpose(0,1))        

    return sims, confidence

class SimilarityDistiller():
    def __init__(self, augment, margin,margin_value, margin_type, sup_term, c):
        self.augment = augment
        self.margin = margin
        self.sup_term = sup_term
        self.c = c
        self.margin_type = margin_type
        self.margin_value = margin_value
        self.transform = SimCLRTransform()
        self.loss_function = nn.MSELoss()

        if self.sup_term:
            self.ce = nn.CrossEntropyLoss()

    def compute_loss(self, student, teacher, data, target):
        if self.augment:
            data = self.transform(data, num_views=2)
            data = torch.cat(data, dim=0)
        if self.margin:            
            teacher_sims = get_margin_similarity(teacher, data,target, self.margin_value,self.margin_type)
        else:
            teacher_sims = get_similarity(teacher, data)
        student_sims = get_similarity(student, data)  


        
        loss = self.loss_function(student_sims, teacher_sims) 
        
        if self.sup_term:
            embs, logits = student.embs_and_logits(data)
            loss = loss + self.c * self.ce(logits,target)

        return loss

class KD():
    def __init__(self, c):
        if c < 0 or c > 1:
            raise ValueError('c must be in [0,1].')
        self.c = c
        self.sup_term = nn.CrossEntropyLoss()
        self.kd_term = nn.KLDivLoss(reduction='batchmean')

    def compute_loss(self, student, teacher, data, target):
        output = student(data)
        model_probs = nn.LogSoftmax(dim=1)(output)
        teacher_probs = nn.Softmax(dim=1)(teacher(data))            
        loss = self.c * self.kd_term(model_probs, teacher_probs) + \
            (1-self.c) * self.sup_term(output, target)
        
        return loss

class WeightedDistiller():
    def __init__(self, teacher_temp = None):
        self.teacher_temp = teacher_temp

    def compute_loss(self, student, teacher, data, target):
        student_sims = get_similarity(student, data)
        teacher_sims, teacher_confidence = get_weighted_similarity(teacher, data, self.teacher_temp)
        
        loss = torch.sum(teacher_confidence * (student_sims - teacher_sims)**2) / student_sims.shape[0]**2 
        
        return loss
