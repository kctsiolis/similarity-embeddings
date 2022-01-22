'''
Train many models: Either students for distillation or linear classifiers to attach on to distilled students
This file is meant to be used to investigate student performance as a function of teacher performance.
'''
import argparse
import subprocess
import os
import re
import time
from log_file_utils import rename_models


def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--base-path', type=str,metavar='B',help = 'Path to directory where models to train on are kept')
    parser.add_argument('--rename',action = 'store_true',help = 'Rename models — needed if models haven\'t been renamed before')
    parser.add_argument('--training-type',type = str,choices = ['distill','linear-classifier'],metavar='R',help = 'Rename models — needed if models haven\'t been renamed before')
    parser.add_argument('--base-model-type',type = str,help = 'What kind of models are stored in the base directory')
    parser.add_argument('--student-type',type = str)
    parser.add_argument('--epochs',type = int,default = 250)
    parser.add_argument('--dataset',type = str,choices=['mnist', 'cifar', 'cifar100', 'imagenet', 'tiny_imagenet'])
    parser.add_argument('--device', type=str, nargs='+', default=['cpu'], help='Name of CUDA device(s) being used - otherwise cpu')                               
 
    args = parser.parse_args()
    return args



def main():
    ''' Train many models — either train a student for each teacher model in a base directory
        or
        train a linear classifier for each student in a base directory
        This call is likely to be used twice in a row—first to train students—then to attach linear classifiers        
     '''
    parser = argparse.ArgumentParser(description='Training many models')
    args = get_args(parser)    

    if args.training_type == 'linear-classifier':        
        # For adding a linear classifier to many students
        
        student_path_list = os.listdir(args.base_path)
        student_path_list = [os.path.abspath(os.path.join(args.base_path,path)) for path in student_path_list]    
        
        if args.rename:                
            rename_models(root_path = args.base_path, path_list = student_path_list)
        
        student_path_list = os.listdir(args.base_path)
        epoch_numbers = [int(re.findall(r'\d+', txt)[0]) for txt in student_path_list]
        student_path_list = [os.path.abspath(os.path.join(args.base_path,path)) for path in student_path_list]    
        
        add_linear_classifier_to_many(student_type = args.base_model_type,path_list = student_path_list,epoch_numbers = epoch_numbers,dataset = args.dataset,devices = args.device)
    elif args.training_type == 'distill':   

        # Used for distillation of many students
        base_teacher_path = args.base_path

        # Get all model paths in base directory
        teacher_models = [name for name in os.listdir(base_teacher_path) if 'model_epoch' in name]
        epoch_numbers = [int(re.findall(r'\d+', txt)[0]) for txt in teacher_models]

        # Sort the model paths in order of epoch they were saved in
        sorter = list(zip(*sorted(zip(epoch_numbers,teacher_models))))
        epoch_numbers, teacher_models = list(sorter[0]),list(sorter[1])
        
        distill_many_students(base_teacher_path,args.base_model_type,args.student_type,epoch_numbers,teacher_models,sim_distillation=True,n_epochs = args.epochs,dataset = args.dataset,devices = args.device)

def add_linear_classifier_to_many(student_type : str,path_list : list,epoch_numbers :list, dataset:str,devices : list):
    '''
    
    Add a linear classifier to many similarity distilled students 
    base_teacher_path = path the directory containing teacher models
    epoch_numbers = expects the teachers to be saved every few epochs — these numbers are a list of the epoch the teacher was saved at
    teacher_models = a list of paths to the teacher models         
    
    '''
    n_device = len(devices)
    validate = '--validate' if ((dataset == 'imagnet') or (dataset == 'tiny_imagenet')) else ''
    i = -1
    for obj in zip(epoch_numbers,path_list):
        epoch,path = obj
        
        # Restarting the process — some error might cause models to not be trained past certain epoch
        if (epoch <= -1) or (epoch >= 1e10):
            continue 
        else:
            i += 1
                                        
        # Parallel GPU training by someone who doesn't do it often          
        print(f'Training a linear classifier on student from epoch {epoch}.')    
        
        load_path = os.path.join(path,'model.pt')
        epochs = 30
        time.sleep(15) # Give the past call a chance to make files and such
                            
        device = devices[i % n_device]                
        bash_string = f"python3 run_training.py --mode linear_classifier --dataset {dataset} --device {device} \
                --load-path {load_path} --student-model {student_type} \
                --lr 0.1 --optimizer sgd --batch-size 128 --epochs {epochs} --early-stop 25 {validate}"  
    
        if i % n_device == (n_device - 1):                              
            subprocess.run(bash_string,shell=True) # Note the 'run'. This causes it to wait            
        else:              
            subprocess.Popen(bash_string,shell=True)            
    
def distill_many_students(base_teacher_path : str,teacher_type : str,student_type : str,epoch_numbers :list ,teacher_models : list,sim_distillation : bool,n_epochs : int,dataset : str,devices : list):
    '''
    
    Train many students from trained teacher models
    base_teacher_path = path the directory containing teacher models
    epoch_numbers = expects the teachers to be saved every few epochs — these numbers are a list of the epoch the teacher was saved at
    teacher_models = a list of paths to the teacher models         
    
    '''
    
    
    distillation_type = 'similarity-based' if sim_distillation else 'class-probs'
    n_device = len(devices)
    validate = '--validate' if ((dataset == 'tiny_imagenet') or (dataset == 'imagenet')) else ''                        
    i = -1
    for obj in zip(epoch_numbers,teacher_models):
        epoch,teacher_path = obj
        
        # Restarting the process — some error caused models to not be trained past certain epoch
        if (epoch >= 34) and (epoch <= 73):
            continue 
        else:
            i += 1 
                                        
        # Parallel GPU training by someone who doesn't do it often
        load_path = os.path.abspath(os.path.join(base_teacher_path,teacher_path))    
        print(f'Training a student on teacher from epoch {epoch}.')    
        
        time.sleep(15) # Give the past call a chance to make files and such 
        device = devices[i % n_device]   
        bash_string = f"python3 run_training.py --mode distillation --lr 0.01 --optimizer adam --batch-size 128 --epochs {n_epochs} --early-stop 50 \
            --dataset {dataset}  --device {device} --teacher-model  {teacher_type} --load-path {load_path} --student-model {student_type} \
            --cosine --distillation-type {distillation_type} {validate}"
    
        if i % n_device == (n_device - 1):                              
            subprocess.run(bash_string,shell=True) # Note the 'run'. This causes it to wait            
        else:              
            subprocess.Popen(bash_string,shell=True)  
    


if __name__ == '__main__':
    main()