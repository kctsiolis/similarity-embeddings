from run_base import get_base_args, run_base
from models.models import get_model, get_model_embedding_dim
from models.general import CLIPDistill
from training.training import CLIPDistillTrainer
from training.logger import Logger
import argparse


def get_student_args(parser):
    parser.add_argument('--teacher-path', type=str,
                        help='Path to the teacher model.')
    parser.add_argument('--student-path', type=str,
                        help='Path to the student model (if loading from checkpoint).')
    parser.add_argument('--teacher-model', type=str, default=None,
                        help='Choice of teacher model.')
    parser.add_argument('--student-model', type=str,
                        help='Choice of student model.')        
    parser.add_argument('--project-embedder', action='store_true',
                        help='Add a projection head to the embedder.')
    parser.add_argument('--augmented-distillation', action='store_true',
                        help='Whether or not to use data augmentation in distillation.')   
    parser.add_argument('--margin', action='store_true',
                        help='(For cosine distillation only) Should angular margin be applied ')                             
    return parser

def get_args(parser):
    parser = get_base_args(parser)
    parser = get_student_args(parser)

    args = parser.parse_args()
    return args

def main():
    """Load arguments, the dataset, and initiate the training loop."""
    parser = argparse.ArgumentParser(description='Training the student model.')
    args = get_args(parser)

    train_loader, val_loader, num_classes, device = run_base(args)
    logger = Logger('clip_distillation', args)

    student = get_model(args.student_model, load_path=args.student_path, project=args.project_embedder, num_classes= num_classes)    
    teacher = get_model(args.teacher_model, load_path=args.teacher_path, project=args.project_embedder, num_classes= num_classes)    
    student.student_mode()
    teacher.teacher_mode(classify = False)

    teacher_dim = get_model_embedding_dim(args.teacher_model)
    student_dim = get_model_embedding_dim(args.student_model)

    model = CLIPDistill(teacher_dim,teacher,student_dim,student)
    model.to(device)

    trainer = CLIPDistillTrainer(model, train_loader, val_loader, device, logger, args)
    trainer.train()

if __name__ == '__main__':
    main()