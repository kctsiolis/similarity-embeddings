from run_base import get_base_args, run_base
from models.models import get_model
from training.training import DistillationTrainer
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
    parser.add_argument('--distillation-loss', type=str, choices=['similarity-based', 'similarity-weighted', 'kd'],
                        default='similarity-based',
                        help='Loss used for distillation.')
    parser.add_argument('--augmented-distillation', action='store_true',
                        help='Whether or not to use data augmentation in distillation.')
    parser.add_argument('-c', type=float, default=0.5,
                        help='Weighing of soft target and hard target loss in Hinton\'s KD.')
    parser.add_argument('--wrap-in-projection', action='store_true',
                        help='Wrap the teacher model in a random projection (For distillation only)')
    parser.add_argument('--projection-dim', type=int, default=None,
                        help='Dimension to of projection to wrap the teacher model in')
    parser.add_argument('--margin', action='store_true',
                        help='(For cosine distillation only) Should angular margin be applied ')
    parser.add_argument('--margin-value', type = float,default = 0.5,
                        help='If [margin] is selected what should it be set to (Default 0.5)')     
    parser.add_argument('--truncate-model', action = 'store_true',
                        help='Should we truncate the (student) model when training a linear classifier?')                        
    parser.add_argument('--truncation-level', type =int,
                        help='How many layers to remove to form the truncated (student) model')     

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
    logger = Logger('distillation', args)

    student = get_model(args.student_model, load_path=args.student_path, project=args.project_embedder, num_classes = num_classes)
    if args.distillation_loss != 'kd':
        student.student_mode()
    
    teacher = get_model(args.teacher_model, load_path=args.teacher_path, project=args.project_embedder, num_classes = num_classes)
    classify = args.distillation_loss == 'kd'
    teacher.teacher_mode(classify)

    student.to(device)
    teacher.to(device)

    trainer = DistillationTrainer(student, teacher, train_loader, val_loader, device, logger, args)
    trainer.train()

if __name__ == '__main__':
    main()