from run_base import get_base_args, run_base
from models import get_model
import argparse
from logger import Logger
from training import SupervisedTrainer

def get_teacher_args(parser):
    parser.add_argument('--teacher-path', type=str,
                        help='Path to the teacher model (if restarting training from checkpoint).')
    parser.add_argument('--teacher-model', type=str, default='resnet50',
                        help='Choice of teacher model.')

    return parser

def get_args(parser):
    parser = get_base_args(parser)
    parser = get_teacher_args(parser)

    args = parser.parse_args()
    return args

def main():
    parser = argparse.ArgumentParser(description='Training the teacher model.')
    args = get_args(parser)

    train_loader, val_loader, num_classes, device = run_base(args)
    logger = Logger('teacher', args)
    teacher = get_model(args.teacher_model, args.teacher_path, num_classes=num_classes)
    teacher.to(device)
    trainer = SupervisedTrainer(teacher, train_loader, val_loader, device, logger, args)
    trainer.train()

if __name__ == '__main__':
    main()