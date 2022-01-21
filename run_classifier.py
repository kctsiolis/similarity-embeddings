from run_base import get_base_args, run_base
from models import get_model
import argparse
from logger import Logger
from training import SupervisedTrainer

def get_teacher_args(parser):
    parser.add_argument('--model-path', type=str,
                        help='Path to the model.')
    parser.add_argument('--model', type=str, default='resnet50',
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
    logger = Logger('linear_classifier', args)
    model = get_model(args.model, args.model_path, num_classes=num_classes)
    model.probing_mode()
    model.to(device)
    trainer = SupervisedTrainer(model, train_loader, val_loader, device, logger, args)
    trainer.train()

if __name__ == '__main__':
    main()