import argparse
from training.tuner import run_tuner
from training.trainer import run_trainer
from training.evaluator import run_evaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the GNN Pipeline")
    parser.add_argument('--mode', choices=['tune', 'train', 'eval'], required=True,
                        help="Select the operation mode: tune, train, or eval")
    parser.add_argument('--epoch', type=int, default=2000,
                        help="Number of training epochs (for tuner/trainer)")
    parser.add_argument('--f1', type=str, required=True,
                        help="Path to the features CSV file")
    parser.add_argument('--f2', type=str, required=True,
                        help="Path to the labels CSV file")
    args = parser.parse_args()
    
    # Pass these parameters into the corresponding functions as needed.
    if args.mode == 'tune':
        run_tuner(epoch=args.epoch, features_file=args.f1, labels_file=args.f2)
    elif args.mode == 'train':
        run_trainer(epoch=args.epoch, features_file=args.f1, labels_file=args.f2)
    elif args.mode == 'eval':
        run_evaluator(features_file=args.f1, labels_file=args.f2)
