from scripts import train_autoencoder
import argparse
import os


def main():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.device = 'cuda'
    args.batch_size = 128
    args.checkpoint_path = '/scratch/fk/ae-checkpoints/'
    args.epochs = 50
    
    os.makedirs(args.checkpoint_path, exist_ok=True)    
    train_autoencoder.run(args)
    
    
if __name__ == "__main__":
    main()