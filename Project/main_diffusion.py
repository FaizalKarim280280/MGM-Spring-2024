from scripts import train_diffusion
import argparse

def main():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.device = 'cuda'
    args.batch_size = 128

    train_diffusion.run(args)
    
    
if __name__ == "__main__":
    main()
