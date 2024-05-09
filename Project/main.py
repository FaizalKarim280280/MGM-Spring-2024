from scripts import train_mnist
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    args.exp_name = "D2"
    args.run_index = 0
    
    train_mnist.run(args)
    
    # for i in range(1000):
    #     args.run_index = i
    #     train_mnist.run(args)

if __name__ == "__main__":
    main()