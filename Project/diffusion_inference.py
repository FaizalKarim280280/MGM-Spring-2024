import torch
import torch.nn as nn
from tqdm import tqdm
from models.autoencoder import AutoEncoder
from models.diffusion import Diffusion, DiffusionNet
from icecream import ic
import argparse

def sample_parameters(args, diffusion, model, ae_models, n):
    layer_codes = []
    
    print("Starting reverse diffusion")
    
    with torch.no_grad():
        code_prev = None
        for l in range(4):
            x = torch.randn(n, 512).to(args.device)
            layer = (torch.ones(n) * (l + 1) * 30).long().to(args.device)
            for i in tqdm(reversed(range(1, diffusion.noise_steps)), position=0):
                if code_prev is None:
                    x = x + 0.8 * torch.zeros_like(x)
                else:
                    x = x + 0.8 * code_prev
                
                t = (torch.ones(n) * i).long().to(args.device)
                pred_noise = model(x, t, layer)
                alpha = diffusion.alpha[t][:, None].to(args.device)
                alpha_hat = diffusion.alpha[t][:, None].to(args.device)
                beta = diffusion.beta[t][:, None].to(args.device)
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise
            
            code_prev = x.clone()
            layer_codes.append(x.clone())
            
            print(f"Layer:{l + 1} done.")
            
    layer_codes = torch.stack(layer_codes, dim = 1)
    print("Latent codes generated for all the layers. Now passing it to the decoder.")
    
    generated_weights = []
    
    for layer in range(4):
        y_pred = ae_models[layer].decoder(layer_codes[:, layer, :])
        generated_weights.append(y_pred)
        print(f"layer: {layer + 1}: {y_pred.shape}")
        
    return generated_weights

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', action='store_true')
    
    args = parser.parse_args()
    args.device = 'cuda'
    args.diffusion_checkpoint = '/scratch/fk/diffusion_checkpoint/280_best_val.pth'
    args.ae_checkpoint = './checkpoint_178_0.00744.pth'
    
    # Initialize the autoenocders
    model_l1 = AutoEncoder(num_layers=32, in_c=1, out_c=1, device=args.device)
    model_l2 = AutoEncoder(num_layers=32, in_c=32, out_c=32, device=args.device)
    model_l3 = AutoEncoder(num_layers=64, in_c=32, out_c=32, device=args.device)
    model_l4 = AutoEncoder(num_layers=64, in_c=64, out_c=64, device=args.device)
    
    ae_models = [model_l1, model_l2, model_l3, model_l4]
    ae_weights = torch.load(args.ae_checkpoint)
    
    # load the weights
    for i, model in enumerate(ae_models):
        model.load_state_dict(ae_weights[i + 1])
        model.to(args.device)
        # put the model in eval mode and freeze the layers
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
    diffusion_model = DiffusionNet().to(args.device)
    diffusion = Diffusion(
        noise_steps=100,
        beta_start=1e-7,
        beta_end=1e-5,
        device=args.device
    )
    
    generated_weights = sample_parameters(args, diffusion, diffusion_model, ae_models, n=32)
    print("Weights saved in ./generated_weights")
    
    if args.save:
        torch.save(generated_weights, './generated_weights/1.pth')
    
    # print(generated_weights)
    
def main():
    run()
    
    
if __name__ == "__main__":
    main()
    
    
    
