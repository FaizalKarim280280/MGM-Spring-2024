import torch

def get_num_model_params(model):
    counter = 0
    for params in model.parameters():
        counter += torch.prod(torch.tensor(params.shape))
    return f"{counter:,}"