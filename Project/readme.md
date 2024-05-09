<!-- ## MGM Project -->

# NN Parameter Generation using Diffusion Models

## Introduction
This project is on generating the parameters of a CNN model using Diffusion Models. Neural Network Diffusion by Wang, Kai et al. only focuses on small networks and generates only a subset of the total parameters. H Neural Network Diffusion directly encodes the whole weight matrix and attempts to generate the parameters in one go. This approach is indeed not scalable. 

Hence, the aim of this project was to generate parameters for a mid-sized CNN network using Diffusion. We took a sequential approach in generating the parameters, that is, generating parameters layer by layer.

## Important
`run_notebook.ipynb` contains inference code. Training code is not given because sharing the dataset is difficult as of now (still finding some place online to store it). 

## Install libraries

``conda env create -f environment.yml``

## Directory structure
Folders:
1. **dataloader**: Contains all the dataloaders
2. **images**: Contains some output images for results
3. **models**: Contains all the models: autoencoder, base_cnn and diffusion
4. **notebooks**: Contains notebooks for experiments and visualization
5. **scripts**: Contains python scripts for training and inference
6. **trainers**: Contains trainer classes for different models
7. **utils**: Contains some helper functions

## How to run
1. Training the Autoencoder: `python3 main_ae.py`
2. Training the diffusion model: `python3 main_diffusion.py`
3. Dataset creation: `python3 main.py` (Need to change the exp folder name)

## Some visualizations
<img src="https://github.com/FaizalKarim280280/MGM-Spring-2024/blob/main/Project/images/mgm-viz1.png"/>
<img src="https://github.com/FaizalKarim280280/MGM-Spring-2024/blob/main/Project/images/mgm-viz2.png"/>

