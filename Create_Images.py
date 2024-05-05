### The following script uses model checkpoints in the Models folder to generate images. Models must have been created first using Visual_GAN_Train.py.

# Base Imports
import os
import numpy as np
import matplotlib.pyplot as plt

# PyTorch Imports
from torchvision.utils import make_grid
import torch

# Required Scripts/Notebooks
from Visual_GANs import *

# Parameters
model_folder_path = './Models/'
output_folder_path = './Images/'
# Input noise dimensions
z_dim = 64
# Images to save per model
images_to_save = 16

# Check available GPU
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    
def load_model(state_dict_path, z_dim, device=device):
    """
    Function to load generator via state dictionary.
    """
    # allowed_model_names = ['DCGAN', 'SNGAN', 'WGAN']
    # if model_name not in allowed_model_names:
    #     raise ValueError(f"Invalid input: '{model_name}'. It should be one of {allowed_model_names}")
        
    # Initialise initial model    
    gen = Base_Generator(z_dim=z_dim).to(device)    
    
    # Load the state dictionary
    state_dict = torch.load(state_dict_path)

    # Load the state dictionary into the model
    gen.load_state_dict(state_dict)
    
    return gen

model_fnames = list()

# Check for models subdirectory and list generator models
if os.path.exists(model_folder_path) and os.path.isdir(model_folder_path):
    print('Model Directory Present.')
    
    # Loop through and find models
    for file in os.listdir(model_folder_path):
        if 'generator' in file:
            model_fnames.append(file)
    
else:
    print('No Model Directory Found.')

# check for output images subdirectory
if os.path.exists(output_folder_path) and os.path.isdir(output_folder_path):
    print('Output Directory Present.')
# If not present create new directory
else:
    print(f'No Output Directory. Creating at {output_folder_path}.')
    os.mkdir(output_folder_path)
    
# Write images file for each model
for model_fname in model_fnames:
    print(model_fname)
    # Check model type
    model_type = model_fname.split('_')[1]
    game = model_fname.split('_')[2]
    state_dict_path = model_folder_path + model_fname
    
    # Name for file and plot title
    cleaned_name = ' '.join([game, model_type, 'Generator'])
    
    # Load model
    gen = load_model(state_dict_path=state_dict_path, z_dim=z_dim, device=device)
    
    # Create input noise for generator
    fake_noise = get_noise(16, z_dim, device=device)
    
    # Create 16 images (for 4 * 4 output plot)
    fake_images = gen(fake_noise)
    
    # Create plot
    
    # Convert from -1 -> +1 to 0 -> +1
    image_tensor = (fake_images + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:images_to_save], nrow=4)
    # Create plot
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.title(cleaned_name)
    
    # Save plot
    plt.savefig(f'{output_folder_path}{cleaned_name}.png', bbox_inches='tight', pad_inches = 0)

    # Close plot
    plt.close()
    
# Per game plots

games = ['BattleZone', 'Breakout', 'Pong', 'SpaceInvaders']

# Create consistent input noise for generator
input_noise = get_noise(4, z_dim, device=device)

for game in games:
    print(f'Generating images for {game}')
    game_model_fnames = [fname for fname in model_fnames if game in fname]
    
    gen_images = dict()

    # Write images file for each model
    for model_fname in game_model_fnames:
        # Check model type
        model_type = model_fname.split('_')[1]
        game = model_fname.split('_')[2]
        state_dict_path = model_folder_path + model_fname

        # Name for file and plot title
        cleaned_name = ' '.join([game, model_type, 'Generator'])

        # Load model
        gen = load_model(state_dict_path=state_dict_path, z_dim=z_dim, device=device) 

        # Create 16 images (for 4 * 4 output plot)
        fake_images = gen(input_noise)
        
        gen_images[model_type] = fake_images

    # Create plot
    
    # Extract images only
    fake_images = [v for k,v in gen_images.items()]
    
    # Combine images
    fake_images = torch.concatenate(fake_images, axis=0)

    # Convert from -1 -> +1 to 0 -> +1
    image_tensor = (fake_images + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat, nrow=4)
    # Create plot
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    # plt.title(cleaned_name)

    # Save plot
    plt.savefig(f'{output_folder_path}{game}_all_generators.png', bbox_inches='tight', pad_inches = 0)

    # Close plot
    plt.close()