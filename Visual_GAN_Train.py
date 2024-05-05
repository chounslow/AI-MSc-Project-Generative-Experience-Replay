# This script is used to create GANs for visual states only and not full RL experiences.

# Base Imports
import os
import gc
import matplotlib.pyplot as plt

# PyTorch Imports
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.autograd import Variable

# Import functionality from other notebooks
from Visual_GANs import *
from Data_Loading_Functions import *
from Helper_Functions import *


# Parameters for Game, GAN Type & Data setting
game = 'Pong' # 'SpaceInvaders', 'Breakout', 'Pong', 'BattleZone' or 'SeaQueast'
gan_type = "DCGAN" # "DCGAN", "WGAN" or "SNGAN"
test_type = 'Medium' # 'High', 'Medium' or 'Low'

# Output folder name
folder_name = 'Visual_Results'

# Create directory if it doesn't already exist
if not os.path.exists(f'./{folder_name}'):
    os.mkdir(f'./{folder_name}')

n_epochs = 10
z_dim = 64
display_step = 1000
batch_size = 64
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
# Loss weight for gradient penalty
lambda_gp = 10

# Flag to save example observations from training data
save_examples = False

# Check available GPU
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')


if test_type == 'High':
    data_size = 200_000
    n_epochs = 2
if test_type == 'Medium':
    data_size = 50_000 # 500_000
    n_epochs = 2 * 4
if test_type == 'Low':
    data_size = 10_000 # 500_000
    n_epochs = 2 * 4 * 4


### Get Data
    
game_data = get_atari_data(game=game, length=data_size, gap=1)['observation']

print(f'Completed loading files for {game}.')

print(f'Data Shape: {game_data.shape}')

### Format Data as Dataloader

dataloader = transform_data(game_data, 
                            batch_size=batch_size, 
                            output_shape=64, 
                            verbose=True, 
                            device=device,
                            channels=1
                           )

### Save Examples

# Save Examples
examples_length = 5

folder_name='Visual_Results'
trial_folder = f'{folder_name}/{game}/{gan_type}'
print(f'{trial_folder = }')

# Create top level directories
create_directories(gan_type, game, folder_name=folder_name)
if not os.path.exists(f'{folder_name}/{game}/Examples'):
    os.mkdir(f'{folder_name}/{game}/Examples')

if save_examples:
    # Loop through batches in dataloader
    for example_idx, examples in enumerate(dataloader):
        if example_idx >= examples_length:
            break
        else:
            directory = f'{folder_name}/{game}/Examples'
                
            # Save example images in subdirectory
            save_tensor_images(examples, directory, f'Example_images_{example_idx+1}')
            
### Create & Train GAN Loop

# Create Model directory if it doesn't exist
if not os.path.exists('./Models/'):
    os.mkdir('./Models/')

# Set how often to save images and model
display_step = 500

## Parameters
# A learning rate of 0.0002 works well on DCGAN
lr = 0.0002
# Momentum Parameters
beta_1 = 0.5 
beta_2 = 0.999

Tensor = torch.cuda.FloatTensor

# Select GAN type
# gan_type = "WGAN"

# Select number of trials
trials = 5

# Generate noise to measure performance of GAN at each stage
static_fake_noise = get_noise(10_000, z_dim, device=device).detach()

# Loop
for trial_idx in range(trials):
    print(f'Trail IDX: {str(trial_idx+1)}\n')
    # Create Trial Directory
    trial_path = f'{trial_folder}/{test_type}_Data_{trial_idx+1}'
    if not os.path.exists(trial_path):
        os.mkdir(trial_path)
        
    # Write parameters
    write_parameters(path=trial_path,
                    game=game,
                    gan_type=gan_type,
                    z_dim=z_dim,
                    batch_size=batch_size,
                    data_size=data_size,
                    n_epochs=n_epochs,
                    c_lambda=c_lambda,
                    crit_repeats=crit_repeats,
                    lr=lr,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    lambda_gp=lambda_gp)
    
    # Create Generator & Discriminator
    gen = Base_Generator(z_dim).to(device)
    if gan_type == 'WGAN':
        disc = Critic().to(device) 
    elif gan_type == 'SNGAN':
        disc = DiscriminatorSN().to(device) 
    else:
        disc = Discriminator().to(device) 

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

    # Initialise Weights
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

    # Loss Criterion
    criterion = nn.BCEWithLogitsLoss()
    
    # Create results dict
    results = dict()

    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for idx, real in enumerate(tqdm(dataloader)):
            cur_batch_size = len(real)
            real = real.to(device)

            ## Update discriminator ##
            disc_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            disc_fake_pred = disc(fake.detach())
            disc_real_pred = disc(real)

            if gan_type == "DCGAN" or gan_type == "SNGAN":
                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

                disc_loss = (disc_fake_loss + disc_real_loss) / 2

            elif gan_type == "WGAN":

                gradient_penalty = compute_gradient_penalty(disc, real, fake)

                disc_loss = -torch.mean(disc_real_pred) + torch.mean(disc_fake_pred) + lambda_gp * gradient_penalty        

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Update gradients
            disc_loss.backward(retain_graph=True)
            # Update optimizer
            disc_opt.step()

            if (idx % crit_repeats == 0) or (gan_type == "DCGAN") or (gan_type == "SNGAN"):
                ## Update generator ##
                gen_opt.zero_grad()
                fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
                fake_2 = gen(fake_noise_2)
                
                if gan_type == "DCGAN"or gan_type == "SNGAN":
                    disc_fake_pred = disc(fake_2)
                    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

                elif gan_type == "WGAN":
                    disc_fake_pred = disc(fake_2)
                    gen_loss = -torch.mean(disc_fake_pred)


                gen_loss.backward()
                gen_opt.step()

                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / display_step

            ## Visualization code ##
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Epoch {epoch}, step {cur_step:_}: Generator loss: {mean_generator_loss:.2f}, discriminator loss: {mean_discriminator_loss:.2f}")

                # Save Images
                save_tensor_images(fake, trial_path, f'fake_images_step_{cur_step}')

                # Save Benchmark Dataframe
                with torch.no_grad():
                    fake_benchmark = gen(static_fake_noise).detach().cpu().numpy()

                    # Save Results
                    np.save(f'{trial_path}/benchmark_fakes_step_{cur_step}.npy', 
                            fake_benchmark #.cpu().numpy()
                           )

                    del fake_benchmark
                    gc.collect()

                # Reset Values
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                
                # Overwrite models at each stage
                torch.save(gen.state_dict(), f'Models/generator_{gan_type}_{game}_model')
                torch.save(disc.state_dict(), f'Models/discriminator_{gan_type}_{game}_model')

            cur_step += 1