# This script is used to measure GANs for visual states only and not full RL experiences.

import torch
from torch import nn
from tqdm.auto import tqdm
import torchvision.models as models
from torchsummary import summary
import pickle
import numpy as np
import sklearn.metrics
import pandas as pd

import os
# from tqdm import tqdm

# Import functionality from notebook
from Data_Loading_Functions import *
from Performance_Metric_Functions import *

### Run this script after creating runs for visual agents (Using Visual_GAN_Train.py) and storing results in Visual_Results
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')

game = 'BreakOut'

# Root folder where Atari Replay Data is stored
root = 'D:/AI MSc/Dissertation/Data'

class gan_performance_measure:
    def __init__(self, device, output_features=10, real_features=100_000, batch_size=128):
        self.device = device
        self.game = None
        self.output_features = output_features
        self.real_feature_count = real_features
        self.batch_size = batch_size
        
        # Load VGG Model
        # Load base model without weights
        self.vgg_model = models.vgg16(weights=None)

        # Reduce input channels to 1
        self.vgg_model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        
        # Remove last 4 layers
        self.vgg_model.classifier = nn.Sequential(*list(self.vgg_model.classifier.children())[:-3])
        # Reduce output of last layer to output_features (paper uses 64).
        self.vgg_model.classifier[-1] = nn.Linear(in_features=4096, out_features=self.output_features, bias=True)

        # Check if file exists
        if os.path.isfile(f'vgg_random_model_{self.output_features}'):
            print('Model detected, loading weights.')
            self.vgg_model.load_state_dict(torch.load(f'vgg_random_model_{self.output_features}'))

        # Otherwise adjust base model
        else:
            print('No model detected, saving random weights.')

            # Save Model
            torch.save(self.vgg_model.state_dict(), f'vgg_random_model_{self.output_features}')

        # Move model to device
        self.vgg_model.to(self.device)

        # Print Model
        # print()
        # print(vgg_model)

        # Show summary with input/output dimensions
        # summary(vgg_model.to(device), (1, 64, 64))
        
        
    def set_game(self, game_name):
        self.game = game_name
        
    def real_features_file_check(self):
        if os.path.isfile(f'{self.game}_vgg_{self.output_features}_output.npy'):
            return True
        else:
            return False
        
    def get_real_features(self, fname=None):
    # check if real features file exists, if not then create
    
        if not self.game:
            raise Exception("No game has been set, use set_game function first.")

        if self.real_features_file_check() == True:
            # Load File
            print('File already exists, loading.')
            self.real_features = np.load(f'{self.game}_vgg_{self.output_features}_output.npy')

        # Create file
        else:
            if fname:
                real_observations = np.load(fname)
                print(f'Importing files for {self.game}.')
            else:
                # Import Data
                real_observations = get_atari_data(game=self.game, root=root, length=self.real_feature_count)['observation']

                print(f'Completed loading files for {self.game}.')

                print(f'Data Shape: {real_observations.shape}')

            # Transform Data

            dataloader = transform_data(real_observations, 
                                        batch_size=self.batch_size, 
                                        output_shape=64, 
                                        verbose=True, 
                                        device=self.device,
                                        channels=1
                                       )

            # Run VGG Model

            self.vgg_model.eval()
            results = list()
            batches = len(dataloader)

            print(f'# of batches: {batches}')
            with torch.no_grad():
                for idx, batch in enumerate(dataloader):
                    print(f'Batch No: {idx}', end='\r')
                    batch_results = self.vgg_model(batch)

                    results.append(batch_results.cpu().numpy())

            # Store as np array of x many 64-dimensional outputs        
            real_features = np.vstack(results)

            print(f'Shape of output array: {real_features.shape}')
            np.save(f'{self.game}_vgg_{self.output_features}_output.npy', real_features)
            
            real_features = np.load(f'{self.game}_vgg_{self.output_features}_output.npy')
            
            self.real_features = real_features
            
    def sample_real_features(self, sample_size=49_984):
        # Get full data size
        data_size = self.real_features.shape[0]
        # Generative indices for sample
        sample_idx = np.random.choice(range(data_size), size=sample_size, replace=False)
        # Take sample and convert to Tensor
        self.real_features = np.float32(self.real_features[sample_idx,:])
        
        
    def get_fake_features(self, fake_filename):

        # Load Fakes
        fakes = np.load(fake_filename)

        # Convert to Tensor
        torch_df = torch.from_numpy(np.float32(fakes)).to(self.device)
        # torch_df = torch_df.expand(torch_df.shape[0], 3, torch_df.shape[2], torch_df.shape[3])

        # Create Dataloader
        dataloader = DataLoader(
            torch_df,
            batch_size=self.batch_size,
            shuffle=True)

        # Run through VGG
        self.vgg_model.eval()
        results = list()
        batches = len(dataloader)

        # print(f'# of batches: {batches}')

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                # print(f'Batch No: {idx} / {batches}', end='\r')
                batch_results = self.vgg_model(batch)

                results.append(batch_results.cpu().numpy())

        # Store as np array of x many 64-dimensional outputs        
        fake_features = np.vstack(results)

        # print(f'Shape of output array: {fake_features.shape}')

        np.save(f'{self.game}_fakes_vgg_{self.output_features}_output.npy', fake_features)
        
        fake_features = np.load(f'{self.game}_fakes_vgg_{self.output_features}_output.npy')
        
        self.fake_features = fake_features
        
    def get_results(self, fake_sample_size=10_000):
    # Generate and return results

        # Get full data size
        data_size = self.fake_features.shape[0]
        # Generative indices for sample
        sample_idx = np.random.choice(range(data_size), size=fake_sample_size, replace=False)
        # Take sample and convert to Tensor
        fake = torch.from_numpy(np.float32(self.fake_features[sample_idx,:]))

        # Load real
        real = torch.from_numpy(self.real_features)

        prdc = compute_prdc(real, fake, verbose=False)
        prdc_rounded = {k:v.round(3) for k,v in prdc.items()}
        print(f'\tPRDC: {prdc_rounded}')
        return prdc

results_root = '../Stage 1 - Generate Visual States/'
games = os.listdir(f'{results_root}Visual_Results')

# models = ['DCGAN', 'WGAN', 'SNGAN']

results_list = list()

for game in games:
    print(game)
    folders = os.listdir(f'{results_root}Visual_Results/{game}')
    model_names_list = [folder for folder in folders if folder != 'Examples']
    for model in model_names_list:
        print(f'\t{model}')
        runs = os.listdir(f'{results_root}Visual_Results/{game}/{model}')
        for run in runs:
            results = dict()
            # print(f'\t\t{run}')
            files = os.listdir(f'{results_root}Visual_Results/{game}/{model}/{run}')
            if 'performance_results.csv' in files:
                next
            elif any(fname.endswith(".npy") for fname in files):
                print(f'\t\t{run}')
                # Store details
                results['game'] = game
                results['model'] = model
                results['Run_Type'] = run.split('_')[0]
                results['Run_Idx'] = run.split('_')[-1]
                results['filepath'] = f'{results_root}Visual_Results/{game}/{model}/{run}'
                
                # Save Entry
                results_list.append(results)
                
                
# Set up object
performance_model = gan_performance_measure(device=device)

prev_game = ''

for run in results_list:
    
    # Store details of run
    game = run['game']
    model = run['model']
    run_type = run['Run_Type']
    run_idx = run['Run_Idx']
    filepath = run['filepath']
    
    print(f'Generating results for: {game}, {model}, {run_type}, {run_idx}')
    
    # Create dictionary for results for each step
    step_results = dict()
    
    # Find each step file
    fake_files = [x for x in os.listdir(filepath) if '.npy' in x]
    steps = [x.split('_')[-1] for x in fake_files]
    steps = [x.split('.')[0] for x in steps]
    steps = [int(x) for x in steps]
    
    # Sort Steps
    steps.sort()
    
    # Check if game features have already been loaded
    if game != prev_game:
        # Set game in performance model
        performance_model.set_game(game)

        # Get real features (real observations ran through the vgg model)
        performance_model.get_real_features()

        # Downsample real features
        performance_model.sample_real_features(sample_size=49_984)
    
    for step in steps:
        print(f'Running for steps: {step:_}')
        fname = f'benchmark_fakes_step_{step}.npy'
        
        # Store fake features
        performance_model.get_fake_features(filepath+'/'+fname)
        
        # Get Results
        results = performance_model.get_results(fake_sample_size=10_000)
        
        # Store Results
        step_results[step] = results
    # Save Results
    
    # Store results in pandas df
    run_results_df = pd.DataFrame(step_results)
    
    # Save results in the relevant folder
    run_results_df.to_csv(filepath+'/'+'performance_results.csv')
    
    # Set prev game for next run
    prev_game = game