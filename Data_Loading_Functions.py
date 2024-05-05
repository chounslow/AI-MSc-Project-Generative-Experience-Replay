# The following scripts contains a range of functions for loading and utilising existing datasets.

# Default Imports
import os
from os.path import isfile
import gzip
import numpy as np
from typing import List, Tuple
from tqdm.auto import tqdm

# Pytorch Imports
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def get_atari_data(game='SpaceInvaders', 
                   root='D:/AI MSc/Dissertation/Data', 
                   elements=['observation'], 
                   length=None,
                   sample_random=True,
                   single_file=True,
                   gap=4):
    """
    Function loads Google DQN Replay Data from a root folder. 
    Directory format expected is as follows:
        Root Folder (Data):
            game:
                1:
                    replay_logs:
                        $store$_action_ckpt.0.gz
                        $store$_observation_ckpt.0.gz
                        $store$_reward_ckpt.0.gz
                        $store$_terminal_ckpt.0.gz
                        add_count_ckpt.0.gz
                        invalid_range_ckpt.0.gz
                        ... for 0 to 50 versions of each of the above.
                2:
                    as above
                3:
                    as above
                4:
                    as above
                5:
                    as above
        From these, actions, observations, rewards and terminal flags are collected.
            
    Default settings loading every 4th observation only for the first document found.
    Args:
        game (string) : - name of game and hence subfolder name.
        root (string): name of root directory to search.
        length (int): number of samples to return, None defaults to all samples.
        sample_random (bool): Flag to determine whether samples are loaded in order or randomly.
        single_file (bool): Flag to determine whether a single file is loaded or all.
        gap (int): gap between successive samples returned. Default 4 means every 4th sample returned.
    Returns:
        data (Dict): dictionary containing numpy arrays (value) for each element (key).
    """
    
    # Formatting prefix based on above directory and filename structure
    STORE_FILENAME_PREFIX = '$store$_'
    
    # Initialise output file
    data = dict()
    
    # Path based on input parameters - only 1st run is taken
    path = f'{root}/{game}/1/replay_logs/'
    
    # Check path exists
    if not os.path.exists(path):
        raise ValueError(f'Invalid value for game: {game}. Directory does not exist.')
    
    # If extracting a single file, take from the end of training
    if single_file:    
        suffix = 40
    # Otherwise start at 0
    else:
        suffix = 0
    # Initially load a single file
    for elem in elements:
        # Load first file
        filename = f'{path}{STORE_FILENAME_PREFIX}{elem}_ckpt.{suffix}.gz'
        print(f'Loading {filename = }')
        
        with open(filename, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as gzip_file:
                data[elem] = np.load(gzip_file)
                
        # Loop through files        
        if not single_file:        
            for suffix in range(1, 100):
                # Update Filename
                filename = f'{path}{STORE_FILENAME_PREFIX}{elem}_ckpt.{suffix}.gz'
                print(f'Loading {filename = }')
                
                with gzip.GzipFile(fileobj=f) as gzip_file:
                    numpy_file = np.load(gzip_file)
                    # Append to existing file on 0th axis
                    data[elem] = np.concatenate((data[elem], numpy_file), axis=0)
                
    # If sampling files            
    if length:
        if sample_random:
            data_length = data[elements[0]].shape[0]
            random_indices = np.random.randint(0, data_length, size=length)
            # Sample all elements with the same indices
            for elem in elements:
                data[elem] = data[elem][random_indices]
        else:
            # Filter all elements to the same length
            for elem in elements:
                data[elem] = data[elem][:length*gap:gap]
    return data



def transform_data(data, 
                   batch_size=64, 
                   output_shape=64, 
                   verbose=True, 
                   device='cuda', 
                   channels=1
                  ):
    """
    Convert data in numpy array to correct format and return as PyTorch dataloader.
    
    Args:
        data (numpy_array): input data array.
        batch_size (int): batch_size for dataloader.
        output_shape (int): output_shape height and width for observations.
        verbose (bool): Flag to determine verbosity.
        device (string): Desired dataloader device.
        channels (int): output channels for observations, for multi-channel, original values are duplicated..
    Returns:
        dataloader (PyTorch Dataloader): data cleaned, transformed and returned in Dataloader object for batch loading.
                                         Output format is: Batch, Channel, Height, Width.
    
    """
    
    # The following is used to ensure the data size divides completely by batch size
    # This ensures all batches will be complete
    data_size = data.shape[0]
    
    if verbose:
        max_value_data = data.max()
    
    # Highest size that leads to complete batches
    full_batch_data_size = data_size - (data_size % batch_size)
    
    # full_batch_data = data[:full_batch_data_size]
    data = data[:full_batch_data_size]
    
    # Normalising input between -1 to 1
    data = data * 2.0 / 255.0 - 1.0
    
    if verbose:
        print(f'Max value before normalisation: {max_value_data}')

        print(f'Max value after normalisation: {round(data.max(), 2)}')
        
    # Convert to Tensor and move to device
    torch_df = torch.from_numpy(np.float32(data)).to(device)
    
    # Expand dimensions as Batch, Single-Channel, X, Y
    torch_df = torch_df[:, None, :, :]
    if channels == 3:
        torch_df = torch_df.expand(torch_df.shape[0], 3, torch_df.shape[2], torch_df.shape[3])

    if verbose: print(f'Dataset Shape: {torch_df.shape}')

    # Transform images to given dimensions
    transform = transforms.Compose([
        transforms.Resize(output_shape, antialias=True),
    ])

    transformed_df = transform(torch_df)

    if verbose: print(f'Shape after Resize: {transformed_df.shape}')

    dataloader = DataLoader(
        transformed_df,
        batch_size=batch_size,
        shuffle=True)
    
    return dataloader