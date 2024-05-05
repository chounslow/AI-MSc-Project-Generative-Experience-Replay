# This script is used to create GANs for full RL experiences.

# Base Imports
import random
import os
from collections import defaultdict, deque
import datetime as dt
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Environment imports
import gym
from gymnasium.wrappers import RescaleAction, TimeLimit, RecordEpisodeStatistics, RecordVideo, NormalizeObservation, TransformReward
from gymnasium.utils.seeding import np_random
import ale_py

# PyTorch Imports
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Relative Imports
from DQN import *
import Environment as environ
from Training_Functions import *

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')

# Parameters
IMAGE_SIZE = 64
buffer_pretraining = 100_000 # 100_000
training_frames = 2_000_000 # 2_000_000

# Set Seed
seed = environ.set_seed(123)

envs = [
    # 'SpaceInvaders-v4', 
    # 'Breakout-v4', 
    'Pong-v4', 
    # 'Seaquest-v4', 
    # 'BattleZone-v4'
]

for env_name in envs:
    print(env_name)
    for agent_gan_type in [
        # 'WGAN', 
        # 'DCGAN', 
        'None'
    ]: # 'DCGAN', 
        print(agent_gan_type)
        if agent_gan_type == 'None':
            agent_gener = False
            agent_batch_size = 16 #Try 64 next
        else:
            agent_gener = True
            agent_batch_size = 16*4
            
        agent_gan_mix = 0.75 #SynthER Paper uses 0.5 for this parameter try this next
        agent_combined_buffer = True

        if agent_gener:
            if agent_combined_buffer:
                buffer_name = 'Combined'
                gan_mix_str = '-' + str(agent_gan_mix) + '-'
            else:
                buffer_name = 'Gen'
                gan_mix_str = ''
            buffer_type = '-' + agent_gan_type + '-'
        else:
            buffer_name = 'Memory'
            buffer_type = ''
            gan_mix_str = ''

        agent_name =  f'{env_name}-BatchSize-{agent_batch_size}-{buffer_name}{gan_mix_str}{buffer_type}Buffer'
        
        print(agent_name)


### SET ENVIRONMENT


        # Set Seed
        seed = environ.set_seed(123)

        # Create Environment
        # env_name="SpaceInvaders-ram-v4"

        base_env = environ.make_env(env_name, seed=seed, ram=False)
        eval_env = environ.make_eval_env(env_name, seed=seed, ram=False, render=False)

        ### CREATE AGENT
        action_size = base_env.action_space.n

        ohe_matrix = np.eye(action_size)

        # We require shape of (frames, image_size, image_size)
        state_size = base_env.observation_space.shape #[:-1]

        agent = DQN_Agent_Combined(state_size=state_size,    
                          action_size=action_size,
                          layer_size=128,
                          n_step=1,
                          batch_size=agent_batch_size, 
                          buffer_size=500_000, 
                          learning_rate=0.0002, #025, #1e-4 #0.0002
                          tau=1e-3, # 0.005
                          gamma=0.95, 
                          update_freq=4, 
                          device=device, 
                          seed=seed,
                          gener=agent_gener,
                          gan_type=agent_gan_type,
                          gan_smoothing_factor=0.1,
                          gan_upsample=False,
                          gan_spectral=False,
                          gan_batch_norm=True,
                          gan_learning_rate=0.00002,
                          gan_disc_learning_rate_division=2.0,
                          gan_mix=agent_gan_mix,
                          gan_lambda_gp=20,
                          combined_buffer=agent_combined_buffer
                             )
        
### Fill Buffer
        run_random_policy(random_frames=buffer_pretraining, env=base_env, agent=agent)
    
### Run Simulation    
    
        writer = SummaryWriter(f"{agent_name}_runs/")
        
        scores, eval_scores = run(agent=agent, env=base_env, eval_env=eval_env, writer=writer, frames=training_frames, run_eval=False)
        
        # Save Agent
        torch.save(agent.qnetwork_local.state_dict(), f'{agent_name}-qnetwork_local-Visual')
        torch.save(agent.qnetwork_target.state_dict(), f'{agent_name}-qnetwork_target-Visual')

        print('Save Complete')
        
        reward_per_episode = scores

        reward_df = pd.DataFrame(scores, columns=['IDX', 'Step', 'Episode_Reward'])
        reward_df['100_Episode_AVG'] = reward_df['Episode_Reward'].rolling(100).mean()
        reward_df['500_Episode_AVG'] = reward_df['Episode_Reward'].rolling(500).mean()
        reward_df['1000_Episode_AVG'] = reward_df['Episode_Reward'].rolling(1000).mean()

        reward_df.to_csv(f'{agent_name}-Visual-results.csv')