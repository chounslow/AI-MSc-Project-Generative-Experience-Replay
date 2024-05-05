# Base Imports
import random
import os
from collections import defaultdict, deque
import datetime as dt
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import cv2

# PyTorch Imports
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Environment Imports
import gym
import ale_py

# Relative Imports
from DQN import *
import Environment as environ
from Training_Functions import *

# Parameters
buffer_pretraining = 100_000 # 100_000
training_frames = 2_000_000 # 2_000_000
env_name="Pong-ram-v4"
storage_root = './' #'D:/AI MSc/Dissertation/Offline_Data/'

# Set Seed
seed = environ.set_seed(123)


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')

base_env = environ.make_env(env_name, seed=seed, ram=True)
eval_env = environ.make_eval_env(env_name, seed=seed, ram=True, render=False)

### Agent

action_size = base_env.action_space.n

ohe_matrix = np.eye(action_size)

# We require shape of (frames, image_size, image_size)
state_size = base_env.observation_space.shape #[:-1]

agent = DQN_Agent_Combined(
    ram=True,
    state_size=state_size[0],    
    action_size=action_size,
    layer_size=128,
    n_step=1,
    batch_size=32, 
    buffer_size=1_000, 
    learning_rate=0.0002, #025, #1e-4 #0.0002
    tau=1e-3, # 0.005
    gamma=0.95, 
    update_freq=4, # 4
    device=device, 
    seed=seed,
    gener=False
)


### Set Agent & Filenames
agent_name = f'{env_name}-DQN-MemoryBuffer'

# Fill Memory Buffer
print('Filling Memory Buffer')
run_random_policy(random_frames=buffer_pretraining, env=base_env, agent=agent)

writer = SummaryWriter(f"{agent_name}_runs/")

scores, eval_scores = run_and_save(agent=agent, 
                                   env=base_env, 
                                   eval_env=eval_env, 
                                   env_name=env_name, 
                                   agent_name=agent_name,
                                   frames=training_frames, 
                                   run_eval=False, 
                                   storage_root=storage_root, 
                                   writer=writer)