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
# How many training Frames
training_frames = 2_000_000 # 2_000_000
# How often to evaluate Run
eval_every = 100_000 # 100_000
# How Many Random States to Load
random_state_number = 100_000 # 100_000

# Loop Parameters for Random Train

environments = [
    # 'SpaceInvaders-ram-v4', 
    # 'Breakout-ram-v4', 
    'Pong-ram-v4', 
    # 'Seaquest-ram-v4', 
    # 'BattleZone-ram-v4'
]

data_setting_options = [
    # {'Name': 'Low Data', 'Samples':10_000},
    # {'Name': 'Medium Data', 'Samples':50_000},
    {'Name': 'High Data', 'Samples':100_000},
                       ]

agent_params = [
    {'gener':False, 'cql':True, 'combined_buffer':False, 'gan_type':'DCGAN', 'gan_mix':0.5, 'batch_size':32},
    # {'gener':False, 'cql':False, 'combined_buffer':False, 'gan_type':'DCGAN', 'gan_mix':0.5, 'batch_size':32},
    # {'gener':True, 'cql':False, 'combined_buffer':True, 'gan_type':'DCGAN', 'gan_mix':0.5, 'batch_size':64},
    # {'gener':True, 'cql':True, 'combined_buffer':True, 'gan_type':'DCGAN', 'gan_mix':0.5, 'batch_size':64},
    # {'gener':True, 'cql':False, 'combined_buffer':True, 'gan_type':'WGAN', 'gan_mix':0.5, 'batch_size':64},
    # {'gener':True, 'cql':True, 'combined_buffer':True, 'gan_type':'WGAN', 'gan_mix':0.50, 'batch_size':64},
]

# Set Seed
seed = environ.set_seed(123)


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')

# Offline Training Functions

def offline_run(eval_env, 
                agent, 
                np_states, 
                np_actions,
                np_rewards,
                np_next_states,
                np_dones, 
                frames=200_000, eval_every=1_000, eval_runs=5, run_eval=True):
    """Deep Q-Learning offline training using numpy experiences and no active environment exploration.
    
    Args:
        np_states (numpy array): training states.
        np_actions (numpy array): training actions.
        np_rewards (numpy array): training rewards.
        np_next_states (numpy array): training next states.
        np_dones (numpy array): training dones.
        frames (int): Number of environment frames to train for.
        eval_every (int): Number of frames to run evaluation.
        eval_runs (int): number of evaluation runs.
    """
    # List containing scores for each episode
    # List containing average scores for each evaluation step
    evaluation_scores = list()
    frame = 0
    np_length = np_rewards.shape[0]
    for frame in range(1, frames+1):
        dataset_idx = frame
        dataset_idx = dataset_idx % np_length
        
        # Get item
        state = np_states[dataset_idx,:] 
        action = np_actions[dataset_idx,:]
        reward = np_rewards[dataset_idx]
        next_state = np_next_states[dataset_idx,:]
        done = np_dones[dataset_idx]

        agent.step(state, action, reward, next_state, done) #, writer)

        # evaluation runs
        if (frame % eval_every == 0 or frame == 1) and run_eval:
            # print('Running Evaluation')
            eval_score, eval_score_std = evaluate(agent, eval_env, eval_runs)
            evaluation_scores.append([frame, eval_score, eval_score_std])
            
            print_str = f'Episode {frame} \tEval Score {eval_score:.2f}\t Eval Score STD {eval_score_std:.2f}'
            # Overwrite print line except every 10k steps
            end = '\n' if frame % 100_000 == 0 else '\r'
            print(print_str, end=end)

    return evaluation_scores


def evaluate(agent, eval_env, eval_runs=10):
    """
    Makes an evaluation runs with eps 0
    """
    reward_batch = []
    for i in range(eval_runs):
        state, _ = eval_env.reset()
        rewards = 0
        steps = 0
        actions = list()
        while True and steps < 100_000:
            action = agent.act(state[None,:], 0.0)[0]
            actions.append(action)
            next_state, reward, done, _, _ = eval_env.step(action)
            state = next_state
            rewards += reward
            steps += 1
            if done:
                break
        reward_batch.append(rewards)

    return np.mean(reward_batch), np.std(reward_batch)

### Create Random frames for training from random


for env_name in environments:
    get_new_states = False

    if os.path.isfile(f'{env_name}_states_ram.npy'):
        print('Files Detected.')
        get_new_states = False

    else:
        print('No Files Detected.')

    if get_new_states:
        print(f'Getting states.')
        # Get states from random actions in the environment
        np_states, np_actions, np_rewards, np_next_states, np_dones = get_states(random_state_number, base_env)

        np.save(f'{env_name}_states_ram.npy', np_states)
        np.save(f'{env_name}_actions_ram.npy', np_actions)
        np.save(f'{env_name}_rewards_ram.npy', np_rewards)
        np.save(f'{env_name}_next_states_ram.npy', np_next_states)
        np.save(f'{env_name}_dones_ram.npy', np_dones)

    else:
        print(f'Loading states.')
        np_states = np.load(f'{env_name}_states_ram.npy')
        np_actions = np.load(f'{env_name}_actions_ram.npy')
        np_rewards = np.load(f'{env_name}_rewards_ram.npy')
        np_next_states = np.load(f'{env_name}_next_states_ram.npy')
        np_dones = np.load(f'{env_name}_dones_ram.npy')


print('Completed Loading States.')


### Training Loop from Random Frames

for env_name in environments:
    print('Getting random data for: ', env_name)
    for row in data_setting_options:
        for params in agent_params:
            # Get Parameters
            agent_batch_size = params['batch_size']
            agent_gener = params['gener']
            agent_gan_type = params['gan_type']
            agent_gan_mix = params['gan_mix']
            agent_combined_buffer = params['combined_buffer']
            agent_cql = params['cql']
            
            # Data Parameters
            data_setting_name = row['Name'].replace(' ', '')
            data_size = row['Samples']

            # Run Name

            if agent_gener:
                if agent_combined_buffer:
                    buffer_name = 'Combined'
                    gan_mix_str = '-' + str(agent_gan_mix)
                else:
                    buffer_name = 'Gen'
                    gan_mix_str = ''
                buffer_type = '-' + agent_gan_type + '-'
            else:
                buffer_name = 'Memory'
                buffer_type = ''
                gan_mix_str = ''
            offline_rl = ''
            if agent_cql:
                offline_rl = '-CQL'

            run_name =  f'{env_name}{offline_rl}-{data_setting_name}-BatchSize-{agent_batch_size}-{buffer_name}{gan_mix_str}{buffer_type}Buffer'
            
            print(f'{run_name = }')

        ### Load Data

            print(f'Loading states for {data_setting_name}.')
            np_states = np.load(f'{env_name}_states_ram.npy')[:data_size,:].copy()
            np_actions = np.load(f'{env_name}_actions_ram.npy')[:data_size,:].copy()
            np_rewards = np.load(f'{env_name}_rewards_ram.npy')[:data_size].copy()
            np_next_states = np.load(f'{env_name}_next_states_ram.npy')[:data_size,:].copy()
            np_dones = np.load(f'{env_name}_dones_ram.npy')[:data_size].copy()
            # print('Completed Loading States.')
            avg_reward = np_rewards.sum() / np_dones.sum()
            print(f'{avg_reward = :.2}')

        ### SET ENVIRONMENT


            # Set Seed
            seed = environ.set_seed(123)

            # Create Environment
            base_env = environ.make_env(env_name, seed=seed, ram=True)
            eval_env = environ.make_eval_env(env_name, seed=seed, ram=True, render=False)

            ### CREATE AGENT
            action_size = base_env.action_space.n
            ohe_matrix = np.eye(action_size)

            # We require shape of (frames, image_size, image_size)
            state_size = base_env.observation_space.shape

            agent = DQN_Agent_Combined(ram=True,
                                       state_size=state_size[0],    
                                       action_size=action_size,
                                       layer_size=128,
                                       n_step=1,
                                       # batch_size=agent_batch_size, 
                                       buffer_size=500_000, 
                                       learning_rate=0.0002, #025, #1e-4 #0.0002
                                       tau=1e-3, # 0.005
                                       gamma=0.95, 
                                       update_freq=4, 
                                       device=device, 
                                       seed=seed,
                                       # gener=agent_gener,
                                       # gan_type=agent_gan_type,
                                       gan_smoothing_factor=0.1,
                                       gan_upsample=False,
                                       gan_spectral=False,
                                       gan_batch_norm=True,
                                       gan_learning_rate=0.00002,
                                       gan_disc_learning_rate_division=2.0,
                                       # gan_mix=agent_gan_mix,
                                       gan_lambda_gp=20,
                                       # combined_buffer=agent_combined_buffer
                                       **params
                                      )

            ### Fill Buffer
            for sample_idx in range(data_size):
                agent.memory_buffer.add(np_states[sample_idx,:], 
                                        np_actions[sample_idx,:],
                                        np_rewards[sample_idx],
                                        np_next_states[sample_idx,:],
                                        np_dones[sample_idx]
                                       )

            ### Run Test
            evaluation_scores = offline_run(eval_env, 
                                            agent, 
                                            np_states, 
                                            np_actions,
                                            np_rewards,
                                            np_next_states,
                                            np_dones, 
                                            frames=training_frames, 
                                            eval_every=eval_every, 
                                            eval_runs=10,
                                            run_eval=True
                                           )

            # frame, eval_score, eval_score_std
            results_df = pd.DataFrame(evaluation_scores, columns=['IDX', 'Evaluation Score', 'Evaluation Score STD'])

            results_df.to_csv(f'./{run_name}-results.csv')