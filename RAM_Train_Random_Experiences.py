# Base Imports
import random
import os
from collections import defaultdict, deque
import datetime as dt
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Environment Imports
import gym
from gymnasium.wrappers import RescaleAction, TimeLimit, RecordEpisodeStatistics, RecordVideo, NormalizeObservation, TransformReward
# from gymnasium.utils.save_video import save_video
from gymnasium.utils.seeding import np_random
import ale_py

# PyTorch Imports
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# Other Imports
import cv2
from scipy.stats import ks_2samp, pearsonr


# Relative Imports
import Environment as environ
from DQN import *

# Parameters
env_name = "Pong-ram-v4"

agent_batch_size = 16*4 # 16 if not gener, 16 * 4 if gener
agent_gener = True
agent_gan_type = 'WGAN' #DCGAN WGAN
agent_gan_mix = 0.5 #SynthER Paper uses 0.5 for this parameter
agent_combined_buffer = True

disc_sampling = True
gan_sample_selection = disc_sampling

# Number of Epochs to train
n_epochs = 5 # 5
number_of_states = 100_000 # 100_000


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')


### Functions

def run_random_policy(random_frames, env, agent):
    """
    Run env with random policy for x frames to fill the replay memory.
    """
    
    action_size = env.action_space.n

    ohe_matrix = np.eye(action_size)
    
    state, _ = env.reset()
    rewards = list()
    current_reward = 0
    for i in range(random_frames):
        print(f'Episode IDX: {i}', end='\r')
        action = np.random.randint(action_size)
        next_state, reward, done, info, detail = env.step(action)
        
        current_reward += reward

        agent.memory_buffer.add(state, ohe_matrix[action], reward, next_state, done)
        state = next_state
        if done == 1:
            state, _ = env.reset()
            rewards.append(current_reward)
            current_reward = 0
    print('\nCompleted.')
    avg_reward = np.mean(rewards)
    print(f'{avg_reward = :.2}')
    
    
def get_states(random_frames, env):
    """
    Run env with random policy for x frames and return experiences.
    """
    action_size = env.action_space.n

    ohe_matrix = np.eye(action_size)
    
    outputs = list()
    # obs = list()
    states = list()
    actions = list()
    rewards = list()
    # next_states = list()
    dones = list()
    state, _ = env.reset() 
    for i in range(random_frames):
        print(f'Episode IDX: {i}', end='\r')
        
        # Randomly choose action and step environment
        action = np.random.randint(action_size)
        next_state, reward, done, info, detail = env.step(action)
        
        states.append(state)
        actions.append(ohe_matrix[action])
        rewards.append(reward)
        # next_states.append(next_states)
        dones.append(done)
        
        state = next_state
        
        if done == 1:
            state, _ = env.reset()
        

    print('\nCompleted.')
    return np.array(states), np.array(actions), np.array(rewards), np.array(states[1:] + [next_state]), np.array(dones)

class rl_dataset(Dataset):
    def __init__(self, states, actions, rewards, next_states, dones):
        # self.obs = obs
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        
    def __getitem__(self, index):
        # obs_sample = self.obs[index,:]
        states_sample = self.states[index,:]
        actions_sample = self.actions[index, :]
        rewards_sample = self.rewards[index]
        next_states_sample = self.next_states[index,:]
        dones_sample = self.dones[index]

        
        return states_sample, actions_sample, rewards_sample, next_states_sample, dones_sample
    
    def __len__(self):
        return len(self.dones)
    
def evaluate(agent, eval_env, eval_runs=10):
    """
    Makes an evaluation runs with eps 0
    """
    reward_batch = []
    for i in range(eval_runs):
        state, _ = eval_env.reset()
        rewards = 0
        steps = 0
        while True and steps < 100_000:
            action = agent.act(state[None,:], 0.0)[0]
            # print(action)
            next_state, reward, done, _, _ = eval_env.step(action)
            state = next_state
            rewards += reward
            steps += 1
            if done:
                break
        reward_batch.append(rewards)
        
    return np.mean(reward_batch)

def run(env, eval_env, frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.05, eval_every=100_000, eval_runs=3, run_eval=True):
    """Deep Q-Learning.
    
    Args:
        frames (int): Number of environment frames to train for.
        eps_fixed (bool): Indicates whether to fix episilon or adjust over time.
        eps_frames (float): Number of frames to decay epsilon exponentially.
        min_eps (float): Minimum value of epsilon to fall to.
        eval_every (int): Number of frames to run evaluation.
        eval_runs (int): number of evaluation runs.
    """
    # List containing scores for each episode
    scores = list()
    # List containing average scores for each evaluation step
    evaluation_scores = list()
    scores_window = deque(maxlen=100)  # last 100 scores
    frame = 0
    frame_count_last_episode = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    d_eps = eps_start - min_eps
    i_episode = 1
    state, _ = env.reset()
    score = 0                  
    for frame in range(1, frames+1):
        action = agent.act(state[None,:], eps)[0]
        next_state, reward, done, _, _ = env.step(action)
        
        agent.step(state, ohe_matrix[action], reward, next_state, done, writer)
        state = next_state
        score += reward
        # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            eps = max(eps_start - ((frame*d_eps)/eps_frames), min_eps)


        # evaluation runs
        if (frame % eval_every == 0 or frame == 1) and run_eval:
            print('Running Evaluation')
            avg_score = evaluate(agent, eval_env, 5)
            evaluation_scores.append([frame, i_episode, avg_score])

        if done == 1:
            scores_window.append(score)       
            scores.append([frame, i_episode, score])              
            items_to_print = ['Episode {}', 'Ep Frames {}', 'Total Frames {}', 'Episode Score  {:.2f}', 'Average Score: {:.2f}', '']
            print_str = '\t '.join(items_to_print)
            if i_episode % 100 == 0:
                end = '\n' if i_episode % 1000 == 0 else '\r'
                # print(end)
                print(print_str.format(i_episode,
                                       frame-frame_count_last_episode,
                                        frame, 
                                        score, 
                                        np.mean(scores_window)), end=end)
            
    
            i_episode +=1 
            state, _ = env.reset()
            score = 0
            frame_count_last_episode = frame

    return scores, evaluation_scores

def marginal_mean_ks(real_data: np.ndarray, fake_data: np.ndarray) -> float:
    # Ensure both fake and real data have the same shape
    assert real_data.shape == fake_data.shape, "Data shapes must match"

    # Number of dimensions
    num_dimensions = fake_data.shape[1]

    # Initialize array to store KS statistics for each dimension
    ks_statistics = np.zeros(num_dimensions)

    # Compute KS statistic for each dimension
    for i in range(num_dimensions):
        # Extract data for the current dimension
        fake_dim = fake_data[:, i]
        real_dim = real_data[:, i]

        # Compute empirical CDFs
        fake_cdf = np.sort(fake_dim)
        real_cdf = np.sort(real_dim)

        # Compute KS statistic for the current dimension
        ks_statistic, _ = ks_2samp(fake_cdf, real_cdf)

        # Store KS statistic
        ks_statistics[i] = ks_statistic

    # Compute the marginal mean KS statistic
    marginal_mean_ks_statistic = np.mean(ks_statistics)

    return marginal_mean_ks_statistic

def mean_correlation_similarity(real_data: np.ndarray, fake_data: np.ndarray) -> float:
    # Ensure both fake and real data have the same shape
    assert real_data.shape == fake_data.shape, "Data shapes must match"

    # Number of dimensions
    num_dimensions = real_data.shape[1]

    # Initialize array to store correlation differences for each pair of dimensions
    correlation_differences = np.zeros(num_dimensions)

    # Compute Pearson correlation coefficients for each pair of dimensions
    for i in range(num_dimensions):
        fake_dim = fake_data[:, i]
        real_dim = real_data[:, i]
        
        # RAM values can be a constant (e.g. always 1)
        # If this occurs skip this dimension
        if min(fake_dim) == max(fake_dim) or min(real_dim) == max(real_dim):
            continue

        # Compute Pearson correlation coefficient
        correlation_coefficient, _ = pearsonr(fake_dim, real_dim)

        # Store absolute difference between fake and real correlation coefficients
        correlation_differences[i] = np.abs(1 - correlation_coefficient)

    # Compute the mean correlation similarity
    mean_correlation_similarity = np.mean(correlation_differences)

    return mean_correlation_similarity

def numpy_to_torch(states, actions, rewards, next_states, dones, device):
    """
    Convert RL Attributes from numpy to Torch Tensor.
    
    """
    states_torch = torch.from_numpy(np.stack(states)).float().to(device)
    actions_torch = torch.from_numpy(np.vstack(actions)).long().to(device)
    rewards_torch = torch.from_numpy(np.vstack(rewards)).float().to(device)
    next_states_torch = torch.from_numpy(np.stack(next_states)).float().to(device)
    dones_torch = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        
    return states_torch, actions_torch, rewards_torch, next_states_torch, dones_torch


# Set Seed
seed = environ.set_seed(123)

# Create Environment

base_env = environ.make_env(env_name, seed=seed, ram=True)
eval_env = environ.make_eval_env(env_name, seed=seed, ram=True, render=False)

action_size = base_env.action_space.n

ohe_matrix = np.eye(action_size)

# We require shape of (frames, image_size, image_size)
state_size = base_env.observation_space.shape #[:-1]

if agent_gener:
    if agent_combined_buffer:
        buffer_name = 'Combined'
        gan_mix_str = '-' + str(agent_gan_mix) + '-'
        if gan_sample_selection:
            gan_mix_str += 'Disc_Sampling'
    else:
        buffer_name = 'Gen'
        gan_mix_str = ''
    buffer_type = '-' + agent_gan_type + '-'
else:
    buffer_name = 'Memory'
    buffer_type = ''
    gan_mix_str = ''

agent_name =  f'{env_name}-BatchSize-{agent_batch_size}-{buffer_name}{gan_mix_str}{buffer_type}Buffer'

print('Agent Name: ' + agent_name)

# Create Agent
agent = DQN_Agent_Combined(
    ram=True,
    state_size=state_size[0],    
    action_size=action_size,
    layer_size=128,
    n_step=1,
    batch_size=agent_batch_size, 
    buffer_size=500_000, 
    learning_rate=0.0002, #025, #1e-4 #0.0002
    tau=1e-3, # 0.005
    gamma=0.95, 
    update_freq=4, # 4
    device=device, 
    seed=seed,
    gener=agent_gener,
    gan_type=agent_gan_type,
    gan_smoothing_factor=0.1,
    gan_upsample=False,
    gan_spectral=False,
    gan_batch_norm=True,
    gan_learning_rate=0.00002,
    gan_disc_learning_rate_division=2.0, # 2
    gan_mix=agent_gan_mix,
    gan_lambda_gp=20, # 20
    combined_buffer=agent_combined_buffer,
    gan_sample_selection=gan_sample_selection
                     )


get_new_states = True

# Check if states already exist
if os.path.isfile(f'{env_name}_states_ram.npy'):
    print('Files Detected.')
    get_new_states = False
    
else:
    print('No Files Detected.')

if get_new_states:
    print(f'Getting states.')
    # Get states from random actions in the environment
    np_states, np_actions, np_rewards, np_next_states, np_dones = get_states(number_of_states, base_env)

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

# Load data as a dataset object
sample_len = 100_000
# Sample length for real and generated sets
testing_sample_len = 50_000

# Get test values for real data
test_states, test_actions, test_rewards, test_next_states, test_dones = numpy_to_torch(np_states[:testing_sample_len].copy(), 
                                                                                       np_actions[:testing_sample_len].copy(),
                                                                                       np_rewards[:testing_sample_len].copy(),
                                                                                       np_next_states[:testing_sample_len].copy(),
                                                                                       np_dones[:testing_sample_len].copy(),
                                                                                       device
                                                                                      )

test_observations = torch.cat((test_states, test_next_states), axis=1)

test_combined = torch.cat((test_states, test_actions, test_rewards, test_next_states, test_dones), axis=1)
test_combined = test_combined.cpu().numpy()

# Create training dataset with real data
dataset = rl_dataset(np_states[:sample_len].copy(), 
                     np_actions[:sample_len].copy(), 
                     np_rewards[:sample_len].copy(), 
                     np_next_states[:sample_len].copy(), 
                     np_dones[:sample_len].copy())

### Create Dataloader

def np_range(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)

real_pre_mean = np_states.mean(axis=0)
real_post_mean = np_next_states.mean(axis=0)
real_pre_std = np_states.std(axis=0)
real_post_std = np_next_states.std(axis=0)
real_pre_range = np_range(np_states, axis=0)
real_post_range = np_range(np_next_states, axis=0)

batch_size = 32

dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True
                           )


### Train in epochs and store samples 

samples = list()

performance_stats = list()

for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}")
    for idx, real in enumerate(tqdm(dataloader)):

        states, actions, rewards, next_states, dones = real
        states, actions, rewards, next_states, dones = numpy_to_torch(states, actions, rewards, next_states, dones, device)
        
        agent.memory_buffer.train_loop(states, 
                                       actions, 
                                       rewards, # [:,None]
                                       next_states,
                                       dones, 
                                       save_images=False)
    
        length = states.shape[0]
        
        if idx % 100 == 0 and idx != 0:
            # Get Generated States
            sample = agent.memory_buffer.sample(testing_sample_len)
            
            ### Get Accuracy of Discriminator
            with torch.no_grad():
                # Run Discriminator on test dataset of real images
                disc_test_results = agent.memory_buffer.disc(test_observations, test_actions, test_rewards, test_dones)
                mean_real_detection = disc_test_results.mean().cpu().numpy()
                
                # Run Discriminator on generated sample
                sample_obvs = torch.cat((sample.state, sample.next_state), 1)
                disc_fake_results = agent.memory_buffer.disc(sample_obvs, sample.action, sample.reward, sample.done)
                mean_fake_detection = disc_fake_results.mean().cpu().numpy()
                
            
            ### Store 5 Generate Images and Step Number
            
            # Get set of 5 states
            sample_states = sample.state[:5].cpu()
            # Get next state
            sampled_next_states = sample.next_state[:5].cpu()
            # Combine for 5, 5, image_size shape
            image_stack = np.concatenate([sample_states, sampled_next_states], axis=1)
            
            samples.append([len(agent.memory_buffer)*batch_size, image_stack])
            
            ### Store difference between AVG value for each RAM byte
            pre_mean = sample.state.mean(axis=0).cpu().numpy()
            pre_mean_diff = np.abs(pre_mean - real_pre_mean).mean()
            
            pre_std = sample.state.std(axis=0).cpu().numpy()
            pre_std_diff = np.abs(pre_std - real_pre_std).mean()
            
            pre_range = np_range(sample.state.cpu().numpy(), axis=0)
            pre_range_diff = np.abs(pre_range - real_pre_range).mean()
            
            post_mean = sample.next_state.mean(axis=0).cpu().numpy()
            post_mean_diff = np.abs(post_mean - real_post_mean).mean()
            
            post_std = sample.next_state.std(axis=0).cpu().numpy()
            post_std_diff = np.abs(post_std - real_post_std).mean()
            
            post_range = np_range(sample.next_state.cpu().numpy(), axis=0)
            post_range_diff = np.abs(post_range - real_post_range).mean()
            
            sample_combined = torch.cat((sample.state, sample.action, sample.reward, sample.next_state, sample.done), axis=1)
            sample_combined = sample_combined.cpu().numpy()
            
            marginal_mean_ks_statistic = marginal_mean_ks(test_combined, sample_combined)
            mean_correlation_similarity_statistic = mean_correlation_similarity(test_combined, sample_combined)
            
            
            # Print loss after every epoch        
            step, gen_loss, disc_loss = agent.memory_buffer.get_loss()
            performance_stats.append([step, 
                                      gen_loss, disc_loss, 
                                      pre_mean_diff, post_mean_diff, 
                                      pre_std_diff, post_std_diff,
                                      pre_range_diff, post_range_diff,
                                      mean_real_detection, mean_fake_detection,
                                      marginal_mean_ks_statistic,
                                      mean_correlation_similarity_statistic
                                     ]
                                    )

df = pd.DataFrame(performance_stats, columns=['Step', 
                                              'Gen Loss', 
                                              'Disc Loss', 
                                              'Pre Mean Diff', 
                                              'Post Mean Diff', 
                                              'Pre STD Diff',
                                              'Post STD Diff',
                                              'Pre Range Diff', 
                                              'Post Range Diff',
                                              'Mean Real Output',
                                              'Mean Fake Output',
                                              'Marginal',
                                              'Correlation'
                                             ]
                 )

f_name = 'Epoch_Training_' + agent_name + '_Tabular_Results' + '.csv'

print(f'Saving CSV File: {f_name}')

df.to_csv(f_name, index=False)

print(df.round(2).head(20))

fig, axes = plt.subplots(4, 2, sharex=True,
                        figsize=(16,8)
                        )

## Create Loss Plot

# Plotting
axes[0, 0].plot(df['Step'], df['Gen Loss'], color='blue', label='Generator Loss')
axes[0, 0].plot(df['Step'], df['Disc Loss'], color='red', label='Discriminator Loss')

# Adding labels, title & legend
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Gen & Disc Loss')
axes[0, 0].legend()

## Create Mean of State Plot

# Plotting
axes[1, 0].plot(df['Step'], df['Pre Mean Diff'], color='blue', label='Pre Mean Diff')
axes[1, 0].plot(df['Step'], df['Post Mean Diff'], color='red', label='Post Mean Diff')

# Adding labels, title & legend
axes[1, 0].set_ylabel('Delta')
axes[1, 0].set_title('Mean Generated RAM State Difference')
axes[1, 0].legend()

## Create STD of State Plot

# Plotting
axes[1, 1].plot(df['Step'], df['Pre STD Diff'], color='blue', label='Pre STD Diff')
axes[1, 1].plot(df['Step'], df['Post STD Diff'], color='red', label='Post STD Diff')

# Adding labels, title & legend
axes[1, 1].set_ylabel('Delta')
axes[1, 1].set_title('STD Generated RAM State Difference')
axes[1, 1].legend()

## Create Range of State Plot

# Plotting
axes[0, 1].plot(df['Step'], df['Pre Range Diff'], color='blue', label='Pre Range Diff')
axes[0, 1].plot(df['Step'], df['Post Range Diff'], color='red', label='Post Range Diff')

# Adding labels and title
axes[0, 1].set_ylabel('Delta')
axes[0, 1].set_title('Range Generated RAM State Difference')
axes[0, 1].legend()

## Create Mean Disc Prediction Plot

# Plotting
axes[2, 0].plot(df['Step'], df['Mean Real Output'], color='blue', label='Mean Real Output')
axes[2, 1].plot(df['Step'], df['Mean Fake Output'], color='red', label='Mean Fake Output')

# Adding labels and title
axes[2, 0].set_ylabel('Mean Prediction')
axes[2, 1].set_ylabel('Mean Prediction')
axes[2, 0].set_title('Mean Disc Prediction on Real Images')
axes[2, 1].set_title('Mean Disc Prediction on Fake Images')

## Create Marginal & Correlation Plots

# Plotting
axes[3, 0].plot(df['Step'], df['Marginal'], color='blue', label='Marginal')
axes[3, 1].plot(df['Step'], df['Correlation'], color='red', label='Correlation')

# Adding labels and title
axes[3, 0].set_ylabel('Mean Score')
axes[3, 1].set_ylabel('Mean Score')
axes[3, 0].set_title('Mean Marginal Score')
axes[3, 1].set_title('Mean Correlation Score')

# Final Details
axes[3, 0].set_xlabel('Step')
axes[3, 1].set_xlabel('Step')

# Saving Figure
f_name = 'Epoch_Training_' + agent_name + '_Figures' + '.png'

print(f'Saving PNG File: {f_name}')

plt.savefig(f_name)

plt.close()