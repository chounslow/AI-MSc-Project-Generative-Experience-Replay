# This script extracts example states from a list of Atari Environments

### Environment
import Environment as environ
from collections import deque
import numpy as np
import os
import matplotlib.pyplot as plt

# Set Seed
seed = environ.set_seed(123)

# Root Directory to save in
root = './Environments/Examples/'

paths = ['./Environments/', './Environments/Examples/']

for path in paths:
    # check for output images subdirectory
    if os.path.exists(path) and os.path.isdir(path):
        print('Output Directory Present.')
    # If not present create new directory
    else:
        print(f'No Output Directory. Creating at {path}.')
        os.mkdir(path)

env_list = ["SpaceInvaders-v4", "Pong-v4", "Breakout-v4", "Seaquest-v4", "BattleZone-v4"]

### Get Initial Frames
examples = list()

for env_name in env_list:
    print(env_name)
    
    # Set Seed
    seed = environ.set_seed(123)

    # Create Environments
    base_env = environ.make_env(env_name, seed=seed, ram=False)
    eval_env = environ.make_eval_env(env_name, seed=seed, ram=False, render=True)

    # Reset environment
    state, _ = eval_env.reset()
    processed_state, _ = base_env.reset()
    
    ### Format Unprocessed State
    
    # Scale from 0 -> +255 to 0 -> +1
    formatted_state = (state) / 255.0
    
    ### Format Processed State
    
    # Convert to height * width * Channel
    processed_state = np.moveaxis(processed_state, 0, 2)

    # Scale from -1 -> +1 to 0 -> +1
    processed_state = (processed_state + 1.0) / 2.0
    
    examples.append({'env_name':env_name, 'initial_state':formatted_state, 'processed_initial_state':processed_state})
    
## Create Initial State Plot

# Create Plot
import re

def clean_env_name(env_name_str):
    """
    Clean environment name by removing version details and adding spaces.
    """
    version_removed = env_name_str.split('-')[0]
    space_added = re.sub(r'(?<!^)(?=[A-Z])', ' ', version_removed)
    return space_added


# Create subplot for 5 images
fig, axes = plt.subplots(1, 5, figsize=(10, 3))

# Plot each image with its name above
for i, ax in enumerate(axes):
    ax.imshow(examples[i]['initial_state'])  # Assuming grayscale images , cmap='gray'
    cleaned_env_name = clean_env_name(examples[i]['env_name'])
    ax.set_title(cleaned_env_name, fontsize=10)
    ax.axis('off')

plt.tight_layout()

plt.savefig(root+'Initial_Frame_All_Games.png')

plt.close()

### Create Plot for Processed States
# Create subplot for 5 images
fig, axes = plt.subplots(1, 5, figsize=(10, 3))

# Plot each image with its name above
for i, ax in enumerate(axes):
    # Show last frame only
    ax.imshow(examples[i]['processed_initial_state'][:,:,-1], cmap='gray')
    cleaned_env_name = clean_env_name(examples[i]['env_name'])
    ax.set_title(cleaned_env_name, fontsize=10)
    ax.axis('off')

plt.tight_layout()

plt.savefig(root+'Initial_Frame_All_Games_Processed.png')

plt.close()

### Get Sequence of Frames

sequence_length = 5

sequence_examples = list()

for env_name in env_list:
    print(f'Extracting frames for {env_name}')
    
    # Set Seed
    seed = environ.set_seed(123)

    # Create Environments
    base_env = environ.make_env(env_name, seed=seed, ram=False)
    eval_env = environ.make_eval_env(env_name, seed=seed, ram=False, render=True)

    ### Agent
    action_size = base_env.action_space.n

    # Reset environment
    state, _ = eval_env.reset()
    processed_state, _ = base_env.reset()
    
    sequence = [state]
    processed_sequence = [processed_state]
    
    for step_idx in range((sequence_length * 2) - 1):
        # Choose Action
        action = np.random.randint(action_size)
        
        # Step Environment (Apply Action)
        state, *_ = eval_env.step(action)
        processed_state, *_ = base_env.step(action)
        
        # Store Resulting State
        sequence.append(state)
        processed_sequence.append(processed_state)
        
    
    # Format Unprocessed States
    # Scale from 0 -> +255 to 0 -> +1
    sequence = [state / 255.0 for state in sequence]
    
    ### Format Processed State
    
    # Convert to height * width * Channel
    processed_sequence = [np.moveaxis(processed_state, 0, 2) for processed_state in processed_sequence]

    # Scale from -1 -> +1 to 0 -> +1
    processed_sequence = [(processed_state + 1.0) / 2.0 for processed_state in processed_sequence]
    
    sequence_examples.append({'env_name':env_name, 'sequence':sequence[::2], 'processed_sequence':processed_sequence[::2]})
    
### Create Sequence Plot

## Unprocessed
rows = 5
columns = 5

# Create subplot for 5 images
fig, axes = plt.subplots(nrows=rows, ncols=columns) #, figsize=(10, 3))

# Plot each image with its name above
for row_idx in range(rows):
    for col_idx in range(columns):
        axes[row_idx, col_idx].imshow(sequence_examples[row_idx]['sequence'][col_idx], aspect='auto') #, interpolation='none')
        
        axes[row_idx, col_idx].axis('off')

plt.tight_layout()

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.02, hspace=0.05)

plt.savefig(root+'Sequences_All_Games.png')

plt.close()

## Processed

rows = 5
columns = 5

# Create subplot for 5 images
fig, axes = plt.subplots(nrows=rows, ncols=columns) #, figsize=(10, 3))

# Plot each image with its name above
for row_idx in range(rows):
    for col_idx in range(columns):
        # Show last frame only
        axes[row_idx, col_idx].imshow(sequence_examples[row_idx]['processed_sequence'][col_idx][:,:,-1], 
                                      aspect='auto', 
                                      cmap='gray'
                                     )
        
        
        axes[row_idx, col_idx].axis('off')

plt.tight_layout()

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.02, hspace=0.05)

plt.savefig(root+'Sequences_All_Games_Processed.png')

plt.close()

### Create Rewarding Sequence Plot

sequence_length = 5

rewarding_sequence_examples = list()

for env_name in env_list:
    print(f'Extracting frames for {env_name}')
    
    # Set Seed
    seed = environ.set_seed(123)

    # Create Environments
    base_env = environ.make_env(env_name, seed=seed, ram=False)
    eval_env = environ.make_eval_env(env_name, seed=seed, ram=False, render=True)

    ### Agent
    action_size = base_env.action_space.n

    # Reset environment
    state, _ = eval_env.reset()
    processed_state, _ = base_env.reset()
    
    sequence = deque(maxlen=sequence_length)
    processed_sequence = deque(maxlen=sequence_length)
    
    sequence.append(state)
    processed_sequence.append(processed_state)
    
    initial_steps = 100
    
    done = False
        
    
    for step_idx in range(100_000):
        # Choose Action
        action = np.random.randint(action_size)
        
        # Step Environment (Apply Action)
        state, reward, done, *_ = eval_env.step(action)
        # processed_state, reward, done, *_ = base_env.step(action)
        
        if done:
            # Clear stored sequence
            sequence.clear()
            state, _ = base_env.reset()
        
        # Store Resulting State
        sequence.append(state)
        
        if reward > 0:
            reward_step_idx = step_idx
            print(f'Found Reward: {reward} at step: {reward_step_idx}.')
            
            for extra_step_idx in range(2):
                # Choose Action
                action = np.random.randint(action_size)
        
                # Step Environment (Apply Action)
                state, reward, done, *_ = eval_env.step(action)
                # processed_state, reward, done, *_ = base_env.step(action)
        
                # Store Resulting State
                sequence.append(state)
            # exit loop    
            break
    
    # Format Unprocessed States
    # Scale from 0 -> +255 to 0 -> +1
    sequence = [state / 255.0 for state in sequence]
    
    
    rewarding_sequence_examples.append({'env_name':env_name, 
                                        'sequence':sequence, 
                                        'step':reward_step_idx
                                        # 'processed_sequence':processed_sequence
                                       })
    
rows = 5
columns = 3

# Create subplot for 5 images
fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(5, 7))

# Plot each image with its name above
for row_idx in range(rows):
    for col_idx in range(columns):
        axes[row_idx, col_idx].imshow(rewarding_sequence_examples[row_idx]['sequence'][col_idx], 
                                      aspect='auto')
        
        axes[row_idx, col_idx].axis('off')

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.02, hspace=0.05)

plt.savefig(root+'Rewarding_Sequences_All_Games.png')

plt.close()