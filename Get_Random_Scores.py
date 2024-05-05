# This script will provide scores and recordings for an agent acting randomly in each environment.  

# To compare with our RAM agents, we will measure max 100 episode average reward and max episode reward across 2 million frames.

### Environment
import Environment as environ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import gym

# Set Seed
seed = environ.set_seed(123)

# Root Directory to save videos in
root = "./Videos/"

# Required subdirectories
paths = [root, './Random_Agent/']

for path in paths:
    # check for output directory
    if os.path.exists(path) and os.path.isdir(path):
        print('Output Directory Present.')
    # If not present create new directory
    else:
        print(f'No Output Directory. Creating at {root}.')
        os.mkdir(path)

env_list = ["SpaceInvaders-v4", "Pong-v4", "Breakout-v4", "Seaquest-v4", "BattleZone-v4"]

def clean_env_name(env_name_str):
    """
    Clean environment name by removing version details and adding spaces.
    """
    version_removed = env_name_str.split('-')[0]
    space_added = re.sub(r'(?<!^)(?=[A-Z])', ' ', version_removed)
    return space_added

### Random Agent - Get Scores

frames_to_run = 3_000


random_agent_results = list()

for env_name in env_list:
    # Initialise empty results list
    base_scores = list()
    eval_scores = list()
    
    print(f'Running random agent for {env_name}')
    print()
    
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
    
    # Set done flags to false
    base_done = False
    eval_done = False
    
    base_episode_reward = 0
    eval_episode_reward = 0
    
    for step_idx in range(frames_to_run):
        print(f'Frame: {step_idx:,}', end='\r')
        # Choose Action
        action = np.random.randint(action_size)
        
        # Step Environment (Apply Action)
        processed_state, base_reward, base_done, *_ = base_env.step(action)
        state, eval_reward, eval_done, *_ = eval_env.step(action)
        
        # Add reward to episode reward
        base_episode_reward += base_reward
        eval_episode_reward += eval_reward
        
        
        if base_done:
            # Store and reset reward
            base_scores.append(base_episode_reward)
            base_episode_reward = 0
            
            # Reset environment
            state, _ = base_env.reset()
        if eval_done:
            # Store and reset reward
            eval_scores.append(eval_episode_reward)
            eval_episode_reward = 0
            
            # Reset environment
            state, _ = eval_env.reset()
    
    print()
    print(f'Base Reward Average {np.mean(base_scores):.2f}')
    print(f'Eval Reward Average {np.mean(eval_scores):.2f}')
    print(f'Base Reward Max {np.max(base_scores):.2f}')
    print(f'Eval Reward Max {np.max(eval_scores):.2f}')
    print(f'Episodes {len(eval_scores):,}')
    
    avg_episode_length = frames_to_run / float(len(eval_scores)) 
    print(f'Average Episode Length {avg_episode_length:.2f}')

    random_agent_results.append({'env_name':env_name, 
                                        'base_reward':base_scores, 
                                        'eval_reward':eval_scores
                                       })
    
    print()

### Analyse Results

results = list()

for idx, env_results in enumerate(random_agent_results):
    row = dict()
    
    row['env_name'] = env_results['env_name']
    row['Base Reward Average'] = np.mean(env_results['base_reward'])
    row['Eval Reward Average'] = np.mean(env_results['eval_reward'])
    row['Base Reward Max'] = np.max(env_results['base_reward'])
    row['Eval Reward Max'] = np.max(env_results['eval_reward'])
    row['Base Reward Min'] = np.min(env_results['base_reward'])
    row['Eval Reward Min'] = np.min(env_results['eval_reward'])
    row['Base Reward STD'] = np.std(env_results['base_reward'])
    row['Eval Reward STD'] = np.std(env_results['eval_reward'])
    
    row['Episodes'] = len(env_results['eval_reward'])
    row['Average Episode Length'] = frames_to_run / float(len(env_results['eval_reward'])) 
    
    results.append(row)
    
results_df = pd.DataFrame(results)

results_df.to_csv('./Random_Agent/Scores.csv')

print(results_df.head())

### Create Videos

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        _, detail = self.env.reset()
        obs, _, done, _, detail = self.env.step(1)
        if done:
            _, detail = self.env.reset()
        obs, _, done, _, detail = self.env.step(2)
        if done:
            _, detail = self.env.reset()
        return obs, detail

for env_name in env_list:

    cleaned_env_name = clean_env_name(env_name)
    
    print(f'Generate Random Agent video for {cleaned_env_name}')

    # Set False for random agent
    agent = False

    eval_env = gym.make(env_name, repeat_action_probability=0, render_mode="rgb_array")

    if agent:
        # If agent, set each action to repeat 4 times
        eval_env = MaxAndSkipEnv(eval_env, max_buffer_len=1)
    if seed:
        eval_env.seed(seed)
    if 'FIRE' in eval_env.unwrapped.get_action_meanings():
        eval_env = FireResetEnv(eval_env)

    # Add Video Wrapper
    env = gym.wrappers.RecordVideo(env=eval_env, 
                                   video_folder="./Videos/", 
                                   name_prefix=f"{cleaned_env_name} Random Agent", 
                                   episode_trigger=lambda x: x % 2 == 0
                                  )

    # wrap the env in the record video

    # env reset for a fresh start
    state, _ = env.reset()

    # Start the recorder
    env.start_video_recorder()


    # Run random actions
    for _ in range(10_000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        state, reward, terminated, truncated, info = env.step(action)
        # env.render()

        if terminated or truncated:
            state, info = env.reset()

    ####
    # Don't forget to close the video recorder before the env!
    env.close_video_recorder()

    # Close the environment
    env.close()