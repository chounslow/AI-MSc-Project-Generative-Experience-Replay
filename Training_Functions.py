import numpy as np
from collections import deque

import torch
from torch.utils.data import Dataset


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

def run(agent, env, eval_env, writer, frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.05, eval_every=100_000, eval_runs=3, run_eval=True):
    """Deep Q-Learning.
    
    Args:
        frames (int): Number of environment frames to train for.
        eps_fixed (bool): Indicates whether to fix episilon or adjust over time.
        eps_frames (float): Number of frames to decay epsilon exponentially.
        min_eps (float): Minimum value of epsilon to fall to.
        eval_every (int): Number of frames to run evaluation.
        eval_runs (int): number of evaluation runs.
    """
    
    action_size = env.action_space.n

    ohe_matrix = np.eye(action_size)
    
    
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

def run_and_save(agent, env, eval_env, writer, env_name, agent_name, frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.05, eval_every=100_000, eval_runs=3, 
        run_eval=True,
       storage_root=None,
       idx_shift=0):
    """Runs Deep Q-Learning training and saves models and experiences in storage root
    
    Args:
        frames (int): Number of environment frames to train for.
        eps_fixed (bool): Indicates whether to fix episilon or adjust over time.
        eps_frames (float): Number of frames to decay epsilon exponentially.
        min_eps (float): Minimum value of epsilon to fall to.
        eval_every (int): Number of frames to run evaluation.
        eval_runs (int): number of evaluation runs.
        storage_root (string): Location to store model, and experiences in.
        idx_shift (int): Index shift for saved states when running agent from checkpoint.
    """
    
    action_size = env.action_space.n

    ohe_matrix = np.eye(action_size)
    
    # List containing scores for each episode
    scores = list()
    # List containing average scores for each evaluation step
    evaluation_scores = list()
    scores_window = deque(maxlen=100)  # last 100 scores
    frame = 0
    frame_count_last_episode = 0
    if eps_fixed:
        eps = min_eps
    else:
        eps = 1
    eps_start = 1
    d_eps = eps_start - min_eps
    i_episode = 1
    state, _ = env.reset()
    score = 0                  
    for frame in range(1, frames+1):
        # print(state.shape)
        action = agent.act(state[None,:], eps)[0]
        # print(action)
        next_state, reward, done, _, _ = env.step(action)
        
        agent.step(state, ohe_matrix[action], reward, next_state, done, writer)
        # for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            # agent.step(s, a, r, ns, d, writer)
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
            
        # Store Agent & Experiences
        # if (frame >= 1_000_000) and (frame % 100_000 == 0):
        if (frame % 100_000 == 0):
            print('Saving Agent & Experiences')
            states, actions, rewards, next_states, dones = agent.memory_buffer.get_history(batch_size=100_000)
            
            idx = frame // 100_000
            idx += idx_shift
            print(idx)

            np.save(storage_root + f'{env_name}_offline_states_ram_{idx}.npy', states.cpu().numpy())
            np.save(storage_root + f'{env_name}_offline_actions_ram_{idx}.npy', actions.cpu().numpy())
            np.save(storage_root + f'{env_name}_offline_rewards_ram_{idx}.npy', rewards.cpu().numpy())
            np.save(storage_root + f'{env_name}_offline_next_states_ram_{idx}.npy', next_states.cpu().numpy())
            np.save(storage_root + f'{env_name}_offline_dones_ram_{idx}.npy', dones.cpu().numpy())

            del states, actions, rewards, next_states, dones

            # Save Agent
            torch.save(agent.qnetwork_local.state_dict(), storage_root + f'{agent_name}-qnetwork_local-{idx}')
            torch.save(agent.qnetwork_target.state_dict(), storage_root + f'{agent_name}-qnetwork_target-{idx}')
            
            print('Save Complete')


        if done == 1:
            # save most recent score for window measure
            scores_window.append(score)       
            # save frame count, episode count and score
            scores.append([frame, i_episode, score])              
            # writer.add_scalar("Average100", np.mean(scores_window), i_episode)
            items_to_print = ['Episode {}', 'Ep Frames {}', 'Total Frames {}', 'Episode Score  {:.2f}', 'Average Score: {:.2f}', '']
            print_str = '\t '.join(items_to_print)
            if i_episode % 100 == 0:
                end = '\n' if i_episode % 1000 == 0 else '\r'
                # print(end)
                print(print_str.format(i_episode*worker,
                                       frame-frame_count_last_episode,
                                        frame*worker, 
                                        score, 
                                        np.mean(scores_window)), end=end)
            
    
            i_episode +=1 
            state, _ = env.reset()
            score = 0
            frame_count_last_episode = frame

    return scores, evaluation_scores


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