# Code Adapted from Lapan, M., 2020. Deep Reinforcement Learning Hands-On - Second Edition [Online]. Packt Publishing. Available from: https://learning.oreilly.com/library/view/deep-reinforcement-learning/9781838826994/ [Accessed 5 September 2023].

#### Imports
import random
import collections
import numpy as np

import torch
import gym.spaces
import gym
import cv2

## Set Seed

def set_seed(seed: int):
    """Set seed across random, numpy and pytorch libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    return seed

## Required Env Functions

class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(float(reward))

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

class MaxAndSkipEnv(gym.Wrapper):
    """
    Certain games have sprites that flicker in and out, such as the bullets in Space Invaders.
    This process takes the max of every two frames to avoid this happening.
    
    """
    def __init__(self, env=None, skip=4, max_buffer_len=2):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=max_buffer_len)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info, details = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info, details
    
    def reset(self):
        self._obs_buffer.clear()
        obs, details = self.env.reset()
        self._obs_buffer.append(obs)
        return obs, details

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None, image_size=84):
        super(ProcessFrame84, self).__init__(env)
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(image_size, image_size, 1), dtype=np.uint8)
    def observation(self, obs):
        return ProcessFrame84.process(obs, self.image_size)
    
    @staticmethod
    def process(frame, image_size):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        # img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + \
              # img[:, :, 2] * 0.114
        # This convert to greyscale approach is from stable baselines
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        
        if image_size != 84:
            x_t = cv2.resize(x_t, (image_size, image_size), interpolation=cv2.INTER_AREA)
            
        x_t = np.reshape(x_t, [image_size, image_size, 1])
        return x_t.astype(np.uint8)

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space 
        self.observation_space = gym.spaces.Box(
        old_space.low.repeat(n_steps, axis=0),
        old_space.high.repeat(n_steps, axis=0), dtype=dtype)
    
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        obs, details = self.env.reset()
        return self.observation(obs), details
    
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        # Adjust values from 0 -> 255 to -1 -> 1
        adjusted_array = np.array(obs).astype(np.float32)
        adjusted_array = (adjusted_array / (255.0/2)) - 1.0
        return adjusted_array
    
class EpisodicLifeEnv(gym.Wrapper):
    """
    Taken from stable-baslines3: https://stable-baselines3.readthedocs.io/en/v2.0.0/_modules/stable_baselines3/common/atari_wrappers.html
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info


    def reset(self, **kwargs):
            """
            Calls the Gym environment reset, only when lives are exhausted.
            This way all states are still reachable even though lives are episodic,
            and the learner need not know about any of this behind-the-scenes.

            :param kwargs: Extra keywords passed to env.reset() call
            :return: the first observation of the environment
            """
            if self.was_real_done:
                obs, info = self.env.reset(**kwargs)
            else:
                # no-op step to advance from terminal/lost life state
                obs, _, terminated, truncated, info = self.env.step(0)

                # The no-op step can lead to a game over, so we need to check it again
                # to see if we should reset the environment and avoid the
                # monitor.py `RuntimeError: Tried to step environment that needs reset`
                if terminated or truncated:
                    obs, info = self.env.reset(**kwargs)
            self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
            return obs, info



def make_env(env_name, ram=False, image_size=64, seed=None, repeat_action_probability=0):
    env = gym.make(env_name, repeat_action_probability=repeat_action_probability)
    if seed:
        env.seed(seed)
    # Max every 2 frames avoid missing flashing sprites unless using RAM
    max_buffer_len = 1 if ram else 2
    env = MaxAndSkipEnv(env, max_buffer_len=max_buffer_len)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # Only add the following processing if not using RAM states
    if not ram:
        env = ProcessFrame84(env, image_size=image_size)
        env = ImageToPyTorch(env)
        # Stack Observation Frames
        # env = FrameStack(env, 4)
        env = BufferWrapper(env, 4)
    env = ClipRewardEnv(env)
    # Convert from 0 -> 255 to 0 -> 1
    env = ScaledFloatFrame(env)
    return env

## Environments

def make_eval_env(env_name, ram=False, image_size=64, seed=None, render=False, repeat_action_probability=0):
    """
    Eval environments do not contain reward clipping, and can be used to render videos of the game being played.

    """
    env = gym.make(env_name, repeat_action_probability=repeat_action_probability)
    if seed:
        env.seed(seed)
    if render or ram:
        max_buffer_len = 1
    else:
        max_buffer_len = 2
    # if not render: max_buffer_len = 1 if ram else 2
    if not render:
        env = MaxAndSkipEnv(env, max_buffer_len=max_buffer_len)
    # env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if not ram and not render:
        env = ProcessFrame84(env, image_size=image_size)
        env = ImageToPyTorch(env)
        # Stack Observation Frames
        # env = FrameStack(env, 4)
        env = BufferWrapper(env, 4)
    if not render:
        env = ScaledFloatFrame(env)
    return env

def format_eval_state(eval_state, image_size=64):
    """
    Takes output from env.step and re-formats it for an agent to consume.
    """
    agent_state = eval_state
    eval_size = eval_state[0,:,:].shape

    # Convert to Greyscale
    agent_state = cv2.cvtColor(agent_state, cv2.COLOR_RGB2GRAY)

    # Downsize Image
    resized_screen = cv2.resize(agent_state, (84, 110), interpolation=cv2.INTER_AREA)
    agent_state = resized_screen[18:102, :]

    if image_size != 84:
        agent_state = cv2.resize(agent_state, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # Re-Format
    agent_state = np.reshape(agent_state, [image_size, image_size, 1])
    agent_state = agent_state.astype(np.uint8)

    # Convert to Pytorch
    agent_state = np.moveaxis(agent_state, 2, 0)

    # Scale to 0 -> 1
    agent_state = np.array(agent_state).astype(np.float32) / 255.0
    return agent_state

def format_eval_reward(eval_reward):
    agent_reward = np.sign(float(reward))
    return agent_reward
                      