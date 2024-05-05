# Intro
# The following notebook will contain classes & functions necessary to build DQN agent.  

# Reference for general approach:  
# https://github.com/BY571/DQN-Atari-Agents/blob/master/Agents/Networks/DQN.py#L45  
  
# **CQL code adapted from:**  
# Kumar, A., Zhou, A., Tucker, G. and Levine, S., 2020. Conservative Q-Learning for Offline Reinforcement Learning. Advances in Neural Information Processing Systems [Online], 33. Curran Associates, Inc., pp.1179–1191. Available from: https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html [Accessed 6 October 2023].

# Base Imports
import numpy as np 
from collections import deque, namedtuple
import random
from itertools import islice

# PyTorch Imports
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from Multi_GANs import *

Tensor = torch.cuda.FloatTensor

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        
def get_noise(n_samples, z_dim, device='cpu', truncated=False):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    if truncated:
        noise = torch.nn.init.trunc_normal_(n_samples, z_dim, device=device)
    else:
        noise = torch.randn(n_samples, z_dim, device=device)
    return noise 

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)

# DQN Network for Visual Atari Games
class DQN_Network(nn.Module):
    def __init__(self, state_size, action_size, layer_size, n_step, seed):
        super(DQN_Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        # self.state_dim = len(state_size)
        
        # Create network for Atari Visual Games
        self.cnn_1 = nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4)
        self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        weight_init([self.cnn_1, self.cnn_2, self.cnn_3])

        self.linear_1 = nn.Linear(self.calc_input_layer(), layer_size)
        self.linear_2 = nn.Linear(layer_size, action_size)
        weight_init([self.linear_1, self.linear_2])
        
    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        return x.flatten().shape[0]
        
        
    def forward(self, inp):
        """
        Forward pass of network.
        """
        # Pass Through Convolutional Layers
        x = torch.relu(self.cnn_1(inp))
        x = torch.relu(self.cnn_2(x))
        x = torch.relu(self.cnn_3(x))
        
        # print(x.shape)
        
        # Flatten Output of CNN Layers
        x = x.view(inp.size(0), -1)
        # x = x.reshape((x.shape[0], np.product(x.shape[1:])))
        
        # Pass Through Linear Layers
        x = torch.relu(self.linear_1(x))
        out = self.linear_2(x)
        
        return out
    
# DQN Network for RAM Atari Games
class DQN_Network_RAM(nn.Module):
    def __init__(self, state_size, action_size, layer_size, n_step, seed, dropout_rate=0.5):
        super(DQN_Network_RAM, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        # self.state_dim = len(state_size)
        
        # Dropout is added between each layer, as per Sygnowski and Michalewski (2016)

        self.linear_1 = nn.Linear(state_size, layer_size)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(state_size, layer_size)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.linear_3 = nn.Linear(state_size, layer_size)
        self.dropout_3 = nn.Dropout(dropout_rate)
        self.linear_4 = nn.Linear(state_size, layer_size)
        self.dropout_4 = nn.Dropout(dropout_rate)
        self.linear_5 = nn.Linear(layer_size, action_size)
        weight_init([self.linear_1, self.linear_2, self.linear_3, self.linear_4, self.linear_5])
        
        
    def forward(self, inp):
        """
        Forward pass of network.
        """
        # Pass Through Linear Layers
        x = torch.relu(self.linear_1(inp))
        X = self.dropout_1(x)
        
        x = torch.relu(self.linear_2(x))
        X = self.dropout_2(x)
        
        x = torch.relu(self.linear_3(x))
        X = self.dropout_3(x)
        
        x = torch.relu(self.linear_4(x))
        X = self.dropout_4(x)
        
        out = self.linear_5(x)
        
        return out
    
# DQN Agent for both RAM & Visual Atari Games.
class DQN_Agent_Combined():
    def __init__(self, 
                 state_size, 
                 action_size, 
                 layer_size, 
                 n_step, 
                 seed,
                 batch_size,
                 learning_rate,
                 tau,
                 gamma,
                 update_freq,
                 device,
                 ram=False,
                 image_full_size=False,
                 buffer_size=100_000,
                 gener=False,
                 gan_smoothing_factor=0.0,
                 gan_type='DCGAN',
                 gan_upsample=False,
                 gan_spectral=False,
                 gan_batch_norm=True,
                 gan_learning_rate=0.0001,
                 gan_disc_learning_rate_division=1.0,
                 gan_sample_selection=False,
                 # gan_batch_size=64,
                 gan_train=True,
                 gan_mix=1.0,
                 gan_lambda_gp = 10,
                 combined_buffer=False,
                 cql=False,
                 huber_loss=True
                ):
        """Initialize an Agent object.
        
        Parameters
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            n_step (int): steps to look ahead
            seed (int): random seed
            batch_size (int): size of the training batch
            buffer_size (int): size of the replay memory
            learning_rate (float): learning rate
            tau (float): tau for soft updating the network weights
            gamma (float): discount factor
            update_freq (int): update frequency
            device (str): device that is used for the compute
            
        """
        self.state_size = state_size
        self.action_size = action_size
        self.eta = 0.1
        self.seed = seed # random.seed(seed)
        self.t_seed = seed #torch.manual_seed(seed)
        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_step = n_step
        self.ram = ram
        # Flag to use 84 * 84 or 64 * 64 images
        self.image_full_size = image_full_size
        # Flag to use Huber Loss or MSE
        self.huber_loss = huber_loss
        
        self.update_freq = update_freq
        self.Q_updates = 0
        self.t_step = 0
        
        # GAN Parameters
        self.gener = gener
        self.gan_train = gan_train
        self.gan_smoothing_factor = gan_smoothing_factor
        self.gan_type = gan_type
        self.gan_upsample = gan_upsample
        self.gan_spectral = gan_spectral
        self.gan_batch_norm = gan_batch_norm
        self.gan_learning_rate = gan_learning_rate
        self.gan_disc_learning_rate_division = gan_disc_learning_rate_division
        self.gan_lambda_gp = gan_lambda_gp
        # self.gan_batch_size = gan_batch_size
        self.gan_mix = gan_mix
        self.combined_buffer = combined_buffer
        self.gan_sample_selection = gan_sample_selection
        
        # Offline Modifications
        self.cql = cql
        self.cql_min_q_weight = 10.0
        
        # Create Q Networks
        if ram:
            self.qnetwork_local = DQN_Network_RAM(state_size, action_size, layer_size, n_step, seed).to(device)
            self.qnetwork_target = DQN_Network_RAM(state_size, action_size, layer_size, n_step, seed).to(device)
        else:
            # Create Q Networks
            self.qnetwork_local = DQN_Network(state_size, action_size, layer_size, n_step, seed).to(device)
            self.qnetwork_target = DQN_Network(state_size, action_size, layer_size, n_step, seed).to(device)
        
        self.optim = torch.optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        if self.gener:
            self.memory_buffer = GAN(device=device, 
                                     ram=self.ram, 
                                     verbose=False, 
                                     gan_type=self.gan_type,
                                     action_size=action_size,
                                     smoothing_factor=self.gan_smoothing_factor,
                                     upsample=self.gan_upsample,
                                     spectral=self.gan_spectral,
                                     batch_norm=self.gan_batch_norm,
                                     lr=self.gan_learning_rate,
                                     disc_learning_rate_division=self.gan_disc_learning_rate_division,
                                     combined_buffer=self.combined_buffer,
                                     gan_mix=self.gan_mix,
                                     memory_buffer_size=5_000,
                                     lambda_gp=gan_lambda_gp,
                                     gamma=gamma,
                                     n_step=n_step,
                                     seed=self.seed,
                                     full_size=self.image_full_size,
                                     sample_selection=gan_sample_selection
                                    )
           
            
        else:
            self.memory_buffer = ReplayBuffer(buffer_size, 
                                              # batch_size,
                                              device, 
                                              seed, 
                                              gamma, 
                                              n_step,
                                              ram=self.ram)

    def step(self, state, action, reward, next_state, done, writer=None):
        # Save experience in replay memory
        self.memory_buffer.add(state, action, reward, next_state, done)

        # Learn every update_freq time steps.
        self.t_step = (self.t_step + 1) % self.update_freq

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory_buffer) > self.batch_size:
                experiences = self.memory_buffer.sample(batch_size=self.batch_size)
                    
                # if self.per == False:
                loss = self.learn(experiences)
                # else: 
                #     loss = self.learn_per(experiences)
                self.Q_updates += 1
                if writer: writer.add_scalar("Agent/Q_loss", loss, self.Q_updates)
                # writer.add_scalar("ICM_loss", icm_loss, self.Q_updates)
                
    def act(self, state, epsilon=0.0):
        """Returns actions for given state as per current policy. Acting only every 4 frames.
        
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
            
        """

        # Epsilon-greedy action selection
        if random.random() > epsilon: # select greedy action if random number is higher than epsilon or noisy network is used!
            state = np.array(state)
            state = torch.from_numpy(state).float().to(self.device)
            network_inp = state #[None, :,:,:]

            # Set Network to eval mode
            self.qnetwork_local.eval()
            
            with torch.no_grad():
                
                action_values = self.qnetwork_local(network_inp)
            
            # Set Network back to train mode
            self.qnetwork_local.train()
            
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
            
        # Select Action Randomly    
        else:
            action = random.choices(np.arange(self.action_size), k=1)
            
        return action

        
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.optim.zero_grad()
        states, actions_ohe, rewards, next_states, dones = experiences
        # If visual, take last 3 state frames as the first 3 next state frames
        if not self.ram:
            next_states = torch.cat((states[:,1:,:,:], next_states), axis=1)
        actions = torch.argmax(actions_ohe, dim=1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma**self.n_step * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_a_s = self.qnetwork_local(states)
        # print('Q_expected shape', Q_expected.shape)
        # print('actions', actions.shape)
        Q_expected = Q_a_s.gather(1, actions[:, None])
        
        if self.huber_loss:
            # Compute Huber Loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(Q_expected, Q_targets)
        else:    
            # Compute MSE loss
            loss = nn.functional.mse_loss(Q_expected, Q_targets)
            
        # Check for Offline modification
        if self.cql:
            
            ### New CQL Process
            # Code converted to PyTorch from https://github.com/aviralkumar2907/CQL (Kumar et al., 2020).
            # Compute the dataset expectation
            dataset_expec = torch.mean(Q_expected)
            
            # Compute negative sampling
            negative_sampling = torch.mean(torch.logsumexp(Q_a_s, dim=1))
            
            # Compute the minimum Q-loss
            min_q_loss = negative_sampling - dataset_expec
            
            min_q_loss = min_q_loss * self.cql_min_q_weight
            
            loss = loss + min_q_loss
        
        # Minimize the loss
        self.optim.zero_grad()
        loss.backward()
        # if self.cql:
        #     clip_grad_norm_(self.qnetwork_local.parameters(), 1.)
        if self.huber_loss:
            clip_grad_norm_(self.qnetwork_local.parameters(), 100)
        self.optim.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        return loss.detach().cpu().numpy()
    
    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions.
        Function taken from:
        https://github.com/BY571/CQL/blob/main/CQL-DQN/agent.py
        """
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)
    
        return (logsumexp - q_a).mean()
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            
            target_param.data.copy_(self.tau * local_param.data 
                                    + (1.0 - self.tau) * target_param.data)
            
    
# Static Memory Replay Buffer
# Adapted from https://github.com/BY571/IQN-and-Extensions/blob/master/ReplayBuffers.py
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, device, seed=None, gamma=0.95, n_step=1, ram=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.device = device
        self.memory = deque(maxlen=buffer_size) 
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.iter_ = 0
        self.ram = ram
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer)
            if self.ram:
                e = self.experience(state, action, reward, next_state, done)
            else:
                e = self.experience(state, action, reward, next_state[-1:,:,:], done)
            self.memory.append(e)
        self.iter_ += 1



    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]
        
    
    
    def sample(self, batch_size=0):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def get_history(self, batch_size=0):
        # Slice to get the last batch_size experiences
        experiences = islice(self.memory, 
                             self.buffer_size-batch_size, 
                             self.buffer_size
                            )
        
        experiences = list(experiences)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class GAN:
    def __init__(self, 
                 device, 
                 seed,
                 gan_type='DCGAN', 
                 z_dim=64, 
                 lr=0.0001, # 0.0002
                 beta_1 = 0.5, 
                 beta_2 = 0.999, 
                 lambda_gp = 10, 
                 crit_repeats = 4, # 4
                 batch_size = 64,
                 ram=False,
                 verbose=False,
                 action_size=2,
                 smoothing_factor=0.0,
                 upsample=False,
                 spectral=False,
                 batch_norm=True,
                 disc_learning_rate_division=1.0,
                 full_size=False,
                 combined_buffer=False,
                 gan_mix=1.0,
                 memory_buffer_size=5_000,
                 gamma=0.95,
                 n_step=1,
                 sample_selection=False
                ):
        # Store Attributes
        self.gan_type = gan_type
        self.device = device
        # Input Noise Dimensions for Generator
        self.z_dim = z_dim
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lambda_gp = lambda_gp
        self.action_size = action_size
        self.verbose = verbose
        self.smoothing_factor = smoothing_factor # Consider changing to 0.1
        self.upsample = upsample
        self.spectral = spectral
        self.batch_norm = batch_norm
        self.disc_learning_rate_division = disc_learning_rate_division
        self.full_size = full_size
        self.seed = seed
        
        # These values determine whether a memory buffer should be used alongside the GAN and to what extent:
        self.combined_buffer = combined_buffer
        self.gan_mix = gan_mix
        
        # Memory Buffer Attributes
        self.gamma = gamma
        self.n_step = n_step
        self.memory_buffer_size = memory_buffer_size
        
        # Internal Experience Type
        # self.experience = namedtuple("experience", field_names=["observation", "action", "reward", "done"])
        # Output Experience Type
        self.output_experience = namedtuple("experience", field_names=["state", "action", "reward", "next_state", "done"])
        # Input buffer for storing experiences before training
        self.input_buffer = deque(maxlen=batch_size)  

        # How many times to train the critic per for each training cycle of the generator
        self.crit_repeats = crit_repeats if gan_type == 'WGAN' else 1
        self.batch_size = batch_size
        self.ram = ram
        
        # Flag to set GAN to sample images with lower Disc score
        self.sample_selection = sample_selection
        
        assert gan_type in ['WGAN', 'DCGAN', 'SNGAN'], 'Unknown GAN type'
        
        # Create Generator & Discriminator
        self.gen = MultiGenerator(z_dim, ram=self.ram, action_size=action_size, upsample=self.upsample, full_size=self.full_size).to(device)
        if gan_type == 'WGAN':
            self.disc = MultiCrit(ram=self.ram, action_size=action_size, spectral=self.spectral, full_size=self.full_size).to(device)
        else:
            self.disc = MultiDisc(ram=self.ram, action_size=action_size, spectral=self.spectral, full_size=self.full_size).to(device)
            
        if self.combined_buffer:
            self.memory_buffer = ReplayBuffer(buffer_size=self.memory_buffer_size, 
                                              device=self.device, 
                                              seed=self.seed, 
                                              gamma=self.gamma, 
                                              n_step=self.n_step,
                                              ram=self.ram)
        
        # Loss Criterion
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCELoss()
        

        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        disc_lr = self.lr / disc_learning_rate_division
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=disc_lr, betas=(self.beta_1, self.beta_2))

        # Initialise Weights
        self.gen = self.gen.apply(weights_init)
        self.disc = self.disc.apply(weights_init)
        
        if verbose: print('GAN initialised.')
        
        # Create empty list of loss values
        self.mean_discriminator_loss = list()
        self.mean_generator_loss = list()
        
        # Create empty list of loss values
        self.mean_discriminator_accuracy = list()
        self.mean_generator_accuracy = list()
        
        # Create empty list for experiences
        # self.experiences = list()
        
        self.train_idx = 0
        self.current_step = 0
        self.gen_train_idx = 0
        self.disc_train_idx = 0
    
    def add(self, state, action, reward, next_state, done):
        # observations = np.concatenate([state, next_state])
        if self.ram:
            e = self.output_experience(state, action, reward, next_state, done)
        else:
            e = self.output_experience(state, action, reward, next_state[-1:,:,:], done)
        self.input_buffer.append(e)
        
        if self.combined_buffer:
                self.memory_buffer.add(state, action, reward, next_state, done)
        
        if len(self.input_buffer) == self.batch_size:
            # Returns Tensors
            state, action, reward, next_state, done = self.get_from_input_buffer()
            
            self.train_loop(state, action, reward, next_state, done, save_images=False)
            
            #Remove all elements from input buffer
            self.input_buffer.clear()
            
    def get_from_input_buffer(self):
        """Randomly sample a batch of experiences from memory."""
        # experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in self.input_buffer])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in self.input_buffer])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in self.input_buffer])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in self.input_buffer])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in self.input_buffer]).astype(np.uint8)).float().to(self.device)
            
        return states, actions, rewards, next_states, dones
        
    def save_benchmark(self, noise_input, file_path):
        # Save Benchmark Dataframe
        with torch.no_grad():
            fake_benchmark = self.gen(noise_input).detach()

            # Save Results
            np.save(file_path, 
                    fake_benchmark.cpu().numpy()
                   )

            del fake_benchmark
            
    def train_disc(self, real_observations, real_actions, real_rewards, real_dones, cur_batch_size, save_images=False, image_path=None):
        
        ## Update discriminator ##
        self.disc_opt.zero_grad()
        fake_noise = get_noise(cur_batch_size, self.z_dim, device=self.device)
        fake_observations, fake_actions, fake_rewards, fake_dones = self.gen(fake_noise)
        
        # run on real values first
        disc_real_pred = self.disc(real_observations, real_actions, real_rewards, real_dones)
        disc_fake_pred = self.disc(fake_observations.detach(), fake_actions.detach(), fake_rewards.detach(), fake_dones.detach())
        
        if self.gan_type == "DCGAN":
            disc_fake_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = self.criterion(disc_real_pred, torch.ones_like(disc_real_pred)-self.smoothing_factor)
            
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            
        elif self.gan_type == "WGAN":
            
            # print(real_observations.shape, 
            #       real_actions.shape, 
            #       real_rewards.shape, 
            #       real_dones.shape)
        
            gradient_penalty = compute_multi_gradient_penalty(self.disc, 
                                                              real_observations, 
                                                              real_actions, 
                                                              real_rewards, 
                                                              real_dones, 
                                                              fake_observations, 
                                                              fake_actions, 
                                                              fake_rewards, 
                                                              fake_dones,
                                                              device=self.device,
                                                              ram=self.ram
                                                             )
            

            disc_loss = (-torch.mean(disc_real_pred) 
                         + torch.mean(disc_fake_pred) 
                         + self.lambda_gp * gradient_penalty
                        )

        # Keep track of the average discriminator loss
        self.mean_discriminator_loss.append(disc_loss.detach().cpu().numpy())
        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        self.disc_opt.step()
        
        if save_images and image_path:
            # Save Images
            save_tensor_images(fake, image_path, f'fake_images_step_{self.current_step}')
            
        self.disc_train_idx += 1
        
        # Clean Up
        if self.gan_type == "DCGAN": del disc_fake_loss, disc_real_loss
        elif self.gan_type == "WGAN": del gradient_penalty
        del disc_loss, disc_real_pred, disc_fake_pred
        del fake_observations, fake_actions, fake_rewards, fake_dones, fake_noise
        del real_observations, real_actions, real_rewards, real_dones
        
        
            
    def train_gen(self, real_observations, real_actions, real_rewards, real_dones, batch_size):
        ## Update generator ##
        self.gen_opt.zero_grad()
        fake_noise = get_noise(batch_size, self.z_dim, device=self.device)
        # print('Fake Noise Shape', fake_noise.shape)
        fake_observations, fake_actions, fake_rewards, fake_dones = self.gen(fake_noise)
        
        if self.gan_type == "DCGAN":
            disc_fake_pred = self.disc(fake_observations, fake_actions, fake_rewards, fake_dones)
            gen_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)-self.smoothing_factor)

        elif self.gan_type == "WGAN":
            disc_fake_pred = self.disc(fake_observations, fake_actions, fake_rewards, fake_dones)
            gen_loss = -torch.mean(disc_fake_pred)
            

        gen_loss.backward()
        self.gen_opt.step()

        # Keep track of the average generator loss
        self.mean_generator_loss.append(gen_loss.detach().cpu().numpy())
        self.gen_train_idx += 1
        
        # Clean Up
        del disc_fake_pred, gen_loss, fake_observations, fake_actions, fake_rewards, fake_dones, fake_noise
        
    def train_loop(self, real_states, real_actions, real_rewards, real_next_states, real_dones, noise_input=None, save_images=False):
        
        real_observations = torch.cat((real_states, real_next_states), axis=1)
        cur_batch_size = len(real_observations)
        # real = torch.tensor(real)
        # real = real.to(self.device)
        
        # if self.cur_step % self.save_image_step == 0 and self.cur_step > 0:
        #     save_images=True
        # else:
        #     save_images=False
        
        
        self.train_disc(real_observations, real_actions, real_rewards, real_dones, cur_batch_size, save_images=save_images)
            
        
        if self.train_idx % self.crit_repeats == 0:
            self.train_idx = 0
            self.train_gen(real_observations, real_actions, real_rewards, real_dones, cur_batch_size)
            
        if save_images:
            step, gen_loss, disc_loss = self.get_loss()
            
            if self.verbose:
                print(f"Step {self.current_step:_}: Generator loss: {self.gen_loss:.2f}, discriminator loss: {self.disc_loss:.2f}")
            # If static noise input available, save down a benchmark file of fakes.
            if noise_input:
                    save_benchmark(noise_input, self.file_path)
                    
        if self.verbose and self.current_step % 100 == 0:
            step, gen_loss, disc_loss = self.get_loss()
            print(f"Step {step:_}: Generator loss: {gen_loss:.2f}, discriminator loss: {disc_loss:.2f}")
            
        self.train_idx += 1
        self.current_step += 1
            
        
        
    def get_loss(self):
        # Calculate average loss value
        mean_discriminator_loss = sum(self.mean_discriminator_loss) / len(self.mean_discriminator_loss)
        mean_generator_loss = sum(self.mean_generator_loss) / len(self.mean_generator_loss)
        
        # Reset Loss List
        self.mean_discriminator_loss = list()
        self.mean_generator_loss = list()
        
        return self.current_step, mean_generator_loss, mean_discriminator_loss
    
    @staticmethod
    def top_percentage_indices(input_tensor, proportion=0.5):
        """
        Helper function to return indices of the top half of values in input_tensor.
        """
       
        denominator = (proportion).as_integer_ratio()[1]
        
         # Calculate the number of elements in the top percetnage
        num_elements = input_tensor.numel()
        num_top_proportion = num_elements // denominator

        # Flatten the input tensor and find the indices of the top half
        flattened_indices = torch.argsort(input_tensor.view(-1), descending=True)
        top_proportion_indices = flattened_indices[:num_top_proportion]

        # Apply indices to tensor
        # top_half_tensor = input_tensor.gather(0, top_half_indices)

        return top_proportion_indices

    
    def sample(self, batch_size=64, fake_noise=None):
        if self.combined_buffer:
            # Determine size of respective buffers
            gan_batch_size = int(batch_size * self.gan_mix)
            memory_batch_size = batch_size - gan_batch_size
        else:
            gan_batch_size = batch_size
            
        # If using Discriminator to filter samples, double initial batch size    
        if self.sample_selection:
            gan_batch_size = gan_batch_size * 2
            
        
        
        with torch.no_grad():
            # If no noise given, generate
            if not fake_noise:
                fake_noise = get_noise(gan_batch_size, self.z_dim, device=self.device)
            observations, actions, rewards, dones = self.gen(fake_noise)
            
            if self.sample_selection:
                disc_scores = self.disc(observations, actions, rewards, dones)
                top_percentage_indices = self.top_proportion_indices(disc_scores, proportion=0.5)
                
                # Filter experiences by indices
                observations = observations[top_percentage_indices,:]
                actions = actions[top_percentage_indices,:]
                rewards = rewards[top_percentage_indices,:]
                dones = dones[top_percentage_indices,:]
                
                # observations = observations.gather(0, top_half_indices)
                # actions = actions.gather(0, top_half_indices)
                # rewards = rewards.gather(0, top_half_indices)
                # observations = observations.gather(0, top_half_indices)
            
            if self.combined_buffer:
                real_states, real_actions, real_rewards, real_next_states, real_dones = self.memory_buffer.sample(memory_batch_size)
                
                # print(real_states.shape, real_next_states.shape)
                real_observations = torch.cat((real_states, real_next_states), axis=1)
                # print(real_observations.shape, observations.shape)
                observations = torch.cat((observations, real_observations), axis=0)
                actions = torch.cat((actions, real_actions), axis=0)
                rewards = torch.cat((rewards, real_rewards), axis=0)
                dones = torch.cat((dones, real_dones), axis=0)
                
            
            if self.ram:
                output = self.output_experience(observations[:,:128], actions, rewards, observations[:,128:], dones)
                
            else:
                output = self.output_experience(observations[:,0:4,:,:], actions, rewards, observations[:,-1:,:,:], dones)
            
        return output
    
    def __len__(self):
        return self.current_step
