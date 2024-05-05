# Imports
import torch
from torch import nn

import numpy as np

from torch.autograd import Variable

# Shared Functions 
# Adapted from https://www.coursera.org/specializations/generative-adversarial-networks-gans
# Generative Adversarial Networks (GANs) Specialization [3 courses] (DeepLearning.AI) | Coursera [Online], n.d. Available from: https://www.coursera.org/specializations/generative-adversarial-networks-gans [Accessed 4 May 2024].

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples (int): the number of samples to generate.
      z_dim (int): the dimension of the noise vector.
    '''
    return torch.randn(n_samples, z_dim, device=device)

def weights_init(m):
    """
    Initialise Conv 2d, Transpose Conv 2d and Batch Norm 2d values.
    
    Args:
        m (Pytorch Model): Model to initialise weights for.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)
        
### Multi-Output Generator
class MultiGenerator(nn.Module):
    '''
    Generator Class
    Args:
        z_dim: the dimension of the noise vector.
        im_chan: When training on visula states, the number of channels in the images.
        hidden_dim: the inner layer dimension.
        action_size (int): The number of possible actions.
        ram (bool): Flag to indicate whether the network create RAM states or Visual observations.
        upsample (bool): Flag to indicate whether to upsample or use deconvolution (a.k.a. Transpose Conv).
        full_size (bool): When creating visual states, flag indicates 84 * 84 dimensions instead of 64 * 64.
    '''
    def __init__(self, z_dim=64, im_chan=5, hidden_dim=64, action_size=2, ram=False, upsample=False, full_size=False):
        # env.action_space.n
        super(MultiGenerator, self).__init__()
        self.z_dim = z_dim
        self.ram = ram
        # Flag to determine whether to use ConvTranspose or Upsample (Upsample can mitigate checkboard artefacts)
        self.upsample = upsample
        
        # Flag to determine whether to build 84*84 or 64*64 images
        self.full_size = full_size
        
        if not self.ram:
            # Build the neural network
            gen_stem_modules = [self.make_gen_block(z_dim, hidden_dim * 8, kernel_size=4, stride=1),
                                self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
                                self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
                               ]
            if upsample:
                gen_stem_modules.append(self.make_gen_block(hidden_dim * 2, hidden_dim * 2, kernel_size=4, stride=2, padding=1))
                
            gen_stem_output_layers = 32768


            self.gen_stem = nn.Sequential(*gen_stem_modules)
            # Observation
            observation_modules = [self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)]
            
            if self.full_size:
                observation_modules.append(self.make_gen_block(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1, final_layer=False))
                # Add Layer to transform from 64*64 -> 84*84
                # Calculator used: https://asiltureli.github.io/Convolution-Layer-Calculator/
                observation_modules.append(self.make_gen_block(hidden_dim, im_chan, kernel_size=4, stride=2, padding=23, final_layer=True))
                
            else:
                observation_modules.append(self.make_gen_block(hidden_dim, im_chan, kernel_size=4, stride=2, padding=1, final_layer=True))
                
                
            self.observation_layers = nn.Sequential(*observation_modules)
        else:
            # RAM Observation
            self.gen_stem = nn.Sequential(
            nn.Linear(z_dim, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(256, out_features=512),
            nn.LeakyReLU()
            )
            gen_stem_output_layers = 512
            
            self.observation_layers = nn.Sequential(
            nn.Linear(512, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(512, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(512, out_features=256),
            nn.Tanh()
            )

    
        # Combined tabular Layers
        self.combined_tabular_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gen_stem_output_layers, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        
        # Action
        self.action_layers = nn.Sequential(
                nn.Linear(128, out_features=128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, out_features=action_size),
                # nn.Sigmoid()
                nn.Softmax(dim=1) # This may slow things down considerably
            )
        
        
        # Reward
        # Assuming rewards are clipped between -1 and 1
        self.reward_layers = nn.Sequential(
                nn.Linear(128, out_features=128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, out_features=1),
                nn.Tanh(),
            )
        
        # self.reward_head = self.reward_layers(self.gen_stem)
        
        # Done / Terminal
        self.done_layers = nn.Sequential(
                nn.Linear(128, out_features=128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, out_features=1),
                nn.Sigmoid(),
            )
        
        # self.done_head = self.done_layers(self.gen_stem)
               

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, 
                       padding=0,
                       final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a single generator block.
        Parameters:
            input_channels (int): how many channels the input feature representation has.
            output_channels (int): how many channels the output feature representation should have.
            kernel_size (int): the size of each convolutional filter, equivalent to (kernel_size, kernel_size).
            stride (int): the stride of the convolution.
            padding (int): Padding to apply to convolutions to increase size of output later.
            final_layer (bool): Indicates if it is the final layer.
        '''
        if self.upsample:
            if not final_layer:
                return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(input_channels, output_channels, 3, 1, 1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
                )
            elif self.full_size:
                return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=21),
                nn.Tanh()
                )
                
            else:
                return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
                )
        else:
            if not final_layer:
                return nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                    # Original code used Tanh, however our games have outputs between 0 and 1
                    # nn.Sigmoid(),
                    nn.Tanh(),
                )
        
    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        if self.ram:
            x = noise
        else:
            x = self.unsqueeze_noise(noise)
        # print('noise shape:', noise.shape)
        # gen_stem = x
        # for layer in self.gen_stem:
        #     gen_stem = layer(gen_stem)
        #     print(gen_stem.size())
        gen_stem = self.gen_stem(x)
        tabular_stem = self.combined_tabular_layers(gen_stem)
        # print('gen_stem shape:', gen_stem.shape)
        # print('tabular_stem shape:', tabular_stem.shape)
        
        
        
        observation_head = self.observation_layers(gen_stem)
        # print('observation_head shape:', observation_head.shape)
        action_head = self.action_layers(tabular_stem)
        reward_head = self.reward_layers(tabular_stem)
        done_head = self.done_layers(tabular_stem)
        
        return observation_head, action_head, reward_head, done_head
    
    
### Multi-Input Discriminator
class MultiDisc(nn.Module):
    '''
    Discriminator Class for DC-GAN
    Args:
        im_chan: When training on visula states, the number of channels in the images.
        hidden_dim: the inner layer dimension.
        action_size (int): The number of possible actions.
        ram (bool): Flag to indicate whether the network create RAM states or Visual observations.
        spectral (bool): Flag to indicate whether to use Spectral Normalisation.
        batch_norm (bool): Flag to indicate whether to use Batch Norm.
        full_size (bool): When creating visual states, flag indicates 84 * 84 dimensions instead of 64 * 64.
    '''
    def __init__(self, im_chan=5, hidden_dim=64, action_size=1, ram=False, spectral=False, batch_norm=True, full_size=False):
        super(MultiDisc, self).__init__()
        self.ram = ram
        self.spectral = spectral
        self.batch_norm = batch_norm
        
        # Flag to determine whether to Input 84*84 or 64*64 images
        self.full_size = full_size
        
        # Visual Disc
        if not self.ram:
            # Add First Layer
            obs_critic_modules = [self.make_crit_block(im_chan, hidden_dim, padding=1)]
            
            # Add optional Second Layer if image size == 84
            if self.full_size:
                obs_critic_modules.append(self.make_crit_block(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=12))
                
            # Add Final Layers
            obs_critic_modules.append(self.make_crit_block(hidden_dim, hidden_dim * 2, padding=1))
            obs_critic_modules.append(self.make_crit_block(hidden_dim * 2, hidden_dim * 4, padding=1))
            obs_critic_modules.append(self.make_crit_block(hidden_dim * 4, hidden_dim * 4))
            obs_critic_modules.append(nn.Flatten())
                
                
            self.observation_critic = nn.Sequential(*obs_critic_modules)
            # obs, act, rew, term/dones
            observation_shape = 9 * hidden_dim * 4
            
        # RAM Disc
        else:
            self.observation_critic = nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                # nn.Linear(128, 128),
                # nn.LeakyReLU(0.2),
                # nn.Dropout(0.3),
                # nn.Linear(128, 128),
                # nn.LeakyReLU(0.2),
                # nn.Dropout(0.3),
            )
            observation_shape = 128
        
       
        # Consider having more layers shared with features and observations.
        self.feature_critic = nn.Sequential(nn.Linear(action_size+2, 128),
                                            nn.LeakyReLU(0.2),
                                            nn.Dropout(0.3),
                                            # nn.Linear(128, 128),
                                            # nn.LeakyReLU(0.2),
                                            # nn.Dropout(0.3),
                                            nn.Linear(128, 64),
                                            nn.LeakyReLU(0.2),
                                            nn.Dropout(0.3),
                                           )

        
        self.combined_head = nn.Sequential(
                                           nn.Linear(observation_shape+64, out_features=128),
                                           nn.LeakyReLU(0.2),
                                           nn.Dropout(0.3),
                                           # nn.Linear(128, out_features=128),
                                           # nn.LeakyReLU(0.2),
                                           # nn.Dropout(0.3),
                                           nn.Linear(128, out_features=32),
                                           nn.LeakyReLU(0.2),
                                           nn.Dropout(0.3), # Recommended value is below 0.5
                                           nn.Linear(32, out_features=1),
                                           nn.Sigmoid()
                                          )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=0, 
                        final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a critic block.
        Parameters:
            input_channels (int): how many channels the input feature representation has.
            output_channels (int): how many channels the output feature representation should have.
            kernel_size (int): the size of each convolutional filter, equivalent to (kernel_size, kernel_size).
            stride (int): the stride of the convolution.
            final_layer (bool): true if it is the final layer and false otherwise.
        '''
        
                # Build the neural network
        crit_block_modules = list()
        
        if self.spectral:
            crit_block_modules.append(nn.utils.spectral_norm(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)))
        else:
            crit_block_modules.append(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding))
            
        if final_layer:
            crit_block_modules.append(nn.Sigmoid(output_channels))
        else:
            if self.batch_norm:
                crit_block_modules.append(nn.BatchNorm2d(output_channels))
                
            crit_block_modules.append(nn.LeakyReLU(0.2, inplace=True))
            
        return nn.Sequential(*crit_block_modules)
        
    
    def forward(self, observation, action, reward, done):
        '''
        Function for completing a forward pass of the critic: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        observation_inp = observation
        feature_inp = torch.cat((action, reward, done), dim=1)
        
        observation_inp = self.observation_critic(observation)
        feature_inp = self.feature_critic(feature_inp)
        
        observation_inp = observation_inp.view(observation_inp.size(0), -1)
        feature_inp = feature_inp.view(feature_inp.size(0), -1)
        
        x = torch.cat((observation_inp, feature_inp), dim=1)
        
        crit_pred = self.combined_head(x)
        
        return crit_pred
    
### Multi-Input Critic    
class MultiCrit(nn.Module):
    '''
    Discriminator Class for W-GAN with GP
    Args:
        im_chan: When training on visula states, the number of channels in the images.
        hidden_dim: the inner layer dimension.
        action_size (int): The number of possible actions.
        ram (bool): Flag to indicate whether the network create RAM states or Visual observations.
        spectral (bool): Flag to indicate whether to use Spectral Normalisation.
        batch_norm (bool): Flag to indicate whether to use Batch Norm.
        full_size (bool): When creating visual states, flag indicates 84 * 84 dimensions instead of 64 * 64.
    '''
    def __init__(self, im_chan=5, hidden_dim=64, action_size=1, ram=False, spectral=False, batch_norm=True, full_size=False):
        super(MultiCrit, self).__init__()
        self.ram = ram
        self.spectral = spectral
        self.batch_norm = batch_norm
        
        # Flag to determine whether to Input 84*84 or 64*64 images
        self.full_size = full_size
        
        # Visual Disc
        if not self.ram:
            # Add First Layer
            obs_critic_modules = [self.make_crit_block(im_chan, hidden_dim, padding=1)]
            
            # Add optional Second Layer if image size == 84
            if self.full_size:
                obs_critic_modules.append(self.make_crit_block(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=12))
                
            # Add Final Layers
            obs_critic_modules.append(self.make_crit_block(hidden_dim, hidden_dim * 2, padding=1))
            obs_critic_modules.append(self.make_crit_block(hidden_dim * 2, hidden_dim * 4, padding=1))
            obs_critic_modules.append(self.make_crit_block(hidden_dim * 4, hidden_dim * 4))
            obs_critic_modules.append(nn.Flatten())
                
                
            self.observation_critic = nn.Sequential(*obs_critic_modules)
            # obs, act, rew, term/dones
            observation_shape = 9 * hidden_dim * 4
            
        # RAM Disc
        else:
            self.observation_critic = nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                # nn.Linear(128, 128),
                # nn.LeakyReLU(0.2),
                # nn.Dropout(0.3),
                # nn.Linear(128, 128),
                # nn.LeakyReLU(0.2),
                # nn.Dropout(0.3),
            )
            observation_shape = 128
        
       
        # Consider having more layers shared with features and observations.
        self.feature_critic = nn.Sequential(nn.Linear(action_size+2, 128),
                                            nn.LeakyReLU(0.2),
                                            nn.Dropout(0.3),
                                            # nn.Linear(128, 128),
                                            # nn.LeakyReLU(0.2),
                                            # nn.Dropout(0.3),
                                            nn.Linear(128, 64),
                                            nn.LeakyReLU(0.2),
                                            nn.Dropout(0.3),
                                           )

        
        self.combined_head = nn.Sequential(
                                           nn.Linear(observation_shape+64, out_features=128),
                                           nn.LeakyReLU(0.2),
                                           nn.Dropout(0.3),
                                           # nn.Linear(128, out_features=128),
                                           # nn.LeakyReLU(0.2),
                                           # nn.Dropout(0.3),
                                           nn.Linear(128, out_features=32),
                                           nn.LeakyReLU(0.2),
                                           nn.Dropout(0.3), # Recommended value is below 0.5
                                           nn.Linear(32, out_features=1),
                                           # nn.Sigmoid()
                                          )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=0, 
                        final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a critic block.
        Parameters:
            input_channels (int): how many channels the input feature representation has.
            output_channels (int): how many channels the output feature representation should have.
            kernel_size (int): the size of each convolutional filter, equivalent to (kernel_size, kernel_size).
            stride (int): the stride of the convolution.
            final_layer (bool): true if it is the final layer and false otherwise.
        '''
        
                # Build the neural network
        crit_block_modules = list()
        
        if self.spectral:
            crit_block_modules.append(nn.utils.spectral_norm(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)))
        else:
            crit_block_modules.append(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding))
            
        if final_layer:
            crit_block_modules.append(nn.Sigmoid(output_channels))
        else:
            if self.batch_norm:
                crit_block_modules.append(nn.BatchNorm2d(output_channels))
                
            crit_block_modules.append(nn.LeakyReLU(0.2, inplace=True))
            
        return nn.Sequential(*crit_block_modules)
        
    
    def forward(self, observation, action, reward, done):
        '''
        Function for completing a forward pass of the critic: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        observation_inp = observation
        feature_inp = torch.cat((action, reward, done), dim=1)
        
        observation_inp = self.observation_critic(observation)
        feature_inp = self.feature_critic(feature_inp)
        
        observation_inp = observation_inp.view(observation_inp.size(0), -1)
        feature_inp = feature_inp.view(feature_inp.size(0), -1)
        
        x = torch.cat((observation_inp, feature_inp), dim=1)
        
        crit_pred = self.combined_head(x)
        
        return crit_pred

def compute_multi_gradient_penalty(disc, 
                                   real_observations, 
                                   real_actions, 
                                   real_rewards, 
                                   real_dones, 
                                   fake_observations, 
                                   fake_actions, 
                                   fake_rewards, 
                                   fake_dones,
                                   device,
                                   ram=True
                                  ):
    """Calculates the gradient penalty loss for WGAN GP"""
    Tensor = torch.cuda.FloatTensor
    # torch.tensor(data, dtype=*, device='cuda')
    
    batch_size = real_observations.size(0)
    
    # Random weight term for interpolation between real and fake samples
    if ram:
        alpha = torch.tensor(np.random.random((batch_size, 1)), dtype=torch.float, device=device)
        alpha_2d = alpha
    else:
        alpha = torch.tensor(np.random.random((batch_size, 1, 1, 1)), dtype=torch.double, device=device)
        alpha_2d = alpha.view(batch_size, 1)
    
    interpolated_observations = (alpha * real_observations + ((1 - alpha) * fake_observations)).requires_grad_(True).float()
    interpolated_actions = (alpha_2d * real_actions + ((1 - alpha_2d) * fake_actions)).requires_grad_(True).float()
    interpolated_rewards = (alpha_2d * real_rewards + ((1 - alpha_2d) * fake_rewards)).requires_grad_(True).float()
    interpolated_dones = (alpha_2d * real_dones + ((1 - alpha_2d) * fake_dones)).requires_grad_(True).float()
    
    disc_interpolates = disc(interpolated_observations, interpolated_actions, interpolated_rewards, interpolated_dones)
    fake = Variable(Tensor(real_observations.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=(interpolated_observations, interpolated_actions, interpolated_rewards, interpolated_dones),
        # grad_outputs=fake,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # print(gradients.shape)
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
