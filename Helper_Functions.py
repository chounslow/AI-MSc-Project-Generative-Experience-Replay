import os

import matplotlib.pyplot as plt

from torch import nn

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
    
def save_tensor_images(image_tensor, folder, filename, num_images=25):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    # Convert from -1 -> +1 to 0 -> +1
    image_tensor = (image_tensor + 1) / 2
    # Convert to cpu
    image_unflat = image_tensor.detach().cpu()
    
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    image_grid = image_grid.permute(1, 2, 0)
    image_grid = image_grid.squeeze()
    try:
        plt.imshow(image_grid)
        plt.axis('off')
        plt.savefig(f'./{folder}/{filename}.png', bbox_inches='tight', pad_inches = 0)
    finally:
        plt.close()
        
def create_directories(gan_type, game, subdirectory=None, folder_name='Figures'):
    # Create directory for images if does not exist
    if not os.path.exists(f'{folder_name}/{game}/'):
        os.mkdir(f'{folder_name}/{game}/')
    if not os.path.exists(f'{folder_name}/{game}/{gan_type}'):
        os.mkdir(f'{folder_name}/{game}/{gan_type}')
    if subdirectory and not os.path.exists(f'{folder_name}/{game}/{gan_type}/{subdirectory}'):
        os.mkdir(f'{folder_name}/{game}/{gan_type}/{subdirectory}')
        
def write_parameters(path,
                     game,
                     gan_type,
                     z_dim,
                     batch_size,
                     data_size,
                     n_epochs,
                     c_lambda,
                     crit_repeats,
                     lr,
                     beta_1,
                     beta_2,
                     lambda_gp
                    ):
    """
    Writes text file with all parameters for a given set up.
    Args:
        path (string): subdirectory in current path to place new text file.
    Returns:
        None
    """
    with open(f"{path}/model_parameters.txt","w") as setupfile:
        
        params = {'game':game, 
                  'gan_type':gan_type, 
                  'z_dim':z_dim, 
                  'batch_size':batch_size, 
                  'data_size':data_size, 
                  'n_epochs':n_epochs, 
                  'c_lambda':c_lambda, 
                  'crit_repeats':crit_repeats, 
                  'lr':lr, 
                  'beta_1':beta_1, 
                  'beta_2':beta_2,
                 'lambda_gp':lambda_gp}
        
        # Loop through parameters and write each as a new line
        for name, value in params.items():
            setupfile.write(name + "=" + str(value) +"\n")
        setupfile.close()
        
    return None