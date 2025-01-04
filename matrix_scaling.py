import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Dloaders import Dloaders

# Define the Temperature Scaling class
class MatrixScaling(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.W = torch.nn.Parameter(torch.eye(num_classes))
        self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return torch.matmul(logits, self.W) + self.b

def optimize_W(output_tensor, labels, scalers, optimizers, lr_schedulers, exit_num=11, max_epochs=200):
    writer = SummaryWriter(writer_path)
    
    for i in range(exit_num):
        scalers[i].train()

    # Optimization loop
    for epoch in tqdm(range(max_epochs), desc='Training scalers'):
        for i in range(exit_num):
            optimizers[i].zero_grad()
            
            logits_scaled = scalers[i](output_tensor[i])
            loss = F.cross_entropy(logits_scaled, labels)
            loss.backward()
            optimizers[i].step()
            lr_schedulers[i].step(loss)
            
            writer.add_scalar(f'loss_{i}th_exit', loss, epoch)

    writer.close()
    return scalers
####################################################################
if __name__=='__main__':
    IMG_SIZE = 224
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_name = {'cifar10':datasets.CIFAR10, 'cifar100':datasets.CIFAR100,'imagenet':None}
    dataset_outdim = {'cifar10':10, 'cifar100':100,'imagenet':1000}
    ##############################################################
    ################ 0. Hyperparameters ##########################
    ##############################################################
    batch_size = 1024
    data_choice='imagenet'
    model_choice='resnet' #vit or resnet
    max_epochs = 1000  # Set your max epochs

    start_lr=1e-3
    weight_decay=1e-2

    exit_num=11
    
    lr_decrease_factor = 0.6
    lr_decrease_patience = 2
    
    cache_file_path = f'cache_result_{model_choice}_{data_choice}.pt'
    writer_path = f'./runs/{model_choice}/{data_choice}/ms/'
    scaler_path = f'models/{model_choice}/{data_choice}/matrix_scaler.pth'
    ##############################################################
    # load cached output tensor
    cache_dict = torch.load(cache_file_path)
    output_tensor , labels = cache_dict['output_tensor'].to(device), cache_dict['labels'].to(device)

    # Temperature Scaling
    mat_scalers = [MatrixScaling(dataset_outdim[data_choice]).to(device) for _ in range(exit_num)]
    optimizers = [optim.Adam(mat_scalers[i].parameters(), lr=start_lr, weight_decay=weight_decay) for i in range(exit_num)]
    lr_schedulers=[ReduceLROnPlateau(optimizers[i], mode='min', factor=lr_decrease_factor, patience=lr_decrease_patience, verbose=True) for i in range(exit_num)]
    # Define a function to optimize the temperature

    # Optimize temperature
    mat_scalers = optimize_W(output_tensor, labels, mat_scalers, optimizers, lr_schedulers, exit_num=exit_num, max_epochs=max_epochs)

    # save temperature scaling values
    torch.save(mat_scalers, scaler_path)
    print(f"Matrix scaling values saved to {scaler_path}")