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
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)  # Initialize temperature as 1.0

    def forward(self, logits):
        return logits / self.temperature

def optimize_temperature(output_tensor, labels, scalers, optimizers, lr_schedulers, exit_num=11, max_epochs=200):
    writer = SummaryWriter(writer_path)
    accs = [0]*exit_num
    for i in range(exit_num):
        accs[i] = labels.eq(output_tensor[i].argmax(dim = 1)).sum().item() / len(labels)
    
    for i in range(exit_num):
        scalers[i].train()

    # Optimization loop
    for epoch in tqdm(range(max_epochs), desc='Training scalers'):
        for i in range(exit_num):
            optimizers[i].zero_grad()
            '''t = scalers[i](output_tensor[i])
            t = F.softmax(t, dim = 1)
            t = torch.max(t, dim = 1).values
            t = (t - accs[i]) ** 2
            loss = t.sum()'''
            logits_scaled = scalers[i](output_tensor[i])
            loss = F.cross_entropy(logits_scaled, labels)
            loss.backward()
            optimizers[i].step()
            lr_schedulers[i].step(loss)
            
            writer.add_scalar(f'loss_{i}th_exit', loss, epoch)
            writer.add_scalar(f'T_val_{i}th_exit', scalers[i].temperature.item(), epoch)
            

    print(f"Optimized Temperature: {[scaler.temperature.item() for scaler in scalers]}")
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
    model_choice='resnet'   # resnet, vit
    mevit_isload=True
    mevit_pretrained_path=f'models/{model_choice}/{data_choice}/integrated_ee.pth'
    max_epochs = 2000  # Set your max epochs

    backbone_path=f'models/{model_choice}/{data_choice}/{model_choice}_{data_choice}_backbone.pth'
    start_lr=1e-3
    weight_decay=1e-2

    ee_list=[0,1,2,3,4,5,6,7,8,9]#exit list ex) [0,1,2,3,4,5,6,7,8,9]
    exit_loss_weights=[1,1,1,1,1,1,1,1,1,1,1]#exit마다 가중치
    exit_num=11
    
    lr_decrease_factor = 0.6
    lr_decrease_patience = 2
    
    cache_file_path = f'cache_result_{model_choice}_{data_choice}.pt'
    writer_path = f'./runs/{data_choice}/ms/'
    scaler_path = f'models/{model_choice}/{data_choice}/temperature_scaler.pth'
    ##############################################################
    dloaders=Dloaders(data_choice=data_choice,batch_size=batch_size,IMG_SIZE=IMG_SIZE)

    # load cached output tensor
    output_tensor = torch.load(cache_file_path).to(device)
    _, test_dataset = dloaders.get_datasets()
    labels_list = test_dataset.targets
    labels=torch.tensor(labels_list).to(device)

    # Temperature Scaling
    temperature_scalers = [TemperatureScaling().to(device) for _ in range(exit_num)]
    optimizers = [optim.Adam([temperature_scalers[i].temperature], lr=start_lr, weight_decay=weight_decay) for i in range(exit_num)]
    lr_schedulers=[ReduceLROnPlateau(optimizers[i], mode='min', factor=lr_decrease_factor, patience=lr_decrease_patience, verbose=True) for i in range(exit_num)]
    # Define a function to optimize the temperature

    # Optimize temperature
    temperature_scaler = optimize_temperature(output_tensor, labels, temperature_scalers, optimizers, lr_schedulers, exit_num=exit_num, max_epochs=max_epochs)

    # save temperature scaling values
    torch.save(temperature_scaler, scaler_path)
