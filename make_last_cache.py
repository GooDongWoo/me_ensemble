import torch
import torch.nn as nn
from torchvision import models,datasets
from me_models import MultiExitViT
from me_ResNet import MultiExitResNet
from tqdm import tqdm
import numpy as np
from Dloaders import Dloaders

if __name__ == '__main__':
    ####################################################################
    IMG_SIZE = 224
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_name = {'cifar10':datasets.CIFAR10, 'cifar100':datasets.CIFAR100,'imagenet':None}
    dataset_outdim = {'cifar10':10, 'cifar100':100,'imagenet':1000}
    ##############################################################
    ################ 0. Hyperparameters ##########################
    ##############################################################
    batch_size = 1024
    data_choice='imagenet'
    model_choice = 'resnet' # ['vit', 'resnet']
    mevit_pretrained_path=f"models/{model_choice}/{data_choice}/integrated_ee.pth"

    backbone_path=f'models/{model_choice}/{data_choice}/{model_choice}_{data_choice}_backbone.pth'
    cache_file_path = f'cache_result_{model_choice}_{data_choice}.pt'
    
    exit_num=11
    ##############################################################
    dloaders=Dloaders(data_choice=data_choice,batch_size=batch_size,IMG_SIZE=IMG_SIZE)
    train_loader,test_loader = dloaders.get_loaders()

    # Load the pretrained ViT model from the saved file
    if model_choice == 'vit':
        # Load the pretrained ViT model from the saved file
        ptd_model = models.vit_b_16(weights=None)
        ptd_model.heads.head = nn.Linear(ptd_model.heads.head.in_features, dataset_outdim[data_choice])  # Ensure output matches the number of classes
        # Load model weights
        new_model = MultiExitViT(base_model=ptd_model,num_classes=dataset_outdim[data_choice])
    elif model_choice == 'resnet':
        # Load the pretrained ResNet model from the saved file
        ptd_model = models.resnet101()
        ptd_model.fc = nn.Linear(ptd_model.fc.in_features, dataset_outdim[data_choice])
        # Load model weights
        new_model = MultiExitResNet(base_model=ptd_model,num_classes=dataset_outdim[data_choice])    
    
    new_model.load_state_dict(torch.load(mevit_pretrained_path))
    new_model.to(device)

    # Initialize lists to store outputs from each exit and labels
    output_list_list = [[] for _ in range(exit_num)]
    labels_list = []

    # Run inference on the test set and collect the logits from each exit
    new_model.eval()  # Set new_model to evaluation mode
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Collecting logits", leave=False):
            images, labels = images.cuda(), labels.cuda()
            output_list = new_model(images)  # Get the output from all exits
            
            # Store the output from each exit
            for i in range(exit_num):
                output_list_list[i].append(output_list[i])
            
            # Store the labels
            labels_list.append(labels)
    # Concatenate the collected outputs and labels
    output_tensor = torch.tensor(np.array([torch.cat(output_list_list[i]).to('cpu') for i in range(exit_num)])).to(device)
    labels = torch.cat(labels_list).to(device)
    cacahe_dict = {'output_tensor':output_tensor,'labels':labels}
    # 리스트를 바이너리 파일로 저장하기
    torch.save(cacahe_dict, cache_file_path)
    
    print('Cache file saved to', cache_file_path)