# utils
import torch
import torch.nn as nn
from torchvision import datasets, models
from tqdm import tqdm  # Importing tqdm for progress bar
from me_models import MultiExitViT
from me_ResNet import MultiExitResNet
from Dloaders import Dloaders

IMG_SIZE = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_name = {'cifar10':datasets.CIFAR10, 'cifar100':datasets.CIFAR100,'imagenet':None}
dataset_outdim = {'cifar10':10, 'cifar100':100,'imagenet':1000}
##############################################################
################ 0. Hyperparameters ##########################
batch_size = 1024
data_choice = 'imagenet'
model_choice = 'resnet'  # ['vit', 'resnet']
mevit_pretrained_path=f'models/{model_choice}/{data_choice}/integrated_ee.pth'

backbone_path = f'models/{model_choice}/{data_choice}/{model_choice}_{data_choice}_backbone.pth'
ee_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # exit list ex) [0,1,2,3,4,5,6,7,8,9]
exit_loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # exit마다 가중치

##############################################################
if __name__ == '__main__':
    dloaders=Dloaders(data_choice=data_choice,batch_size=batch_size,IMG_SIZE=IMG_SIZE)
    train_loader,test_loader = dloaders.get_loaders()
    train_dataset, test_dataset = dloaders.get_datasets()
    
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
    
    new_model.eval()
    running_metric = [0.0] * new_model.exit_num
    len_data = len(test_dataset)

    with torch.no_grad():
        with tqdm(test_loader, unit="batch", leave=False) as t:
            for xb, yb in t:
                xb, yb = xb.to(device), yb.to(device)
                output_list = new_model(xb)
                accs = [output.argmax(1).eq(yb).sum().item() for output in output_list]
                running_metric = [sum(x) for x in zip(running_metric, accs)]
                
                t.set_postfix(accuracy=[round(100 * acc / len(xb),3) for acc in accs])

    running_acc = [100 * metric / len_data for metric in running_metric]
    print(f'total Test Accuracy: {running_acc}')
