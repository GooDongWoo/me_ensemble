# utils
import torch
import torch.nn as nn
from torchvision import datasets,models
from tqdm import tqdm  # Importing tqdm for progress bar
from Dloaders import Dloaders
# Check if GPU is available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
IMG_SIZE = 224
batch_size = 32
dataset_name = {'cifar10':datasets.CIFAR10, 'cifar100':datasets.CIFAR100,'imagenet':None}
dataset_outdim = {'cifar10':10, 'cifar100':100,'imagenet':1000}

##############################################################
data_choice='cifar10'
model_choice = 'resnet' # ['vit', 'resnet']
##############################################################

dloaders=Dloaders(data_choice=data_choice,batch_size=batch_size,IMG_SIZE=IMG_SIZE)
train_loader,test_loader = dloaders.get_loaders()
if model_choice == 'resnet':
    model = models.resnet101(weights=None)
    model.fc = nn.Linear(model.fc.in_features, dataset_outdim[data_choice])
    model = model.to(device)
elif model_choice == 'vit':
    # Load the pretrained ViT model from the saved file
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, dataset_outdim[data_choice])  # Ensure output matches the number of classes

backbone_path=f'models/{model_choice}/{data_choice}/{model_choice}_{data_choice}_backbone.pth'
# Load model weights
model.load_state_dict(torch.load(backbone_path, map_location=device))
model = model.to(device)

def getTestACC():
    # Set the model to evaluation mode
    model.eval()
    # Test the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(test_loader, unit="batch",leave=False) as t:
            for images, labels in tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                t.set_postfix(accuracy=100*correct/total)

    print(f"Accuracy of the model on the test images: {100 * correct / total}%")

getTestACC()
from torchinfo import summary
summary(model,input_size= (64, 3, IMG_SIZE, IMG_SIZE))