# utils
import torch
import torch.nn as nn
from torchvision import datasets,models
from me_train import MultiExitViT
from me_ResNet import *
from collections import OrderedDict


IMG_SIZE = 224
dataset_name = {'cifar10':datasets.CIFAR10, 'cifar100':datasets.CIFAR100,'imagenet':None}
dataset_outdim = {'cifar10':10, 'cifar100':100,'imagenet':1000}

##############################################################
################ 0. Hyperparameters ##########################
data_choice='imagenet'
model_choice = 'resnet' # ['vit', 'resnet']
# Path to the saved model
ee0_path=f'models/{model_choice}/{data_choice}/0/best_model.pth'
ee1_path=f'models/{model_choice}/{data_choice}/1/best_model.pth'
ee2_path=f'models/{model_choice}/{data_choice}/2/best_model.pth'
ee3_path=f'models/{model_choice}/{data_choice}/3/best_model.pth'
ee4_path=f'models/{model_choice}/{data_choice}/4/best_model.pth'
ee5_path=f'models/{model_choice}/{data_choice}/5/best_model.pth'
ee6_path=f'models/{model_choice}/{data_choice}/6/best_model.pth'
ee7_path=f'models/{model_choice}/{data_choice}/7/best_model.pth'
ee8_path=f'models/{model_choice}/{data_choice}/8/best_model.pth'
ee9_path=f'models/{model_choice}/{data_choice}/9/best_model.pth'
##############################################################
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
##############################################################
paths=[ee0_path,ee1_path,ee2_path,ee3_path,ee4_path,ee5_path,ee6_path,ee7_path,ee8_path,ee9_path]
##############################################################
checkpoint = torch.load(paths[0])['model_state_dict']
new_model.load_state_dict(checkpoint,strict=True)
##############################################################
for i in range(1,10):
    checkpoint = torch.load(paths[i])['model_state_dict']
    # 2. 로드하려는 모델의 특정 부분에 해당하는 파라미터 필터링
    partial_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        # 특정 부분의 이름을 필터링하여 추가합니다. 예: 'ees'나 'classifiers'로 시작하는 키들만 선택
        if k.startswith(f'ees.{i}') or k.startswith(f'classifiers.{i}'):
            partial_state_dict[k] = v

    # 3. 기존 모델에 부분적으로 적용
    # strict=False로 하여 일부 파라미터만 로드하도록 설정합니다.
    new_model.load_state_dict(partial_state_dict,strict=False)

##############################################################
# Save the model
torch.save(new_model.state_dict(), f'models/{model_choice}/{data_choice}/integrated_ee.pth')
print('Integrated model saved successfully!')