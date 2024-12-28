import webdataset as wds
import torch
from torch.utils.data import DataLoader
import io
from PIL import Image
import os
from tqdm import tqdm
import tarfile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

def convert_arrow_to_tars(arrow_dataset, output_dir, samples_per_tar=1000):
    """Arrow 데이터셋을 TAR 파일들로 변환"""
    os.makedirs(output_dir, exist_ok=True)
    
    for shard_idx in tqdm(range(0, len(arrow_dataset), samples_per_tar)):
        tar_filename = os.path.join(output_dir, f'shard_{shard_idx:06d}.tar')
        
        with tarfile.open(tar_filename, 'w') as tar:
            for idx in range(shard_idx, min(shard_idx + samples_per_tar, len(arrow_dataset))):
                sample = arrow_dataset[idx]
                img = sample["image"]
                label = sample["label"]
                
                # 이미지를 바이트로 변환
                img_byte = io.BytesIO()
                img.save(img_byte, format='JPEG')
                img_byte = img_byte.getvalue()
                
                # TAR에 추가
                img_info = tarfile.TarInfo(f'{idx:08d}.jpg')
                img_info.size = len(img_byte)
                tar.addfile(img_info, io.BytesIO(img_byte))
                
                # 레이블 추가
                label_byte = str(label).encode('utf-8')
                label_info = tarfile.TarInfo(f'{idx:08d}.cls')
                label_info.size = len(label_byte)
                tar.addfile(label_info, io.BytesIO(label_byte))

class WebDatasetImageNet:
    def __init__(self, data_dir, batch_size, num_workers, transforms=None):
        self.transforms = transforms
        
        # WebDataset 파이프라인 생성
        dataset = (
            wds.WebDataset(f"{data_dir}/shard_*.tar")
            .decode('rgb')
            .to_tuple('jpg', 'cls')
            .map_tuple(self.process_image, lambda x: int(x))
        )
        
        # DataLoader 생성
        self.loader = DataLoader(
            dataset.batched(batch_size),
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def process_image(self, img):
        if self.transforms:
            return self.transforms(img)
        return img

# Dloaders 클래스 수정
class Dloaders:
    def __init__(self, data_choice='cifar100', batch_size=1024, IMG_SIZE=224, num_workers=6):
        self.dataset_name = {'cifar10':datasets.CIFAR10, 'cifar100':datasets.CIFAR100,'imagenet':None}
        self.dataset_outdim = {'cifar10':10, 'cifar100':100,'imagenet':1000}
        
        self.data_choice = data_choice
        if data_choice == 'imagenet':
            import imagenet_load
            
            # Arrow 데이터셋을 TAR로 변환 (처음 한 번만 실행)
            train_output_dir = './data/imagenet_tars/train'
            test_output_dir = './data/imagenet_tars/val'
            
            if not os.path.exists(train_output_dir):
                print("Converting training data to TAR format...")
                convert_arrow_to_tars(imagenet_load.IMAGENET_DATASET_TRAIN.dataset, train_output_dir)
            
            if not os.path.exists(test_output_dir):
                print("Converting validation data to TAR format...")
                convert_arrow_to_tars(imagenet_load.IMAGENET_DATASET_TEST.dataset, test_output_dir)
            
            # WebDataset 로더 생성
            self.train_loader = WebDatasetImageNet(
                train_output_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                transforms=imagenet_load.trans
            ).loader
            
            self.test_loader = WebDatasetImageNet(
                test_output_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                transforms=imagenet_load.trans
            ).loader
        else:
            transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.train_dataset = self.dataset_name[data_choice](root='./data', train=True, download=True, transform=transform)
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.test_dataset = self.dataset_name[data_choice](root='./data', train=False, download=True, transform=transform)
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return None
    
    def get_loaders(self):
        return self.train_loader,  self.test_loader
    
    def get_datasets(self):
        return self.train_dataset,  self.test_dataset
