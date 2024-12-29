# -----------------------------------------------------------------------------
# Imports and Basic Setup
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
from torch.utils.tensorboard import SummaryWriter
from me_models import MultiExitViT
from me_ResNet import MultiExitResNet
from Dloaders import Dloaders
from myArgparser import parse_args

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset configurations
IMG_SIZE = 224
dataset_name = {'cifar10': datasets.CIFAR10,'cifar100': datasets.CIFAR100,'imagenet': None}
dataset_outdim = {'cifar10': 10,'cifar100': 100,'imagenet': 1000}

# -----------------------------------------------------------------------------
# Configuration Parameters
# -----------------------------------------------------------------------------
# Model and training setup
unfreeze_ees_list = [4]  # List of exit layers to unfreeze [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
batch_size = 1024
data_choice = 'imagenet'  # Options: 'cifar10', 'cifar100', 'imagenet'
model_choice = 'resnet'  # Options: 'vit' or 'resnet'
num_workers = 6

# Model loading configuration
mevit_isload = False  # Whether to load a pre-trained multi-exit model
mevit_pretrained_path = f'models/{model_choice}/{data_choice}/{unfreeze_ees_list[0]}/best_model.pth'
backbone_path = f'models/{model_choice}/{data_choice}/{model_choice}_{data_choice}_backbone.pth'

# Training hyperparameters
max_epochs = 200
start_lr = 1e-4
weight_decay = 1e-4

# Early stopping and learning rate scheduling
early_stop_patience = 8
lr_decrease_factor = 0.6
lr_decrease_patience = 2

# Exit configuration
ee_list = [0,1,2,3,4,5,6,7,8,9]  # List of all possible exits
exit_loss_weights = [1,1,1,1,1,1,1,1,1,1,1]  # Weight for each exit's loss
classifier_wise = True  # Whether to train specific classifiers only
unfreeze_ees = []  # Exits to unfreeze during training

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
args = parse_args(unfreeze_ees_list)
if args.unfreeze_exits is not None:
    unfreeze_ees_list = args.unfreeze_exits

# -----------------------------------------------------------------------------
# Trainer Class Definition
# -----------------------------------------------------------------------------
class Trainer:
    """
    A trainer class for multi-exit neural networks.
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self, model, params):
        """
        Initialize the trainer with model and training parameters.
        
        Args:
            model: The neural network model to train
            params: Dictionary containing training parameters and configurations
        """
        # Basic setup
        self.model = model
        self.device = next(model.parameters()).device
        
        # Training parameters
        self.num_epochs = params['num_epochs']; self.loss_func = params["loss_func"]
        self.opt = params["optimizer"]; self.lr_scheduler = params["lr_scheduler"]
        
        # Data loaders
        self.dloaders = params["dloaders"]
        self.train_dl, self.val_dl = self.dloaders.get_loaders()
        self.train_dataset, self.val_dataset = self.dloaders.get_datasets()
        
        # Model configuration
        self.data_choice = params["data_choice"]; self.model_choice = params["model_choice"]
        self.classifier_wise = params["classifier_wise"]; self.unfreeze_ees = params["unfreeze_ees"]
        
        # Checkpoint handling
        self.isload = params["isload"]; self.path_chckpnt = params["path_chckpnt"]
        self.best_loss = float('inf'); self.old_epoch = 0
        
        # Early stopping
        self.early_stop_patience = params["early_stop_patience"]
        
        # Setup directories and logging
        self._setup_directories()
        self._initialize_logging()
        
        # Model preparation
        if self.isload:
            self._load_checkpoint()
        if self.classifier_wise:
            self._freeze_layers()
            
        # Save initial specifications
        self._save_specifications()

    def _setup_directories(self):
        """Set up necessary directories for model checkpoints and logs."""
        self.current_time = time.strftime('%m%d_%H%M%S', time.localtime())
        self.path = f'./models/{self.model_choice}/{self.data_choice}/{self.unfreeze_ees[0]}'
        os.makedirs(self.path, exist_ok=True)

    def _initialize_logging(self):
        """Initialize TensorBoard writer for logging training progress."""
        self.writer = SummaryWriter(f'./runs/{self.data_choice}/{self.unfreeze_ees[0]}')

    def _load_checkpoint(self):
        """Load model and optimizer state from checkpoint."""
        checkpoint = torch.load(self.path_chckpnt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.old_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']

    def _freeze_layers(self):
        """Freeze all layers except specified exits and classifiers."""
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze specified exits and their classifiers
        for idx in self.unfreeze_ees:
            if model_choice == 'vit':
                for param in self.model.classifiers[idx].parameters():
                    param.requires_grad = True
            for param in self.model.ees[idx].parameters():
                param.requires_grad = True

    def _save_specifications(self):
        """Save model and training specifications to a text file."""
        spec_txt = (
            f'Optimizer: {self.opt.__class__.__name__}\n'
            f'Learning Rate: {self.opt.param_groups[0]["lr"]}\n'
            f'Batch Size: {self.train_dl.batch_size}\n'
            f'Total Epochs: {self.num_epochs}\n'
            f'Load Pretrained: {self.isload}\n'
            f'Checkpoint Path: {self.path_chckpnt}\n'
            f'Exit Loss Weights: {self.model.getELW()}\n'
        )
        with open(f"{self.path}/spec.txt", "w") as file:
            file.write(spec_txt)

    @staticmethod
    def get_lr(opt):
        """Get current learning rate from optimizer."""
        return opt.param_groups[0]['lr']

    @staticmethod
    def metric_batch(output, label):
        """Calculate accuracy for a batch."""
        pred = output.argmax(1, keepdim=True)
        return pred.eq(label.view_as(pred)).sum().item()

    def loss_batch(self, output_list, label, elws, mode):
        """
        Calculate loss and accuracy for a batch.
        
        Args:
            output_list: List of model outputs from different exits
            label: Ground truth labels
            elws: Exit loss weights
            mode: 'train' or 'val'
        """
        # Calculate losses and accuracies for each exit
        losses = [self.loss_func(output, label) * elw 
                 for output, elw in zip(output_list, elws)]
        accs = [self.metric_batch(output, label) for output in output_list]
        
        # Perform backpropagation if in training mode
        if mode == "train":
            self.opt.zero_grad()
            cnter = 1
            tot_train = len(self.unfreeze_ees)
            
            # Backward pass for each exit
            for idx, loss in enumerate(losses):
                if idx in self.unfreeze_ees:
                    if cnter < tot_train:
                        loss.backward(retain_graph=True)
                    else:
                        loss.backward()
                    cnter += 1
                    
            self.opt.step()
            
        return [loss.item() for loss in losses], accs

    def loss_epoch(self, data_loader, epoch, mode="train"):
        """
        Calculate loss and accuracy for an entire epoch.
        
        Args:
            data_loader: DataLoader for either training or validation
            epoch: Current epoch number
            mode: 'train' or 'val'
        """
        running_loss = [0.0] * self.model.exit_num
        running_metric = [0.0] * self.model.exit_num
        len_data = len(self.train_dataset) if mode == "train" else len(self.val_dataset)
        elws = self.model.getELW()

        # Process each batch with progress bar
        with tqdm(data_loader, desc=f"{mode}: Epoch {epoch}", unit="batch", leave=False) as t:
            for xb, yb in t:
                xb, yb = xb.to(self.device), yb.to(self.device)
                output_list = self.model(xb)
                losses, accs = self.loss_batch(output_list, yb, elws, mode)

                # Update running statistics
                running_loss = [sum(x) for x in zip(running_loss, losses)]
                running_metric = [sum(x) for x in zip(running_metric, accs)]
                
                # Update progress bar
                t.set_postfix(accuracy=[round(100 * acc / len(xb), 3) for acc in accs])

        # Calculate final metrics
        running_loss = [loss / len_data for loss in running_loss]
        running_acc = [100 * metric / len_data for metric in running_metric]

        # Log metrics to TensorBoard
        self._log_metrics(running_loss, running_acc, epoch, mode)

        return sum(running_loss), running_acc

    def _log_metrics(self, losses, accuracies, epoch, mode):
        """Log training metrics to TensorBoard."""
        loss_dict = {f'exit{idx}': loss for idx, loss in enumerate(losses)}
        acc_dict = {f'exit{idx}': acc for idx, acc in enumerate(accuracies)}
        
        self.writer.add_scalars(f'{mode}/loss', loss_dict, epoch)
        self.writer.add_scalars(f'{mode}/accuracy', acc_dict, epoch)
        self.writer.add_scalar(f'{mode}/total_loss', sum(losses), epoch)

    def train(self):
        """Main training loop."""
        start_time = time.time()
        early_stop_counter = 0

        for epoch in range(self.old_epoch, self.old_epoch + self.num_epochs):
            current_lr = self.get_lr(self.opt)
            print(f'Epoch {epoch}/{self.old_epoch + self.num_epochs - 1}, LR={current_lr}')

            # Training phase
            self.model.train()
            train_loss, train_accs = self.loss_epoch(self.train_dl, epoch, mode="train")

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_loss, val_accs = self.loss_epoch(self.val_dl, epoch, mode="val")

            # Model checkpoint handling
            if val_loss < self.best_loss:
                self._save_best_model(epoch, val_loss)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"No improvement in validation loss. Counter: {early_stop_counter}")
                if early_stop_counter >= self.early_stop_patience:
                    print("Early stopping triggered!")
                    break

            # Learning rate scheduling
            self.lr_scheduler.step(val_loss)

            # Print epoch summary
            self._print_epoch_summary(epoch, train_loss, train_accs, val_loss, val_accs, start_time)

        # Save final model and close TensorBoard writer
        self._save_final_model(val_loss)
        self.writer.close()

        # Add final metrics to specifications file
        with open(f"{self.path}/spec.txt", "a") as file:
            file.write(f"\nFinal Training Accuracy: {train_accs}\n")
            file.write(f"Final Validation Accuracy: {val_accs}\n")

    def _save_best_model(self, epoch, val_loss):
        """Save model checkpoint when validation loss improves."""
        self.best_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'loss': val_loss,
        }, f'{self.path}/best_model.pth')
        print("Saved best model weights!")

    def _save_final_model(self, val_loss):
        """Save final model state and training results."""
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'loss': val_loss,
        }, f'{self.path}/final_model.pth')

    def _print_epoch_summary(self, epoch, train_loss, train_accs, val_loss, val_accs, start_time):
        """Print summary of training progress."""
        elapsed_time = (time.time() - start_time) / 60
        hours, minutes = divmod(elapsed_time, 60)
        print(f'Training Loss: {train_loss:.6f}, Training Accuracy: {train_accs}')
        print(f'Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accs}')
        print(f'Time Elapsed: {int(hours)}h {int(minutes)}m')
        print('-' * 50)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("data_choice: ", data_choice)
    print("model_choice: ", model_choice)
    print("unfreeze_ees_list: ", unfreeze_ees_list)
    
    # 1. Data Loader 초기화
    print("Initializing data loaders...")
    dloaders = Dloaders(data_choice=data_choice, batch_size=batch_size, IMG_SIZE=IMG_SIZE, num_workers=num_workers)

    # 2. 기본 모델 초기화 (ResNet 또는 ViT)
    print(f"Initializing base {model_choice} model...")
    if model_choice == 'resnet':
        # ResNet 모델 초기화
        ptd_model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        
        # ImageNet이 아닌 경우 출력층 수정
        if data_choice != 'imagenet':
            ptd_model.fc = nn.Linear(ptd_model.fc.in_features, dataset_outdim[data_choice])
            # 사전 학습된 가중치 로드
            ptd_model.load_state_dict(torch.load(backbone_path))
        
        ptd_model = ptd_model.to(device)
        
    elif model_choice == 'vit':
        # ViT 모델 초기화
        ptd_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        
        # ImageNet이 아닌 경우 출력층 수정
        if data_choice != 'imagenet':
            ptd_model.heads.head = nn.Linear(ptd_model.heads.head.in_features, dataset_outdim[data_choice])
            # 사전 학습된 가중치 로드
            ptd_model.load_state_dict(torch.load(backbone_path))
            
        ptd_model = ptd_model.to(device)

    # 3. 각 exit layer에 대해 학습 수행
    print("Starting training for each exit layer...")
    for i in unfreeze_ees_list:
        print(f"\nTraining for exit layer {i}")
        
        # 3.0 path 설정
        mevit_pretrained_path = f'models/{model_choice}/{data_choice}/{i}/best_model.pth'
        
        # 3.1 Multi-Exit 모델 초기화
        if model_choice == 'resnet':
            model = MultiExitResNet(ptd_model,num_classes=dataset_outdim[data_choice]).to(device)
        elif model_choice == 'vit':
            model = MultiExitViT(ptd_model,num_classes=dataset_outdim[data_choice]).to(device)
        
        # 3.2 옵티마이저 초기화
        optimizer = optim.Adam(model.parameters(),lr=start_lr,weight_decay=weight_decay)
        
        # 3.3 손실 함수 정의
        criterion = nn.CrossEntropyLoss()
        
        # 3.4 학습률 스케줄러 초기화
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_decrease_factor, patience=lr_decrease_patience,verbose=True)

        # 3.5 학습 파라미터 설정
        training_params = {
            'num_epochs': max_epochs,'loss_func': criterion,
            'optimizer': optimizer,'data_choice': data_choice,
            'model_choice': model_choice,'dloaders': dloaders, 'lr_scheduler': lr_scheduler,
            'isload': mevit_isload,'path_chckpnt': mevit_pretrained_path,
            'classifier_wise': classifier_wise,'unfreeze_ees': [i],  # 특정 exit layer만 학습
            'early_stop_patience': early_stop_patience
        }

        # 3.6 트레이너 초기화 및 학습 시작
        print(f"Initializing trainer for exit {i}")
        trainer = Trainer(model=model, params=training_params)
        trainer.train()
        
        print(f"Completed training for exit layer {i}")
        print("=" * 80)

    print("\nTraining completed for all specified exit layers!")