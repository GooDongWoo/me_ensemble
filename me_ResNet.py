import torch
import torch.nn as nn
from torchvision import models

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # BatchNorm include bias, therefore, set conv2d as bias=False
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        # Initialize residual function
        for m in self.residual_function.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize shortcut
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class BottleNeck(BasicBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()  # Note: Changed to BasicBlock to avoid double initialization

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        # Initialize residual function
        for m in self.residual_function.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize shortcut
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ResNetBranchExit(nn.Module):
    def __init__(self, input_shape, num_classes=1000, reduction=True):
        super(ResNetBranchExit, self).__init__()
        self.input_shape = input_shape
        
        layers = nn.ModuleList()
        in_channels = input_shape[1]
        
        if reduction:
            self.reduce = BottleNeck(in_channels, in_channels//4)
            layers.append(self.reduce)
        else:
            self.reduce = None

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        layers.append(self.gap)
        
        self.layers = layers
        self.linear_dim = self._get_linear_size(layers)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.linear_dim, num_classes)
        )

        # Initialize classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_linear_size(self, layers):
        tmp = torch.rand(*self.input_shape)
        for layer in layers:
            with torch.no_grad():
                tmp = layer(tmp)
        return int(tmp.shape[1])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)

class MultiExitResNet(nn.Module):
    def __init__(self,base_model=models.resnet101(),
                ee_list=[12, 21, 30, 39, 48, 57, 66, 75, 84, 93],
                exit_loss_weights=[1,1,1,1,1,1,1,1,1,1,1],  # 10개의 early exit + 1개의 final exit = 11개
                num_classes=1000,
                data_shape=[3,3,224,224]):
        super(MultiExitResNet, self).__init__()
        assert len(ee_list) + 1 == len(exit_loss_weights), 'len(ee_list)+1==len(exit_loss_weights) should be True'
        
        self.base_model = base_model
        self.data_shape = data_shape
        self.num_classes = num_classes
        
        # base model components
        self.init_conv = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            nn.ReLU(inplace=True),
            base_model.maxpool
        )
        
        # Create backbone
        self.backbone = nn.ModuleList()
        for layer in [base_model.layer1, base_model.layer2, 
                     base_model.layer3, base_model.layer4]:
            self.backbone.extend(layer)
            
        self.end_layers = nn.Sequential(
            base_model.avgpool,
            nn.Flatten(),
            base_model.fc
        )
        
        # Multiple Exit configuration
        self.exit_loss_weights = [elw/sum(exit_loss_weights) for elw in exit_loss_weights]
        self.ee_list = ee_list
        self.exit_num = len(ee_list) + 1
        self.ees = nn.ModuleList()
        self._build_exits()
        
        self.each_ee_test_mode = False

    def _build_exits(self):
        device = next(self.parameters()).device
        with torch.no_grad():
            previous_shapes = []
            tmp = self.init_conv(torch.rand(*(self.data_shape)).to(device))
            
            eidx = 0
            for idx, module in enumerate(self.backbone):
                tmp = module(tmp)
                if eidx < self.exit_num-1 and idx+1 == (self.ee_list[eidx]//3):
                    previous_shapes.append(tmp.shape)
                    eidx += 1
            
            for shape in previous_shapes:
                self.ees.append(ResNetBranchExit(shape, self.num_classes))

    def getELW(self):
        if self.exit_loss_weights is None:
            self.exit_loss_weights = [1] * self.exit_num
        return self.exit_loss_weights

    def set_each_ee_test_mode(self, num=None):
        self.each_ee_test_mode = True
        self.each_ee_test_what_num = num if num is not None else self.exit_num - 1

    def set_MC_dropout_mode(self, p):
        self.each_ee_test_mode = False
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = p
        return

    def forward_each_ee(self, x):
        target_exit = self.each_ee_test_what_num
        features = self.init_conv(x)

        if target_exit == (self.exit_num - 1):
            for module in self.backbone:
                features = module(features)
            return self.end_layers(features)
        else:
            for idx, module in enumerate(self.backbone):
                features = module(features)
                if idx + 1 == (self.ee_list[target_exit]//3):
                    return self.ees[target_exit](features)

    def forward(self, x):
        if self.each_ee_test_mode:
            return self.forward_each_ee(x)

        outputs = []
        features = self.init_conv(x)

        eidx = 0
        for idx, module in enumerate(self.backbone):
            features = module(features)
            if eidx < self.exit_num-1 and idx+1 == (self.ee_list[eidx]//3):
                outputs.append(self.ees[eidx](features))
                eidx += 1

        outputs.append(self.end_layers(features))
        return outputs

if __name__ == '__main__':
    import torchvision.models as models
    resnet101 = models.resnet101()
    model = MultiExitResNet(resnet101)
    tmp = model(torch.rand(1,3,224,224))
    print(tmp[0].shape)