import torch
import torch.nn as nn
import torchvision.models as models

class PainModel(nn.Module):
    def __init__(self, num_classes=5, backbone='resnet18', pretrained=True):
        super(PainModel, self).__init__()
        
        # 选择骨干网络 - 对于小数据集使用较小的模型
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone for pain model: {backbone}")
        
        # 提取特征提取部分
        self.features = nn.Sequential(*list(base_model.children())[:-1])
            
        # 分类器头部
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 减小dropout以适应小数据集
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        # 提取特征
        features = self.features(x)
        features = torch.flatten(features, 1)
        
        # 分类
        logits = self.classifier(features)
        
        return logits
