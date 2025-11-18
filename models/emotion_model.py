import torch
import torch.nn as nn
import torchvision.models as models

class EmotionModel(nn.Module):
    def __init__(self, num_classes=7, backbone='resnet50', pretrained=True):
        super(EmotionModel, self).__init__()
        
        # 选择骨干网络
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 提取特征提取部分
        if 'resnet' in backbone:
            self.features = nn.Sequential(*list(base_model.children())[:-1])
        elif 'efficientnet' in backbone:
            self.features = base_model.features
            
        # 分类器头部
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )
        
        # 特征嵌入层 (用于特征提取)
        self.embedding = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, return_features=False):
        # 提取特征
        features = self.features(x)
        features = torch.flatten(features, 1)
        
        # 生成嵌入特征
        embeddings = self.embedding(features)
        
        # 分类
        logits = self.classifier(features)
        
        if return_features:
            return logits, embeddings
        else:
            return logits
