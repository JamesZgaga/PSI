import os
import torch
import numpy as np
import cv2
import yaml
from torchvision import transforms
import torch.nn.functional as F

from models.pain_model import PainModel

def read_config():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

class PainAssessor:
    def __init__(self, model_path=None):
        """初始化痛苦评估器"""
        # 读取配置
        self.config = read_config()
        
        # 获取痛苦等级名称
        self.pain_levels = self.config['data']['pain']['classes']
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        self.model = PainModel(
            num_classes=len(self.pain_levels),
            backbone=self.config['model']['pain']['backbone'],
            pretrained=False
        )
        
        # 加载训练好的模型
        if model_path is None:
            model_path = self.config['model']['pain']['save_path']
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Pain model loaded from {model_path}")
        else:
            print(f"Warning: Pain model not found at {model_path}")
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['model']['pain']['img_size'], 
                              self.config['model']['pain']['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 痛苦等级权重 (LV1->0.0, LV5->1.0)
        self.pain_weights = {level: i/4 for i, level in enumerate(self.pain_levels)}
    
    def predict_pain_level(self, face_img):
        """
        预测人脸图像的痛苦等级
        
        参数:
            face_img: 人脸图像
            
        返回:
            result: 包含痛苦分数和等级概率的字典
        """
        if face_img is None:
            print("No face image provided for pain assessment")
            return {
                'pain_score': 0.0,
                'pain_level_probs': {level: 1.0 if level == self.pain_levels[0] else 0.0 
                                   for level in self.pain_levels}
            }
        
        try:
            # 预处理图像
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(face_img_rgb)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
            # 创建等级概率字典
            pain_level_probs = {self.pain_levels[i]: float(probs[i]) for i in range(len(self.pain_levels))}
            
            # 计算加权痛苦分数
            pain_score = sum(prob * self.pain_weights[level] for level, prob in pain_level_probs.items())
            
            return {
                'pain_score': pain_score,
                'pain_level_probs': pain_level_probs
            }
            
        except Exception as e:
            print(f"Error in pain assessment: {e}")
            # 返回默认结果作为fallback
            return {
                'pain_score': 0.0,
                'pain_level_probs': {level: 1.0 if level == self.pain_levels[0] else 0.0 
                                   for level in self.pain_levels}
            }
