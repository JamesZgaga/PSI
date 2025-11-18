import os
import torch
import numpy as np
import cv2
import yaml
from torchvision import transforms
import torch.nn.functional as F

from models.emotion_model import EmotionModel

def read_config():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

class EmotionAnalyzer:
    def __init__(self, model_path=None):
        """初始化表情分析器"""
        # 读取配置
        self.config = read_config()
        
        # 获取类别名称
        self.emotion_labels = self.config['data']['expression']['classes']
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        self.model = EmotionModel(
            num_classes=len(self.emotion_labels),
            backbone=self.config['model']['emotion']['backbone'],
            pretrained=False
        )
        
        # 加载训练好的模型
        if model_path is None:
            model_path = self.config['model']['emotion']['save_path']
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Emotion model loaded from {model_path}")
        else:
            print(f"Warning: Emotion model not found at {model_path}")
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['model']['emotion']['img_size'], 
                              self.config['model']['emotion']['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def analyze_emotion(self, face_img):
        """
        分析人脸图像的情绪概率
        
        参数:
            face_img: 对齐后的人脸图像
            
        返回:
            emotion_probs: 情绪概率字典
        """
        if face_img is None:
            print("No face image provided for emotion analysis")
            return {emotion: 1.0/len(self.emotion_labels) if emotion == 'Neutral' else 0.0 
                   for emotion in self.emotion_labels}
        
        try:
            # 预处理图像
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(face_img_rgb)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
            # 创建概率字典
            emotion_probs = {self.emotion_labels[i]: float(probs[i]) for i in range(len(self.emotion_labels))}
            
            return emotion_probs
            
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
            # 返回默认概率（中性情绪）作为fallback
            return {emotion: 1.0/len(self.emotion_labels) if emotion == 'Neutral' else 0.0 
                   for emotion in self.emotion_labels}
