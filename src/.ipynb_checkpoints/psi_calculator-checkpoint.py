import numpy as np
import yaml

def read_config():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

class PSICalculator:
    def __init__(self):
        """初始化心理状态指数计算器"""
        # 读取配置
        config = read_config()
        psi_config = config['psi']
        
        # 设置权重
        self.negative_emotion_weight = psi_config['negative_emotion_weight']
        self.pain_weight = psi_config['pain_weight']
        
        # 获取负面情绪列表
        self.negative_emotions = psi_config['negative_emotions']
    
    def calculate_negative_emotion_score(self, emotion_probs):
        """计算负面情绪分数"""
        # 提取负面情绪的概率总和
        negative_score = sum(emotion_probs.get(emotion, 0) for emotion in self.negative_emotions)
        return negative_score
    
    def calculate_psi(self, emotion_probs, pain_result):
        """
        计算综合心理状态指数(PSI)
        
        参数:
            emotion_probs: 情绪概率字典
            pain_result: 痛苦评估结果
            
        返回:
            psi: 综合心理状态指数 (0-1，值越高表示心理状态越糟糕)
        """
        # 计算负面情绪分数
        negative_emotion_score = self.calculate_negative_emotion_score(emotion_probs)
        
        # 获取痛苦分数
        pain_score = pain_result['pain_score'] if pain_result else 0
        
        # 计算加权PSI
        psi = (self.negative_emotion_weight * negative_emotion_score + 
               self.pain_weight * pain_score)
        
        # 确保PSI在0-1范围内
        psi = max(0, min(1, psi))
        
        return psi*100
