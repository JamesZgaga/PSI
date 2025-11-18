import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from models.emotion_model import EmotionModel
from models.pain_model import PainModel
from utils.data_utils import FaceDataset

def read_config():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_emotion_model(model_path=None):
    """评估表情识别模型"""
    # 读取配置
    config = read_config()
    model_config = config['model']['emotion']
    data_config = config['data']['expression']
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载测试数据集
    test_df = pd.read_csv('data/processed/expression_test.csv')
    test_dataset = FaceDataset(test_df, img_size=model_config['img_size'], augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=model_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = EmotionModel(
        num_classes=len(data_config['classes']),
        backbone=model_config['backbone'],
        pretrained=False
    )
    
    # 加载训练好的模型
    if model_path is None:
        model_path = model_config['save_path']
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 评估模型
    all_preds = []
    all_labels = []
    all_probs = []
    
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="评估表情模型"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    eval_time = time.time() - start_time
    
    # 计算分类报告
    class_names = data_config['classes']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n表情识别模型评估报告:")
    print(report)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('表情识别混淆矩阵')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在混淆矩阵中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 保存混淆矩阵
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/emotion_confusion_matrix_eval.png')
    
    print(f"\n评估完成，耗时 {eval_time:.2f} 秒")
    print(f"混淆矩阵已保存至 results/emotion_confusion_matrix_eval.png")
    
    return report

def evaluate_pain_model(model_path=None):
    """评估痛苦等级模型"""
    # 读取配置
    config = read_config()
    model_config = config['model']['pain']
    data_config = config['data']['pain']
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载验证数据集
    val_df = pd.read_csv('data/processed/pain_val.csv')
    val_dataset = FaceDataset(val_df, img_size=model_config['img_size'], augment=False, crop_face=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = PainModel(
        num_classes=len(data_config['classes']),
        backbone=model_config['backbone'],
        pretrained=False
    )
    
    # 加载训练好的模型
    if model_path is None:
        model_path = model_config['save_path']
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 评估模型
    all_preds = []
    all_labels = []
    all_probs = []
    
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="评估痛苦等级模型"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    eval_time = time.time() - start_time
    
    # 计算分类报告
    class_names = data_config['classes']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n痛苦等级模型评估报告:")
    print(report)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('痛苦等级混淆矩阵')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # 在混淆矩阵中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 保存混淆矩阵
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/pain_confusion_matrix_eval.png')
    
    print(f"\n评估完成，耗时 {eval_time:.2f} 秒")
    print(f"混淆矩阵已保存至 results/pain_confusion_matrix_eval.png")
    
    return report

def main():
    parser = argparse.ArgumentParser(description='评估面部表情分析模型')
    parser.add_argument('--model', type=str, choices=['emotion', 'pain', 'all'], default='all',
                        help='要评估的模型')
    parser.add_argument('--emotion_model', type=str, default=None,
                        help='表情识别模型路径')
    parser.add_argument('--pain_model', type=str, default=None,
                        help='痛苦等级模型路径')
    
    args = parser.parse_args()
    
    if args.model == 'emotion' or args.model == 'all':
        print("\n===== 评估表情识别模型 =====")
        evaluate_emotion_model(args.emotion_model)
    
    if args.model == 'pain' or args.model == 'all':
        print("\n===== 评估痛苦等级模型 =====")
        evaluate_pain_model(args.pain_model)

if __name__ == "__main__":
    main()
