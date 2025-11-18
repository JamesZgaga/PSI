import sys
from pathlib import Path  # 导入Path类

# 获取当前脚本所在目录
current_script_dir = Path(__file__).parent
# 获取项目根目录
project_root = current_script_dir.parent
# 将根目录添加到Python搜索路径
sys.path.append(str(project_root))

import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# 导入自定义模块
from models.emotion_model import EmotionModel
from utils.data_utils import create_dataloaders

def read_config():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    """训练模型"""
    model = model.to(device)
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
        # 【在验证阶段结束后，传入验证指标调用 scheduler.step()】
        if scheduler:
            scheduler.step(val_epoch_acc)  # 传入验证准确率（因mode='max'）
        
        # 保存最佳模型
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), 'models/best_emotion_model.pth')
            print(f'Best model saved with accuracy: {best_acc:.4f}')
    
    return model, history

def evaluate_model(model, val_loader, device='cuda', class_names=None):
    """评估模型性能"""
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Evaluation'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算分类报告
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    if class_names:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/emotion_confusion_matrix.png')
    plt.close()

def plot_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/emotion_training_history.png')
    plt.close()

def main():
    # 读取配置
    config = read_config()
    model_config = config['model']['emotion']
    data_config = config['data']['expression']
    
    # 读取处理过的数据
    train_df = pd.read_csv('data/processed/expression_train.csv')
    test_df = pd.read_csv('data/processed/expression_test.csv')
    class_names = data_config['classes']
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        train_df, 
        test_df, 
        img_size=model_config['img_size'],
        batch_size=model_config['batch_size']
    )
    
    # 创建模型
    model = EmotionModel(
        num_classes=len(class_names),
        backbone=model_config['backbone'],
        pretrained=model_config['pretrained']
    )
    
    # 加载类别权重来处理不平衡问题
    class_weights = np.load('data/processed/expression_class_weights.npy')
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    
    # 学习率调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True
    )
    
    # 训练模型
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=model_config['epochs'],
        device=device
    )
    
    # 绘制训练历史
    plot_history(history)
    
    # 加载最佳模型并评估
    model.load_state_dict(torch.load('models/best_emotion_model.pth'))
    evaluate_model(model, val_loader, device, class_names)
    
    # 保存最终模型
    torch.save(model.state_dict(), model_config['save_path'])
    print(f"最终模型已保存到 {model_config['save_path']}")

if __name__ == "__main__":
    main()