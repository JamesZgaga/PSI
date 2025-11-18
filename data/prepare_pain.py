import os
import yaml
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def read_config():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_pain_dataset_df(config):
    """创建痛苦表情数据集的DataFrame"""
    data_config = config['data']['pain']
    root_dir = data_config['root_dir']
    classes = data_config['classes']
    
    data = []
    
    # 遍历所有图像文件
    for filename in os.listdir(root_dir):
        if not filename.endswith(('.jpg', '.JPG', '.jpeg', '.png')):
            continue
            
        # 解析文件名，获取用户ID和痛苦等级
        parts = os.path.splitext(filename)[0].split('_')
        if len(parts) != 2:
            print(f"Warning: 文件名格式不符合预期: {filename}")
            continue
            
        user_id = parts[0]
        pain_level = parts[1]
        
        # 验证pain_level是否有效
        if pain_level not in classes:
            print(f"Warning: 痛苦等级不在配置的类别中: {pain_level}")
            continue
        
        img_path = os.path.join(root_dir, filename)
        
        # 读取图像以获取尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: 无法读取图像: {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        data.append({
            'image_path': img_path,
            'user_id': user_id,
            'pain_level': pain_level,
            'class_id': classes.index(pain_level),
            'width': w,
            'height': h
        })
    
    df = pd.DataFrame(data)
    return df

def main():
    config = read_config()
    
    # 创建痛苦表情数据集DataFrame
    print("创建痛苦表情数据集DataFrame...")
    df = create_pain_dataset_df(config)
    
    # 由于样本量较小(50张)，使用交叉验证更合适
    # 但这里为了示例，我们仍分出一个小测试集
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['class_id'], random_state=42
    )
    
    # 保存DataFrame
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/pain_train.csv', index=False)
    val_df.to_csv('data/processed/pain_val.csv', index=False)
    
    # 打印统计信息
    print(f"\n总样本数: {len(df)}")
    print(f"训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(val_df)}")
    
    class_counts = df['pain_level'].value_counts()
    print("\n痛苦等级分布:")
    for level, count in class_counts.items():
        print(f"{level}: {count} ({count/len(df)*100:.2f}%)")
    
    user_counts = df['user_id'].value_counts()
    print("\n用户分布:")
    for user, count in user_counts.items():
        print(f"{user}: {count}")

if __name__ == "__main__":
    main()
