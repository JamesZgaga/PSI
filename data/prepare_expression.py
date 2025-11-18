import os
import yaml
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import class_weight

def read_config():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_yolo_label(label_path, img_width, img_height, num_classes):
    """解析YOLO格式标签文件"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    labels = []
    for line in lines:
        values = line.strip().split()
        class_id = int(values[0])
        x_center = float(values[1]) * img_width
        y_center = float(values[2]) * img_height
        width = float(values[3]) * img_width
        height = float(values[4]) * img_height
        
        # 计算边界框坐标
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        labels.append({
            'class_id': class_id,
            'bbox': [x1, y1, x2, y2]
        })
    
    return labels

def create_dataset_df(config, split='train'):
    """创建表情数据集的DataFrame"""
    data_config = config['data']['expression']
    root_dir = data_config['root_dir']
    
    if split == 'train':
        image_dir = os.path.join(root_dir, data_config['train_images'])
        label_dir = os.path.join(root_dir, data_config['train_labels'])
    else:
        image_dir = os.path.join(root_dir, data_config['test_images'])
        label_dir = os.path.join(root_dir, data_config['test_labels'])
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    data = []
    classes = data_config['classes']
    
    for img_file in tqdm(image_files, desc=f"Processing {split} dataset"):
        img_path = os.path.join(image_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        labels = parse_yolo_label(label_path, w, h, len(classes))
        
        for label in labels:
            class_id = label['class_id']
            if class_id < len(classes):
                class_name = classes[class_id]
                x1, y1, x2, y2 = label['bbox']
                
                # 确保坐标在图像范围内
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                # 只有当边界框有效时添加数据
                if x2 > x1 and y2 > y1:
                    data.append({
                        'image_path': img_path,
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id,
                        'class_name': class_name,
                        'width': w,
                        'height': h
                    })
    
    df = pd.DataFrame(data)
    return df

def calculate_class_weights(df):
    """计算类别权重以处理类别不平衡问题"""
    class_ids = df['class_id'].values
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(class_ids),
        y=class_ids
    )
    return weights

def main():
    config = read_config()
    
    # 创建训练集和测试集DataFrame
    print("创建表情数据集DataFrame...")
    train_df = create_dataset_df(config, 'train')
    test_df = create_dataset_df(config, 'test')
    
    # 保存DataFrame
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/expression_train.csv', index=False)
    test_df.to_csv('data/processed/expression_test.csv', index=False)
    
    # 计算并保存类别权重
    class_weights = calculate_class_weights(train_df)
    np.save('data/processed/expression_class_weights.npy', class_weights)
    
    # 打印统计信息
    print(f"\n训练集样本数: {len(train_df)}")
    print(f"测试集样本数: {len(test_df)}")
    
    class_counts = train_df['class_name'].value_counts()
    print("\n训练集类别分布:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} ({count/len(train_df)*100:.2f}%)")
    
    print("\n类别权重:")
    for i, weight in enumerate(class_weights):
        class_name = config['data']['expression']['classes'][i]
        print(f"{class_name}: {weight:.4f}")

if __name__ == "__main__":
    main()
