# Facial Expression Analysis and Mental State Assessment System 🌍

**Choose Language:**
- [English](README.md) • [中文(简体)](README.zh-CN.md)

## Project Introduction

This project implements a deep learning-based facial expression analysis and mental state assessment system. Taking single-face static images as input, the system gradually extracts emotion-related facial features through a series of computational modules, and finally outputs a Comprehensive Psychological State Index (PSI) as the core output of the emotion perception subsystem in a multimodal closed-loop system.

The overall workflow consists of face detection and alignment, expression probability inference, pain intensity estimation, and multi-indicator fusion, providing users with comprehensive facial expression and emotional state analysis.

## System Architecture

### Technology Stack

* Python 3.8+
* PyTorch 2.1.1
* OpenCV
* DeepFace
* Scikit-learn
* Pandas & NumPy
* Matplotlib
* Albumentations

### Core Features

* Supports multiple expression recognition (Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral)
* Pain level assessment (LV1-LV5)
* Comprehensive Psychological State Index (PSI) calculation
* Single-image and batch processing modes
* Visualized analysis results
* Custom model training

### Project Structure
```
facial\_emotion\_analysis/

├── config/               # 配置文件

│   └── config.yaml       # 主配置文件

├── data/                 # 数据处理相关

│   ├── prepare\_expression.py  # 表情数据集预处理脚本

│   └── prepare\_pain.py       # 痛苦表情数据集预处理脚本

├── models/               # 模型定义

│   ├── emotion\_model.py  # 表情识别模型

│   └── pain\_model.py     # 痛苦等级评估模型

├── src/                  # 核心源码

│   ├── face\_detection.py  # 人脸检测与对齐

│   ├── emotion\_analysis.py # 表情概率推理

│   ├── pain\_assessment.py  # 痛苦强度推理

│   └── psi\_calculator.py   # PSI计算

├── train/                # 训练脚本

│   ├── train\_emotion.py  # 表情模型训练

│   └── train\_pain.py     # 痛苦模型训练

├── utils/                # 工具函数

│   ├── data\_utils.py     # 数据处理工具

│   ├── model\_utils.py    # 模型工具

│   └── visualization.py  # 可视化工具

├── eval.py               # 模型评估脚本

├── main.py               # 主程序

├── requirements.txt      # 依赖包列表

└── README.md             # 项目文档
```

## Installation & Configuration

### Environment Requirements

* CUDA-compatible GPU (Recommended)
* At least 8GB RAM
* Python 3.8+

### Installation Steps

1. Clone the repository
   ```
git clone https://github.com/JamesZgaga/PSI.git
   ```
2. Create a virtual environment (Recommended)
   ```
conda create -n face_emotion python=3.8
conda activate face_emotion
   ```
3. Install dependencies
   ```
pip install -r requirements.txt
   ```
4. Configure data paths
   Edit the `config/config.yaml` file to set dataset paths and model parameters.

## Dataset Introduction

The project uses two main datasets:

### Expression Dataset

* Number of classes: 7 (Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral)
* Total data volume: 15,339 images
* Training set: 12,271 images
* Test set: 3,068 images
* Class distribution: Imbalanced - "Happiness" is the most (39%), "Fear" is the least (2.3%)
* Format: YOLO format annotations

### Pain Expression Dataset

This dataset is derived from the research results of other peers and requires application for access.

* Number of classes: 5 pain levels (LV1-LV5)
* For details: [https://github.com/ais-lab/RU-PITENS-database](https://github.com/ais-lab/RU-PITENS-database)

### Data Preprocessing

Before training the model, preprocess the raw datasets:
   ```
Process expression dataset
python data/prepare_expression.py
Process pain expression dataset
python data/prepare_pain.py
   ```
### Preprocessing Steps Include

* Parse YOLO format annotations
* Extract face regions
* Handle class imbalance
* Generate training/validation data framework
* Calculate class weights

## Model Training

### Train Expression Recognition Model
   ```
python train/train_emotion.py
   ```
### The training process will automatically:

* Load the preprocessed expression dataset
* Handle imbalance using class weights
* Train an expression recognition model with ResNet50 or specified backbone
* Save the best model and training history
* Generate evaluation report and confusion matrix

### Train Pain Level Assessment Model
   ```
python train/train_pain.py
   ```
Due to the small sample size of the pain dataset, training uses:

* Lightweight ResNet18 model
* Data augmentation techniques
* Small batch size
* Early stopping strategy

## Model Evaluation

Evaluate model performance:
   ```
# Evaluate all models
python eval.py
# Evaluate only expression recognition model
python eval.py --model emotion
# Evaluate only pain level model
python eval.py --model pain
# Evaluate specified model files
python eval.py --emotion_model models/custom_emotion_model.pth --pain_model models/custom_pain_model.pth
   ```

The evaluation will generate:

* Classification report (Precision, Recall, F1-score)
* Confusion matrix visualization
* Performance metrics summary

## System Usage

### Analyze a Single Image

   ```
# Adjust the image path according to your actual project
python main.py --input /mnt/MCP/Deepface/data/testImage/xxx.jpg --output results
   ```

### Batch Process Images
   ```
# Adjust the directory according to your actual project
python main.py --input /mnt/MCP/Deepface/data/testImage/ --output results --batch
   ```
## Output Results

The system will generate the following outputs:
   ```
I0000 00:00:1763460310.539051 90522 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22298 MB memory: -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:1b:00.0, compute capability: 8.6Emotion Probabilities: {'Surprise': '0.0000', 'Fear': '0.0447', 'Disgust': '0.0000', 'Happiness': '0.0004', 'Sadness': '0.9548', 'Anger': '0.0001', 'Neutral': '0.0000'}Pain Score: 0.5576Pain Level Probabilities: {'LV1': '0.1523', 'LV2': '0.1968', 'LV3': '0.2155', 'LV4': '0.1391', 'LV5': '0.2963'}Psychological State Index (PSI): 91.1171Visualization result saved to results/test6_analysis.png
   ```
