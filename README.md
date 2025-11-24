# Facial Expression Analysis and Mental State Assessment System 🌍

**Choose Language:**
- [English](README.md) • [Chinese (Simplified)](README.zh-CN.md)

## Project Introduction

This project implements a deep learning-based facial expression analysis and mental state assessment system. The system takes single-face static images as input, gradually extracts emotion-related facial features through a series of computational modules, and finally outputs a comprehensive Psychological State Index (PSI), which serves as the core output of the emotion perception subsystem in a multimodal closed-loop system.

The overall process consists of face detection and alignment, expression probability inference, pain intensity estimation, and multi-index fusion, providing users with comprehensive facial expression and emotional state analysis.

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

### Features

* Supports multiple expression recognition (Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral)
* Pain level assessment (LV1-LV5)
* Comprehensive Psychological State Index (PSI) calculation
* Single image and batch processing modes
* Visualization of analysis results
* Custom model training

### Project Structure


```
facial\_emotion\_analysis/

├── config/               # 配置文件

│   └── config.yaml       # 主配置文件

├── data/                 # 数据处理相关

│   ├── expression  # Expression Dataset
│   ├── Level 1-5   # Pain Expression Dataset: requires a separate application to peers; see the dataset description below for details.
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

## Installation and Configuration

### Environment Requirements

* CUDA-compatible GPU (recommended)
* At least 8GB RAM
* Python 3.8+

### Installation Steps

1. Clone the repository


```
git clone https://github.com/JamesZgaga/PSI.git
```



Create a virtual environment (recommended)



```
conda create -n face\_emotion python=3.8

conda activate face\_emotion
```



Install dependencies



```
pip install -r requirements.txt
```



Configure data paths

   Edit the `config/config.yaml` file to set dataset paths and model parameters

## Dataset Introduction

This project uses two main datasets:

### Expression Dataset

* Number of classes: 7 (Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral)
* Total data: 15,339 images
* Training set: 12,271 images
* Test set: 3,068 images
* Class distribution: Imbalanced, with "Happiness" being the most frequent (39%) and "Fear" the least (2.3%)
* Format: YOLO format annotations

### Pain Expression Dataset

This dataset is derived from research results of other peers and requires application for access.

* Number of classes: 5 pain levels (LV1-LV5)
* For details: [https://github.com/ais-lab/RU-PITENS-database](https://github.com/ais-lab/RU-PITENS-database)

### Data Preprocessing

Before training the models, you need to process the original datasets:



```
# Process expression dataset

python data/prepare\_expression.py

# Process pain expression dataset  

python data/prepare\_pain.py
```

### Preprocessing Steps Include:

* Parse YOLO format labels
* Extract facial regions
* Handle class imbalance
* Generate training/validation data frameworks
* Calculate class weights

## Model Training

### Train Expression Recognition Model



```
python train/train\_emotion.py
```

### The Training Process Will Automatically:

* Load the preprocessed expression dataset
* Use class weights to handle imbalance issues
* Train an expression recognition model with ResNet50 or specified backbone network
* Save the best model and training history
* Generate evaluation reports and confusion matrices

### Train Pain Level Assessment Model



```
python train/train\_pain.py
```

Due to the small sample size of the pain dataset, training uses:

* Lightweight ResNet18 model
* Data augmentation techniques
* Smaller batch size
* Early stopping strategy

## Model Evaluation

Evaluate model performance:


```
# 评估所有模型

python eval.py

# 仅评估表情识别模型

python eval.py --model emotion

# 仅评估痛苦等级模型

python eval.py --model pain

# 评估指定模型文件

python eval.py --emotion\_model models/custom\_emotion\_model.pth --pain\_model models/custom\_pain\_model.pth
```

Evaluation will generate:

* Classification report (precision, recall, F1-score)
* Confusion matrix visualization
* Summary of performance metrics

## Using the System

### Analyze a Single Image


```
#根据自己实际项目调整图像地址
python main.py --input /mnt/MCP/Deepface/data/testImage/test6.jpg --output results
```

### Batch Process Images



```
#根据自己实际项目调整目录
python main.py --input /mnt/MCP/Deepface/data/testImage/ --output results --batch
```

## Output Results

The system will generate the following outputs:
'''
I0000 00:00:1763460310.539051   90522 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22298 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:1b:00.0, compute capability: 8.6
情绪概率: {'Surprise': '0.0000', 'Fear': '0.0447', 'Disgust': '0.0000', 'Happiness': '0.0004', 'Sadness': '0.9548', 'Anger': '0.0001', 'Neutral': '0.0000'}
痛苦分数: 0.5576
痛苦等级概率: {'LV1': '0.1523', 'LV2': '0.1968', 'LV3': '0.2155', 'LV4': '0.1391', 'LV5': '0.2963'}
心理状态指数(PSI): 91.1171
可视化结果已保存至 results/test6_analysis.png
'''

* Visualization of analysis results (including original image, detected face, emotion probabilities, pain levels, and PSI index)
* Detailed analysis data in JSON format
* Statistical summaries and distribution charts in batch processing mode

## Psychological State Index (PSI) Calculation

PSI comprehensively considers the weighted combination of negative emotions and pain levels:



```
PSI = clip(w₁×(1 - P₊) + w₂×E + w₃×D₋, 0, 1) × 100
```

---

###  Parameter Description

| Parameter | Description | Value Range | Default Weight |
|-----------|-------------|-------------|----------------|
| `w₁` | Weight ratio for negative emotions | 0–1 | 0.4 |
| `w₂` | Weight for emotional fluctuation entropy | 0–1 | 0.3 |
| `w₃` | Weight for normalized negative persistence | 0–1 | 0.3 |
| `P₊` | Positive-emotion ratio (share of positive frames in the time window) | 0–1 | — |
| `E`  | Emotional fluctuation entropy (normalized Shannon entropy; higher ⇒ less stable) | 0–1 | — |
| `D₋` | Normalized negative-emotion persistence (continuous negative duration ÷ max allowed) | 0–1 | — |
| `clip(..., 0, 1)` | Clips intermediate result to [0, 1] | — | — |

---

---

### PSI Level Classification

| PSI Range | Level | Description |
|-----------|-------|-------------|
| 0–30 | **Good** | Stable psyche, positivity dominant, low fluctuation |
| 31–60 | **Moderate** | Medium fluctuation or occasional negative spells |
| 61–100 | **Poor** | Persistent negativity, high instability, or prolonged negative state |

---

### Configuration Flexibility

- All settings—weights (`w₁, w₂, w₃`), emotion categories, time-window parameters, and level thresholds—can be customized in the `psi_time_based` section of `config/config.yaml`
- The system automatically validates and corrects invalid configurations (e.g., negative weights, implausible thresholds) to guarantee robust computation

## Example Results
(This project does not provide a visual web frontend; you can design it yourself)
<img width="1537" height="1172" alt="image" src="https://github.com/user-attachments/assets/bf6e9ec2-8933-41c1-a3b4-d233fcc2234c" />

Analysis example: The above figure shows an example of the system's analysis results, including the original image, detected face, emotion probability distribution, pain level distribution, and PSI index dashboard.

## Performance Optimization

To improve system performance, we adopt the following optimization strategies:

* Batch processing mode: Reduce IO overhead for multiple image processing
* GPU acceleration: Automatically detect and use available GPU resources
* Model quantization: Optional int8 quantization to speed up inference
* Image size optimization: Adjust the size of processed images according to task requirements

## Frequently Asked Questions

* **Q: What if the system cannot detect a face?**

  A: Please ensure that the face in the input image is clearly visible and of appropriate size. You can try adjusting the `min_face_size` parameter in `config.yaml` to lower the detection threshold.

* **Q: What if the pain level assessment is inaccurate?**

  A: Due to the small sample size of the pain expression dataset, the model's generalization ability is limited. You can improve it in the following ways:

  * Collect more pain expression data
  * Adjust `pain_weight` in `config.yaml` to reduce its weight in PSI calculation
  * Use cross-validation to adjust model parameters

* **Q: What if the system runs slowly?**

  A: Performance can be optimized through the following methods:

  * Use GPU acceleration
  * Reduce the size of processed images
  * Use lightweight backbone networks (such as MobileNet)
  * Enable batch processing mode for multiple images
