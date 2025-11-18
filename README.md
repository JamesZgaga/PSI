<img width="1537" height="1172" alt="image" src="https://github.com/user-attachments/assets/bf6e9ec2-8933-41c1-a3b4-d233fcc2234c" /># 面部表情分析与心理状态评估系统

## 项目简介

本项目实现了一个基于深度学习的面部表情分析与心理状态评估系统。系统以单人脸静态图像作为输入，通过一系列计算模块逐步提取面部情绪相关特征，最终输出一个综合心理状态指数（PSI），作为多模态闭环系统中情绪感知子系统的核心输出。

整体流程由人脸检测与对齐、表情概率推理、痛苦强度推断以及多指标融合等环节构成，为用户提供全面的面部表情和情绪状态分析。

## 系统架构

### 技术栈



* Python 3.8+

* PyTorch 2.1.1

* OpenCV

* DeepFace

* Scikit-learn

* Pandas & NumPy

* Matplotlib

* Albumentations

### 功能特点



* 支持多种表情识别（惊讶、恐惧、厌恶、开心、悲伤、愤怒、中性）

* 痛苦等级评估（LV1-LV5）

* 综合心理状态指数（PSI）计算

* 单图和批量处理模式

* 可视化分析结果

* 自定义模型训练

### 项目结构



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

## 安装与配置

### 环境要求



* CUDA 兼容 GPU (推荐)

* 至少 8GB RAM

* Python 3.8+

### 安装步骤



1. 克隆代码库



```
git clone https://github.com/JamesZgaga/PSI.git
```



1. 创建虚拟环境 (推荐)



```
conda create -n face\_emotion python=3.8

conda activate face\_emotion
```



1. 安装依赖



```
pip install -r requirements.txt
```



1. 配置数据路径

   编辑`config/config.yaml`文件，设置数据集路径和模型参数

## 数据集介绍

本项目使用两个主要数据集：

### 表情数据集



* 类别数量: 7 类 (惊讶、恐惧、厌恶、开心、悲伤、愤怒、中性)

* 数据总量: 15,339 张图像

* 训练集: 12,271 张

* 测试集: 3,068 张

* 类别分布：不平衡，"开心" 类别最多 (39%)，"恐惧" 类别最少 (2.3%)

* 格式: YOLO 格式标注

### 痛苦表情数据集

该数据集来源于其他同行的研究成果，该数据集需申请方可使用。

* 类别数量: 5 级痛苦等级 (LV1-LV5)

* 详情请见：[https://github.com/ais-lab/RU-PITENS-database](https://github.com/ais-lab/RU-PITENS-database)

### 数据预处理

在训练模型前，需要先处理原始数据集：



```
# 处理表情数据集

python data/prepare\_expression.py

# 处理痛苦表情数据集

python data/prepare\_pain.py
```

### 预处理步骤包括：



* 解析 YOLO 格式标签

* 提取人脸区域

* 处理类别不平衡

* 生成训练 / 验证数据框架

* 计算类别权重

## 模型训练

### 训练表情识别模型



```
python train/train\_emotion.py
```

### 训练过程将自动：



* 加载预处理后的表情数据集

* 使用类别权重处理不平衡问题

* 训练 ResNet50 或指定骨干网络的表情识别模型

* 保存最佳模型和训练历史

* 生成评估报告和混淆矩阵

### 训练痛苦等级评估模型



```
python train/train\_pain.py
```

由于痛苦数据集样本量较小，训练采用：



* 轻量级 ResNet18 模型

* 数据增强技术

* 较小的批量大小

* 提前停止策略

## 模型评估

评估模型性能：



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

评估将生成：



* 分类报告（精度、召回率、F1 分数）

* 混淆矩阵可视化

* 性能指标汇总

## 使用系统

### 分析单张图像



```
#根据自己实际项目调整图像地址
python main.py --input /mnt/MCP/Deepface/data/testImage/test6.jpg --output results
```

### 批量处理图像



```
#根据自己实际项目调整目录
python main.py --input /mnt/MCP/Deepface/data/testImage/ --output results --batch
```

## 输出结果

系统将生成以下输出：



* 可视化分析结果（包含原图、检测到的人脸、情绪概率、痛苦等级和 PSI 指数）

* JSON 格式的详细分析数据

* 批处理模式下的统计摘要和分布图

## 心理状态指数 (PSI) 计算

PSI 综合考虑了负面情绪和痛苦等级的加权组合：



```
PSI = negative\_emotion\_weight \* negative\_emotion\_score + pain\_weight \* pain\_score
```

其中：



* negative\_emotion\_score: 负面情绪 (恐惧、厌恶、悲伤、愤怒) 概率之和

* pain\_score: 痛苦等级评估分数 (0-1 范围)

* 权重默认为: negative\_emotion\_weight=0.6, pain\_weight=0.4
* PSI=计算的结果*100

PSI 值范围在 0-100 之间，值越大表示心理状态越差。系统根据 PSI 值将风险等级分为 5 级：



* 0-20: 很低风险

* 20-40: 低风险

* 40-60: 中等风险

* 60-80: 高风险

* 80-100: 很高风险

## 示例结果
（可视化Web前端不项目不提供，可自行设计）

分析示例：上图展示了系统分析结果示例，包含原始图像、检测到的人脸、情绪概率分布、痛苦等级分布以及 PSI 指数仪表盘

## 性能优化

为提高系统性能，我们采用以下优化策略：


* 批处理模式：减少多张图像处理的 IO 开销

* GPU 加速：自动检测并使用可用的 GPU 资源

* 模型量化：可选的 int8 量化以加速推理

* 图像尺寸优化：根据任务需求调整处理图像的尺寸

## 常见问题


* **Q: 系统无法检测到人脸怎么办？**

  A: 请确保输入图像中人脸清晰可见且大小适中。您可以尝试调整`config.yaml`中的`min_face_size`参数降低检测阈值。

* **Q: 痛苦等级评估不准确怎么办？**

  A: 由于痛苦表情数据集样本量较小，模型泛化能力有限。您可以通过以下方式改进：


  * 收集更多痛苦表情数据

  * 调整`config.yaml`中的`pain_weight`降低其在 PSI 计算中的权重

  * 使用交叉验证调整模型参数

* **Q: 系统运行速度慢怎么办？**

  A: 可以通过以下方法优化性能：


  * 使用 GPU 加速

  * 减小处理图像的尺寸

  * 使用轻量级骨干网络 (如 MobileNet)

  * 启用批处理模式处理多张图像
