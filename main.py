import os
import argparse
import yaml
import cv2
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import json

from src.face_detection import CustomFaceDetector  
from src.emotion_analysis import EmotionAnalyzer
from src.pain_assessment import PainAssessor
from src.psi_calculator import PSICalculator
from utils.visualization import visualize_results, plot_batch_results

def read_config():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_single_image(image_path, output_dir, face_detector, emotion_analyzer, pain_assessor, psi_calculator):
    """处理单张图像并返回分析结果"""
    print(f"\n处理图像: {image_path}")
    start_time = time.time()
    
    # 步骤1: 人脸检测与对齐
    aligned_face, face_coords = face_detector.detect_and_align(image_path=image_path)
    if aligned_face is None:
        print("未检测到人脸，跳过")
        return None
    
    # 步骤2: 表情概率推理
    emotion_probs = emotion_analyzer.analyze_emotion(aligned_face)
    print("情绪概率:", {k: f"{v:.4f}" for k, v in emotion_probs.items()})
    
    # 步骤3: 痛苦强度推理
    pain_result = pain_assessor.predict_pain_level(aligned_face)
    if pain_result:
        print(f"痛苦分数: {pain_result['pain_score']:.4f}")
        print("痛苦等级概率:", {k: f"{v:.4f}" for k, v in pain_result['pain_level_probs'].items()})
    
    # 步骤4: 计算PSI
    psi = psi_calculator.calculate_psi(emotion_probs, pain_result)
    print(f"心理状态指数(PSI): {psi:.4f}")
    
    # 可视化结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_analysis.png"
        output_path = os.path.join(output_dir, output_filename)
        visualize_results(
            image_path, 
            aligned_face, 
            emotion_probs, 
            pain_result, 
            psi, 
            face_coords, 
            output_path
        )
    
    # 处理时间
    process_time = time.time() - start_time
    print(f"处理完成，耗时: {process_time:.2f}秒")
    
    # 返回分析结果
    return {
        'image_path': image_path,
        'emotion_probs': emotion_probs,
        'pain_result': pain_result,
        'psi': psi,
        'process_time': process_time
    }

def process_directory(directory, output_dir, face_detector, emotion_analyzer, pain_assessor, psi_calculator):
    """处理目录中的所有图像"""
    results = []
    
    # 获取所有图像文件
    image_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(os.path.join(directory, filename))
    
    print(f"发现 {len(image_files)} 张图像")
    
    # 处理每一张图像
    for image_path in tqdm(image_files, desc="处理图像"):
        result = process_single_image(
            image_path, 
            output_dir, 
            face_detector, 
            emotion_analyzer, 
            pain_assessor, 
            psi_calculator
        )
        if result:
            results.append(result)
    
    return results

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='面部表情分析与心理状态评估')
    parser.add_argument('--input', type=str, required=True, help='输入图像或目录路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--batch', action='store_true', help='批处理模式')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 初始化模块
    face_detector = CustomFaceDetector()
    emotion_analyzer = EmotionAnalyzer()
    pain_assessor = PainAssessor()
    psi_calculator = PSICalculator()
    
    # 判断输入是文件还是目录
    if os.path.isfile(args.input):
        # 处理单个文件
        result = process_single_image(
            args.input, 
            args.output, 
            face_detector, 
            emotion_analyzer, 
            pain_assessor, 
            psi_calculator
        )
        results = [result] if result else []
    else:
        # 处理目录
        results = process_directory(
            args.input, 
            args.output, 
            face_detector, 
            emotion_analyzer, 
            pain_assessor, 
            psi_calculator
        )
    
    # 保存结果
    if results:
        # 创建可序列化的结果
        serializable_results = []
        for r in results:
            if r:  # 检查结果是否有效
                sr = {
                    'image_path': r['image_path'],
                    'emotion_probs': {k: float(v) for k, v in r['emotion_probs'].items()},
                    'pain_score': float(r['pain_result']['pain_score']),
                    'pain_level_probs': {k: float(v) for k, v in r['pain_result']['pain_level_probs'].items()},
                    'psi': float(r['psi']),
                    'process_time': float(r['process_time'])
                }
                serializable_results.append(sr)
        
        # 保存JSON结果
        results_path = os.path.join(args.output, 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n成功处理 {len(results)} 张图像")
        print(f"详细结果已保存至 {results_path}")
        
        # 如果是批处理模式，生成汇总可视化
        if args.batch or len(results) > 1:
            plot_batch_results(results, args.output)
    else:
        print("未能成功处理任何图像")

if __name__ == "__main__":
    main()
