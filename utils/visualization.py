import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def visualize_results(image_path, aligned_face, emotion_probs, pain_result, psi, face_coords, output_path):
    """可视化分析结果"""
    # 读取原始图像
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # 创建图形
    fig = plt.figure(figsize=(18, 10))
    
    # 设置网格布局
    gs = plt.GridSpec(2, 3, figure=fig)
    
    # 原始图像和人脸框
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_img)
    ax1.set_title("原始图像", fontsize=12)
    
    if face_coords:
        x, y, w, h = face_coords
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
    
    ax1.axis('off')
    
    # 显示对齐后的人脸
    if aligned_face is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
        ax2.set_title("检测到的人脸", fontsize=12)
        ax2.axis('off')
    
    # 情绪概率条形图
    ax3 = fig.add_subplot(gs[0, 2])
    if emotion_probs:
        emotions = list(emotion_probs.keys())
        probs = list(emotion_probs.values())
        
        # 定义负面情绪
        negative_emotions = ['Fear', 'Disgust', 'Sadness', 'Anger']
        
        # 为负面情绪使用红色，正面或中性情绪使用绿色
        colors = ['red' if e in negative_emotions else 'green' for e in emotions]
        
        # 水平条形图以便标签更清晰
        y_pos = np.arange(len(emotions))
        ax3.barh(y_pos, probs, color=colors)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(emotions)
        ax3.set_title("情绪概率分布", fontsize=12)
        ax3.set_xlim(0, 1.0)
        ax3.set_xlabel("概率")
    
    # 痛苦等级概率饼图
    ax4 = fig.add_subplot(gs[1, 0])
    if pain_result and 'pain_level_probs' in pain_result:
        pain_level_probs = pain_result['pain_level_probs']
        labels = list(pain_level_probs.keys())
        sizes = list(pain_level_probs.values())
        
        # 定义颜色映射，从绿色(LV1)到红色(LV5)
        colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red'][:len(labels)]
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.axis('equal')
        ax4.set_title(f"痛苦等级分布 (分数: {pain_result['pain_score']:.2f})", fontsize=12)
    # PSI仪表盘
    ax5 = fig.add_subplot(gs[1, 1:])
    
    # 创建一个简单的仪表盘效果
    # 定义PSI值对应的颜色和风险级别
    if psi < 0.2:
        psi_color = 'green'
        risk_level = '很低'
    elif psi < 0.4:
        psi_color = 'yellowgreen'
        risk_level = '低'
    elif psi < 0.6:
        psi_color = 'yellow'
        risk_level = '中等'
    elif psi < 0.8:
        psi_color = 'orange'
        risk_level = '高'
    else:
        psi_color = 'red'
        risk_level = '很高'
    
    # 绘制半圆仪表盘
    theta = np.linspace(0, np.pi, 100)
    r = 1.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax5.plot(x, y, color='black')
    
    # 绘制PSI值对应的指针
    psi_theta = np.pi * psi
    x_psi = r * np.cos(psi_theta)
    y_psi = r * np.sin(psi_theta)
    ax5.plot([0, x_psi], [0, y_psi], color=psi_color, linewidth=3)
    
    # 添加PSI值标签
    ax5.text(0, -0.2, f"心理状态指数(PSI): {psi:.2f}", ha='center', fontsize=14, fontweight='bold')
    ax5.text(0, -0.4, f"风险级别: {risk_level}", ha='center', fontsize=12, color=psi_color)
    
    # 添加刻度
    for i in range(5):
        t = i * np.pi / 4
        xt = 1.1 * r * np.cos(t)
        yt = 1.1 * r * np.sin(t)
        ax5.text(xt, yt, f"{i*0.25:.1f}", ha='center', va='center')
    
    ax5.set_aspect('equal')
    ax5.axis('off')
    
    # 调整布局并保存
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"可视化结果已保存至 {output_path}")

def plot_batch_results(results, output_dir):
    """可视化批处理结果"""
    # 提取PSI值
    psi_values = [r['psi'] for r in results if 'psi' in r]
    
    if not psi_values:
        print("没有有效的PSI值可供可视化")
        return
    
    # 绘制PSI分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(psi_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(psi_values), color='red', linestyle='dashed', linewidth=2, 
                label=f'平均值: {np.mean(psi_values):.2f}')
    plt.axvline(np.median(psi_values), color='green', linestyle='dashed', linewidth=2, 
                label=f'中位数: {np.median(psi_values):.2f}')
    plt.xlabel('心理状态指数(PSI)', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.title('PSI分布直方图', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 保存图表
    hist_path = os.path.join(output_dir, 'psi_distribution.png')
    plt.savefig(hist_path, dpi=150)
    plt.close()
    
    print(f"PSI分布直方图已保存至 {hist_path}")
    
    # 计算统计数据
    psi_mean = np.mean(psi_values)
    psi_median = np.median(psi_values)
    psi_min = np.min(psi_values)
    psi_max = np.max(psi_values)
    psi_std = np.std(psi_values)
    
    # 保存统计数据
    stats_path = os.path.join(output_dir, 'psi_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("心理状态指数(PSI)统计信息\n")
        f.write("====================\n")
        f.write(f"样本数: {len(psi_values)}\n")
        f.write(f"平均值: {psi_mean:.4f}\n")
        f.write(f"中位数: {psi_median:.4f}\n")
        f.write(f"最小值: {psi_min:.4f}\n")
        f.write(f"最大值: {psi_max:.4f}\n")
        f.write(f"标准差: {psi_std:.4f}\n")
    
    print(f"PSI统计信息已保存至 {stats_path}")
