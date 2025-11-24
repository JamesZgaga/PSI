import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_results(image_path, aligned_face, emotion_probs, pain_result, psi, face_coords, output_path, psi_level=None):
    """可视化分析结果"""
    # 读取原始图像
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"⚠️  无法读取图像：{image_path}")
        return
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # 创建图形
    fig = plt.figure(figsize=(18, 10))
    
    # 设置网格布局
    gs = plt.GridSpec(2, 3, figure=fig)
    
    # 原始图像（仅显示图像，不绘制人脸框）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_img)
    ax1.set_title("原始图像", fontsize=12)
    ax1.axis('off')  # 关闭坐标轴
    
    # 显示对齐后的人脸
    ax2 = fig.add_subplot(gs[0, 1])
    if aligned_face is not None:
        # 处理aligned_face可能的格式（灰度图转RGB）
        if len(aligned_face.shape) == 2:
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_GRAY2RGB)
        else:
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        ax2.imshow(aligned_face)
        ax2.set_title("检测到的人脸", fontsize=12)
    else:
        ax2.text(0.5, 0.5, "无对齐人脸", ha='center', va='center', transform=ax2.transAxes)
    ax2.axis('off')
    
    # 情绪概率条形图
    ax3 = fig.add_subplot(gs[0, 2])
    if emotion_probs and isinstance(emotion_probs, dict):
        emotions = list(emotion_probs.keys())
        probs = list(emotion_probs.values())
        
        # 定义负面情绪（与config.yaml一致）
        negative_emotions = ['Fear', 'Disgust', 'Sadness', 'Anger']
        colors = ['red' if e in negative_emotions else 'green' for e in emotions]
        
        # 水平条形图以便标签更清晰
        y_pos = np.arange(len(emotions))
        ax3.barh(y_pos, probs, color=colors)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(emotions)
        ax3.set_title("情绪概率分布", fontsize=12)
        ax3.set_xlim(0, 1.0)
        ax3.set_xlabel("概率")
    else:
        ax3.text(0.5, 0.5, "无情绪数据", ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')
    
    # 痛苦等级概率饼图
    ax4 = fig.add_subplot(gs[1, 0])
    if pain_result and isinstance(pain_result, dict) and 'pain_level_probs' in pain_result:
        pain_level_probs = pain_result['pain_level_probs']
        labels = list(pain_level_probs.keys())
        sizes = list(pain_level_probs.values())
        
        # 定义颜色映射，从绿色(LV1)到红色(LV5)
        colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red'][:len(labels)]
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.axis('equal')
        ax4.set_title(f"痛苦等级分布 (分数: {pain_result['pain_score']:.2f})", fontsize=12)
    else:
        ax4.text(0.5, 0.5, "无痛苦数据", ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')
    
    # PSI仪表盘（适配0-100的PSI范围）
    ax5 = fig.add_subplot(gs[1, 1:])
    if isinstance(psi, (int, float)):
        psi = max(0, min(100, psi))  # 强制限制在0-100
        # 定义PSI值对应的颜色、风险级别（与config.yaml一致）
        if psi <= 30:
            psi_color = 'green'
            risk_level = '良好'
        elif psi <= 60:
            psi_color = 'yellow'
            risk_level = '中等'
        else:
            psi_color = 'red'
            risk_level = '较差'
        
        # 绘制半圆仪表盘
        theta = np.linspace(0, np.pi, 100)
        r = 1.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax5.plot(x, y, color='black')
        
        # 绘制PSI指针
        psi_theta = np.pi * (psi / 100.0)
        x_psi = r * np.cos(psi_theta)
        y_psi = r * np.sin(psi_theta)
        ax5.plot([0, x_psi], [0, y_psi], color=psi_color, linewidth=3)
        
        # 添加标签
        display_level = psi_level if psi_level and isinstance(psi_level, str) else risk_level
        ax5.text(0, -0.2, f"心理状态指数(PSI): {psi}", ha='center', fontsize=14, fontweight='bold')
        ax5.text(0, -0.4, f"状态等级: {display_level}", ha='center', fontsize=12, color=psi_color)
        
        # 添加刻度
        tick_labels = ['0', '25', '50', '75', '100']
        for i in range(5):
            t = i * np.pi / 4
            xt = 1.1 * r * np.cos(t)
            yt = 1.1 * r * np.sin(t)
            ax5.text(xt, yt, tick_labels[i], ha='center', va='center')
    else:
        ax5.text(0.5, 0.5, "PSI数据无效", ha='center', va='center', transform=ax5.transAxes)
    
    ax5.set_aspect('equal')
    ax5.axis('off')
    
    # 调整布局并保存
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化结果已保存至 {output_path}")

def plot_batch_results(results, output_dir):
    """可视化批处理结果（适配0-100的PSI范围）"""
    # 提取有效PSI值
    psi_values = []
    for r in results:
        if isinstance(r, dict) and 'psi' in r and isinstance(r['psi'], (int, float)):
            psi_values.append(max(0, min(100, r['psi'])))  # 强制限制在0-100
    
    if not psi_values:
        print("⚠️  没有有效的PSI值可供可视化")
        return
    
    # 绘制PSI分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(psi_values, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(psi_values), color='red', linestyle='dashed', linewidth=2, 
                label=f'平均值: {np.mean(psi_values):.1f}')
    plt.axvline(np.median(psi_values), color='green', linestyle='dashed', linewidth=2, 
                label=f'中位数: {np.median(psi_values):.1f}')
    # 添加等级阈值线（与config.yaml一致）
    plt.axvline(30, color='green', linestyle='solid', linewidth=1, alpha=0.5, label='良好阈值(30)')
    plt.axvline(60, color='orange', linestyle='solid', linewidth=1, alpha=0.5, label='中等阈值(60)')
    
    plt.xlabel('心理状态指数(PSI)', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.title('PSI分布直方图（0-100分）', fontsize=14)
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 保存图表
    hist_path = os.path.join(output_dir, 'psi_distribution.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ PSI分布直方图已保存至 {hist_path}")
    
    # 计算统计数据
    psi_mean = np.mean(psi_values)
    psi_median = np.median(psi_values)
    psi_min = np.min(psi_values)
    psi_max = np.max(psi_values)
    psi_std = np.std(psi_values)
    
    # 保存统计数据
    stats_path = os.path.join(output_dir, 'psi_statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("心理状态指数(PSI)统计信息\n")
        f.write("====================\n")
        f.write(f"样本数: {len(psi_values)}\n")
        f.write(f"平均值: {psi_mean:.1f}\n")
        f.write(f"中位数: {psi_median:.1f}\n")
        f.write(f"最小值: {psi_min:.1f}\n")
        f.write(f"最大值: {psi_max:.1f}\n")
        f.write(f"标准差: {psi_std:.1f}\n")
        f.write(f"\n等级划分标准：\n")
        f.write(f"良好：PSI ≤ 30\n")
        f.write(f"中等：31 ≤ PSI ≤ 60\n")
        f.write(f"较差：PSI > 60\n")
    
    print(f"✅ PSI统计信息已保存至 {stats_path}")
