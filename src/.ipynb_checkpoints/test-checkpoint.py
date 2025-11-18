import os
import yaml
from PIL import Image
from collections import defaultdict

def load_yaml_config(config_path):
    """加载loopy.yaml配置文件，获取类别信息"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # 提取类别名称（确保与配置文件一致）
        class_names = config.get('names', [])
        num_classes = config.get('nc', len(class_names))
        print(f"成功加载配置文件：{config_path}")
        print(f"类别数量：{num_classes}")
        print(f"类别名称：{class_names}\n")
        return class_names
    except Exception as e:
        print(f"加载配置文件失败：{e}")
        return []

def get_file_list(folder, exts=('jpg', 'jpeg', 'png', 'bmp')):
    """获取指定文件夹下特定格式的文件列表（不含后缀）"""
    if not os.path.exists(folder):
        print(f"文件夹不存在：{folder}")
        return []
    # 只保留指定格式的文件，并提取文件名（不含后缀）
    file_list = []
    for file in os.listdir(folder):
        name, ext = os.path.splitext(file)
        if ext.lower().lstrip('.') in exts:
            file_list.append(name)
    return sorted(file_list)

def count_class_distribution(labels_folder, class_names):
    """统计标签文件中的类别分布"""
    class_counts = defaultdict(int)
    invalid_files = []  # 记录格式错误的标签文件
    
    if not os.path.exists(labels_folder):
        print(f"标签文件夹不存在：{labels_folder}")
        return class_counts, invalid_files
    
    # 遍历所有标签文件（假设标签为YOLO格式的txt文件）
    for label_file in os.listdir(labels_folder):
        if not label_file.endswith('.txt'):
            continue  # 只处理txt标签文件
        label_path = os.path.join(labels_folder, label_file)
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                # 假设每个图像对应一个表情（每行一个目标，取第一行的类别）
                if lines:
                    first_line = lines[0].strip().split()
                    if len(first_line) >= 1:  # 至少包含类别ID
                        class_id = int(first_line[0])
                        # 检查类别ID是否在有效范围内
                        if 0 <= class_id < len(class_names):
                            class_name = class_names[class_id]
                            class_counts[class_name] += 1
                        else:
                            invalid_files.append(f"{label_file}（无效类别ID：{class_id}）")
                    else:
                        invalid_files.append(f"{label_file}（标签格式错误，无内容）")
                else:
                    invalid_files.append(f"{label_file}（标签文件为空）")
        except Exception as e:
            invalid_files.append(f"{label_file}（读取错误：{str(e)}）")
    
    return class_counts, invalid_files

def check_image_sizes(image_folder, sample_size=100):
    """抽样检查图像尺寸分布（避免全量检查耗时）"""
    if not os.path.exists(image_folder):
        return {}
    image_sizes = defaultdict(int)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    # 抽样（最多检查sample_size个图像）
    sample_files = image_files[:sample_size]
    for img_file in sample_files:
        img_path = os.path.join(image_folder, img_file)
        try:
            with Image.open(img_path) as img:
                size = img.size  # (width, height)
                image_sizes[size] += 1
        except Exception as e:
            print(f"读取图像失败（{img_file}）：{e}")
    return image_sizes

def analyze_dataset(data_type, root_dir, class_names):
    """分析单个数据集（训练集/测试集）"""
    print(f"\n===== 开始分析 {data_type} 集 =====")
    # 定义图像和标签路径（根据实际目录结构：train/image 和 train/labels）
    image_dir = os.path.join(root_dir, data_type, 'image')
    labels_dir = os.path.join(root_dir, data_type, 'labels')
    
    # 1. 获取图像和标签文件列表（不含后缀）
    image_names = get_file_list(image_dir)
    label_names = get_file_list(labels_dir, exts=('txt',))  # 标签是txt文件
    print(f"图像文件数量：{len(image_names)}（路径：{image_dir}）")
    print(f"标签文件数量：{len(label_names)}（路径：{labels_dir}）")
    
    # 2. 检查图像与标签是否一一对应
    images_set = set(image_names)
    labels_set = set(label_names)
    # 只有图像没有标签的文件
    missing_labels = images_set - labels_set
    # 只有标签没有图像的文件
    missing_images = labels_set - images_set
    print(f"缺失标签的图像数量：{len(missing_labels)}（示例：{list(missing_labels)[:5]}）")
    print(f"缺失图像的标签数量：{len(missing_images)}（示例：{list(missing_images)[:5]}）")
    
    # 3. 统计类别分布
    class_counts, invalid_files = count_class_distribution(labels_dir, class_names)
    print("\n类别分布：")
    for class_name in class_names:  # 按配置文件顺序输出
        print(f"  {class_name}: {class_counts.get(class_name, 0)}")
    if invalid_files:
        print(f"\n无效标签文件（共{len(invalid_files)}个）：{invalid_files[:5]}...")  # 只显示前5个
    
    # 4. 抽样检查图像尺寸
    image_sizes = check_image_sizes(image_dir)
    if image_sizes:
        print("\n抽样图像尺寸分布（最多100张）：")
        for size, count in image_sizes.items():
            print(f"  尺寸 {size}（宽x高）：{count} 张")
    
    print(f"===== {data_type} 集分析结束 =====\n")

if __name__ == "__main__":
    # 数据集根目录（根据你的路径修改，当前为/mnt/MCP/Deepface/data/expression）
    dataset_root = "/mnt/MCP/Deepface/data/expression"
    # 配置文件路径
    config_path = os.path.join(dataset_root, "loopy.yaml")
    
    # 1. 加载配置文件获取类别信息
    class_names = load_yaml_config(config_path)
    if not class_names:
        print("无法获取类别信息，退出分析")
        exit(1)
    
    # 2. 分别分析训练集和测试集
    analyze_dataset("train", dataset_root, class_names)
    analyze_dataset("test", dataset_root, class_names)
    
    print("数据集分析完成！")
    