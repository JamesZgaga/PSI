import os
import cv2
import numpy as np
import sys
from deepface import DeepFace
import traceback

def test_image_properties(image_path):
    """测试图像基本属性"""
    print("\n=== 测试图像属性 ===")
    try:
        # 使用OpenCV加载图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法加载图像: {image_path}")
            return False
        
        print(f"图像尺寸: {img.shape}")
        print(f"图像类型: {img.dtype}")
        print(f"像素值范围: 最小={np.min(img)}, 最大={np.max(img)}")
        
        return True
    except Exception as e:
        print(f"测试图像属性出错: {e}")
        return False

def test_deepface_direct(image_path):
    """直接使用DeepFace测试人脸检测"""
    print("\n=== 测试DeepFace直接检测 ===")
    try:
        # 使用原始图像路径
        print("尝试使用图像路径:")
        try:
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend="mtcnn",
                enforce_detection=False
            )
            print(f"DeepFace检测到 {len(faces)} 个人脸")
        except Exception as e:
            print(f"使用图像路径时出错: {e}")
        
        # 使用numpy数组
        print("\n尝试使用numpy数组:")
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法加载图像: {image_path}")
            return False
            
        # 确保图像是uint8类型
        if img.dtype != np.uint8:
            print(f"将图像从 {img.dtype} 转换为 uint8")
            img = img.astype(np.uint8)
            
        # RGB转换(DeepFace需要RGB格式)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            faces = DeepFace.extract_faces(
                img_path=img_rgb,
                detector_backend="mtcnn",
                enforce_detection=False
            )
            print(f"DeepFace检测到 {len(faces)} 个人脸")
            if len(faces) > 0:
                print("检测成功!")
                return True
        except Exception as e:
            print(f"使用numpy数组时出错: {e}")
            
        return False
    except Exception as e:
        print(f"DeepFace测试失败: {e}")
        traceback.print_exc()
        return False

def test_custom_detector(image_path):
    """测试并修复CustomFaceDetector"""
    print("\n=== 测试修改后的CustomFaceDetector ===")
    
    # 定义修复后的检测方法
    def fixed_detect_and_align(image_path):
        """修复后的人脸检测与对齐函数"""
        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法加载图像: {image_path}")
            return None, None
            
        # 确保图像是uint8类型
        if img.dtype != np.uint8:
            print(f"将图像从 {img.dtype} 转换为 uint8")
            img = img.astype(np.uint8)
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            faces = DeepFace.extract_faces(
                img_path=img_rgb,  # 注意：使用RGB格式的numpy数组
                detector_backend="mtcnn",
                enforce_detection=False,  # 不强制检测
                align=True
            )
            
            if len(faces) == 0:
                print("未检测到人脸")
                return None, None
                
            face_obj = faces[0]
            facial_area = face_obj["facial_area"]
            aligned_face = face_obj["face"]  # RGB格式
            
            # 转换回BGR格式用于OpenCV
            aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
            
            # 获取面部坐标
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
            
            print(f"成功检测并对齐人脸，面部区域: x={x}, y={y}, w={w}, h={h}")
            return aligned_face_bgr, (x, y, w, h)
            
        except Exception as e:
            print(f"人脸检测与对齐出错: {e}")
            traceback.print_exc()
            return None, None
    
    # 测试修复后的函数
    try:
        aligned_face, face_coords = fixed_detect_and_align(image_path)
        if aligned_face is not None:
            print("修复后的检测器成功!")
            print(f"对齐后的人脸尺寸: {aligned_face.shape}, 类型: {aligned_face.dtype}")
            
            # 保存调试图像
            output_path = "debug_aligned_face.jpg"
            cv2.imwrite(output_path, aligned_face)
            print(f"已保存调试图像到: {output_path}")
            return True
        else:
            print("修复后的检测器仍然失败")
            return False
    except Exception as e:
        print(f"修复测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python debug_face_detection.py <图像路径>")
        return
        
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return
        
    print(f"测试图像: {image_path}")
    
    # 运行测试
    test_image_properties(image_path)
    test_deepface_direct(image_path)
    test_custom_detector(image_path)
    
    print("\n=== 解决方案 ===")
    print("根据测试结果，需要修改CustomFaceDetector类的detect_and_align方法:")
    print("1. 确保图像始终为uint8类型")
    print("2. 正确处理RGB和BGR格式转换")
    print("3. 避免在使用numpy数组时将其再次读取为图像路径")
    print("\n下面是修复后的CustomFaceDetector类:")
    
    fix_code = """
class CustomFaceDetector:
    def __init__(self):
        # 初始化人脸检测器
        self.detector_backend = "mtcnn"  # 可选: opencv, ssd, dlib, mtcnn, retinaface
    
    def detect_and_align(self, image_path=None, image=None):
        # 检测并对齐人脸
        # 加载图像
        if image is not None:
            img = image.copy()
        else:
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法加载图像: {image_path}")
                return None, None
        
        # 确保图像是uint8类型
        if img.dtype != np.uint8:
            print(f"将图像从 {img.dtype} 转换为 uint8")
            img = img.astype(np.uint8)
            
        # 转换为RGB格式(DeepFace内部使用RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            faces = DeepFace.extract_faces(
                img_path=img_rgb,  # 传入RGB格式的numpy数组
                detector_backend=self.detector_backend,
                enforce_detection=False,  # 更宽容的检测设置
                align=True
            )
            
            if len(faces) == 0:
                print("未检测到人脸")
                return None, None
            
            # 获取第一个检测到的人脸
            face_obj = faces[0]
            facial_area = face_obj["facial_area"]
            aligned_face = face_obj["face"]  # RGB格式
            
            # 转换回BGR格式
            aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
            
            # 获取坐标
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
            
            # 扩大边界框（可选）
            img_h, img_w = img.shape[:2]
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img_w, x + w + margin_x)
            y2 = min(img_h, y + h + margin_y)
            
            return aligned_face_bgr, (x1, y1, x2-x1, y2-y1)
            
        except Exception as e:
            print(f"人脸检测与对齐出错: {e}")
            return None, None
    """

    print(fix_code)

if __name__ == "__main__":
    main()
