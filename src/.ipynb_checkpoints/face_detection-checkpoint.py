import cv2
import numpy as np
from deepface import DeepFace  # 仅导入主模块

class CustomFaceDetector:
    def __init__(self):
        """初始化人脸检测器"""
        self.detector_backend = "mtcnn"  # 可选: opencv, ssd, dlib, mtcnn, retinaface
    
    def detect_and_align(self, image_path=None, image=None):
        """检测并对齐人脸"""
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
            aligned_face = face_obj["face"]  # RGB格式的人脸图像
            
            # 关键修复：确保aligned_face是uint8类型
            if aligned_face.dtype != np.uint8:
                # 如果是浮点类型(如float64)，需要先归一化到0-255范围，再转为uint8
                if aligned_face.dtype == np.float64 or aligned_face.dtype == np.float32:
                    # 如果值范围在0-1之间，需要扩展到0-255
                    if np.max(aligned_face) <= 1.0:
                        aligned_face = (aligned_face * 255).astype(np.uint8)
                    else:
                        # 如果已经在0-255范围，直接转换
                        aligned_face = aligned_face.astype(np.uint8)
                else:
                    aligned_face = aligned_face.astype(np.uint8)
            
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