import cv2
import numpy as np
from deepface import DeepFace
# from deepface.models.face_detection.RetinaFace import RetinaFace 
class FaceDetector:
    def __init__(self):
        """初始化人脸检测器（使用 Deepface 高层 API）"""
        # 可指定检测器后端为 retinaface（保持与原逻辑一致）
        self.detector_backend = "mtcnn"
    
    def detect_faces(self, image_path=None, image=None):
        """检测图像中的人脸（基于 DeepFace.detect_faces()）"""
        try:
            if image is not None:
                # 转换为 RGB 格式（Deepface 要求）
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 调用高层 API 检测人脸，指定检测器
                result = DeepFace.detect_faces(img_path=image_rgb, detector_backend=self.detector_backend)
            else:
                # 从路径检测
                result = DeepFace.detect_faces(img_path=image_path, detector_backend=self.detector_backend)
            
            # 解析结果（Deepface 返回的是字典，键为图像路径，值为检测到的人脸列表）
            # 提取人脸列表（忽略键名，取第一个值）
            faces = next(iter(result.values())) if result else []
            return faces
        
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def detect_and_align(self, image_path=None, image=None):
        """检测并对齐人脸（逻辑保持不变，适配新的检测结果格式）"""
        # 优先使用传入的图像数据
        if image is not None:
            img = image.copy()
        else:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                return None, None
        
        # 检测人脸（使用重写后的 detect_faces 方法）
        faces = self.detect_faces(image=img)
        
        if len(faces) == 0:
            print("No face detected")
            return None, None
        
        # 取第一个检测到的人脸（Deepface 返回的人脸信息中，边界框键为 'facial_area'）
        face_obj = faces[0]
        face_coords = face_obj['facial_area']  # 格式：{'x': int, 'y': int, 'w': int, 'h': int}
        
        # 后续的裁剪和对齐逻辑保持不变...
        img_h, img_w = img.shape[:2]
        margin_x = int(face_coords['w'] * 0.1)
        margin_y = int(face_coords['h'] * 0.1)
        x1 = max(0, face_coords['x'] - margin_x)
        y1 = max(0, face_coords['y'] - margin_y)
        x2 = min(img_w, face_coords['x'] + face_coords['w'] + margin_x)
        y2 = min(img_h, face_coords['y'] + face_coords['h'] + margin_y)
        
        detected_face = img[y1:y2, x1:x2]
        
        # 对齐人脸（使用 Deepface 内置的对齐方法）
        try:
            # 注意：DeepFace.alignment.align_face 可能需要 RGB 格式输入
            detected_face_rgb = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
            aligned_face = DeepFace.alignment.align_face(img=detected_face_rgb, detector_backend='skip')
            # 转回 BGR 格式（与 OpenCV 兼容）
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error in face alignment: {e}")
            aligned_face = detected_face
        
        return aligned_face, (x1, y1, x2-x1, y2-y1)