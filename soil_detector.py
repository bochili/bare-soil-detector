from ultralytics import YOLO
from typing import List, Dict, Any
import os
import uuid
import time

class SoilDetector:
    def __init__(self, model_path: str = "best.pt", conf_threshold: float = 0.5, output_dir: str = "output"):
        """
        初始化裸土检测器
        
        Args:
            model_path: 模型路径
            conf_threshold: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.output_dir = output_dir
    def detect(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        检测多个图片中的裸土
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            结果列表，每个元素包含:
            - image_path: 原始图片路径
            - has_soil: 是否检测到裸土
            - soil_confidence: 裸土置信度(如未检测到则为0)
            - result_path: 结果图片路径
        """
        results = []
        
        # 创建保存目录 - 使用时间戳确保唯一性
        timestamp = int(time.time())
        save_dir = os.path.join(self.output_dir, f"predict_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        for img_path in image_paths:
            # 运行检测
            model_results = self.model(img_path, conf=self.conf_threshold)
            
            # 分析结果
            has_soil = False
            soil_confidence = 0.0
            
            for result in model_results:
                # 获取类别名称和置信度
                names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
                confs = result.boxes.conf
                
                # 检查是否有裸土类别
                for i, name in enumerate(names):
                    if name.lower() == 'soil':
                        has_soil = True
                        # 如果有多个裸土检测，取最高置信度
                        if confs[i] > soil_confidence:
                            soil_confidence = confs[i].item()
            
            # 生成唯一文件名
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            unique_name = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
            result_path = os.path.join(save_dir, unique_name)
            
            # 保存检测结果图片
            # 调用save方法保存结果，并获取实际保存路径
            model_results[0].save(filename=result_path)[0]
            
            results.append({
                "image_path": img_path,
                "has_soil": has_soil,
                "soil_confidence": soil_confidence,
                "result_path": result_path
            })
        
        return results


# 使用示例
if __name__ == "__main__":
    detector = SoilDetector(model_path="train_result/weights/best.pt", conf_threshold=0.3, output_dir="output")
    
    # 示例图片路径列表
    image_paths = [
        "test_imgs/1.jpg",
        "test_imgs/2.jpg",
        "test_imgs/3.jpg",
        "test_imgs/4.jpg",
        "test_imgs/5.jpg",
        "test_imgs/6.jpg",
        "test_imgs/7.jpg"
    ]
    
    # 检测裸土
    detection_results = detector.detect(image_paths)
    print(detection_results)
    # 输出结果
    for result in detection_results:
        if result["has_soil"]:
            print(f"图片 {result['image_path']} 中检测到裸土，置信度: {result['soil_confidence']:.2f}")
        else:
            print(f"图片 {result['image_path']} 中未检测到裸土")
        print(f"结果图片保存在: {result['result_path']}")
        print("-" * 30)