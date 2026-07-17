import sys
import os
import cv2
import json
import base64
import numpy as np
import dashscope
from dashscope import MultiModalConversation
sys.path.insert(0, 'ancient_text')
from train_det_full import TextDetector
from evaluate_v29 import predict_single_image

dashscope.api_key = 'sk-e8eda2b983714762a6ba49f8a7a2c238'

class AncientTextSkillQwen:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.detector = TextDetector()
    
    def detect(self, image_path):
        result = predict_single_image(
            self.detector, 
            image_path, 
            threshold=self.threshold
        )
        return result
    
    def enhance_image(self, image_path):
        """增强图像对比度和锐度，用于天文图等细节较弱的图片"""
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        # 增加对比度
        enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=10)
        
        # 锐化
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        enhanced_path = image_path.replace('.', '_enhanced.')
        cv2.imwrite(enhanced_path, enhanced)
        return enhanced_path
    
    def recognize_with_context(self, image_path, box):
        """上下文感知识别：同时考虑天文图和古文字"""
        x1, y1, x2, y2 = map(int, box)
        img = cv2.imread(image_path)
        crop = img[y1:y2, x1:x2]
        crop_path = "temp_context.jpg"
        cv2.imwrite(crop_path, crop)
        
        with open(crop_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        os.remove(crop_path)
        
        prompt = """请分析这张图片的内容，它可能是古文字（甲骨文/金文/篆书），也可能是天文图（星宿/天象）。

请按照以下格式输出：

【类型判断】
（输出：古文字 或 天文图）

【内容识别】
- 如果是古文字：逐行列出识别的古文字，并给出对应的现代汉字
- 如果是天文图：严格按照以下格式列出星宿信息
  ★ 星宿名称：（列出图中出现的星宿，如：北斗七星、紫微星、天狼星等）
  ★ 星官分类：（如：三垣、二十八宿、四象等）
  ★ 天象布局：（描述星宿在画面中的相对位置和排列）

【解读】
- 如果是古文字：给出历史背景解读
- 如果是天文图：给出天文学背景解读，包括：
  ★ 该星象在古代天文学中的意义
  ★ 可能对应的历史时期或文献记载
  ★ 与现代天文学的对应关系

如果无法识别，请输出：无法识别。"""
        
        messages = [{
            'role': 'user',
            'content': [
                {'image': f'data:image/jpeg;base64,{img_base64}'},
                {'text': prompt}
            ]
        }]
        response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
        return response.output.choices[0].message.content[0]['text']
    
    def predict(self, image_path, enhance=False):
        """完整预测流程：检测 → 上下文感知识别"""
        # 可选：增强图像
        if enhance:
            image_path = self.enhance_image(image_path)
        
        boxes = self.detect(image_path)
        if not boxes:
            return {"error": "未检测到文字区域", "total_boxes": 0}
        
        results = []
        for box in boxes:
            text = self.recognize_with_context(image_path, box)
            results.append({"box": box, "text": text})
        
        return {
            "boxes": boxes,
            "results": results,
            "total_boxes": len(boxes)
        }

if __name__ == "__main__":
    skill = AncientTextSkillQwen(threshold=0.3)
    
    print("=== 天文图识别（上下文感知）===")
    result = skill.predict("Screenshot_20260424_092039_edit_108043313304346.jpg", enhance=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n=== 甲骨文识别（上下文感知）===")
    result = skill.predict("jiaguwentuopian.jpg")
    print(json.dumps(result, ensure_ascii=False, indent=2))