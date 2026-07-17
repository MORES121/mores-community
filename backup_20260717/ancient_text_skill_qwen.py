import sys
import os
import cv2
import json
import base64
import dashscope
from dashscope import MultiModalConversation
sys.path.insert(0, 'ancient_text')
from train_det_full import TextDetector
from evaluate_v29 import predict_single_image

# 千问API配置
DASHSCOPE_API_KEY = "sk-e8eda2b983714762a6ba49f8a7a2c238"
dashscope.api_key = DASHSCOPE_API_KEY

class AncientTextSkillQwen:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.detector = TextDetector()
        
    def detect(self, image_path):
        """检测文字区域"""
        result = predict_single_image(
            self.detector, 
            image_path, 
            threshold=self.threshold
        )
        return result
    
    def recognize_with_qwen(self, image_path, box=None):
        """调用千问API识别图片中的古文字"""
        # 如果指定了检测框，裁剪图片
        if box:
            x1, y1, x2, y2 = map(int, box)
            img = cv2.imread(image_path)
            crop = img[y1:y2, x1:x2]
            crop_path = "temp_crop.jpg"
            cv2.imwrite(crop_path, crop)
            target_path = crop_path
        else:
            target_path = image_path
        
        # 读取图片并转Base64
        with open(target_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # 清理临时文件
        if box and os.path.exists("temp_crop.jpg"):
            os.remove("temp_crop.jpg")
        
        # 构建请求
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{image_base64}"},
                    {"text": """请识别这张图片中的所有文字，严格按照以下格式输出：

【识别结果】
（逐行列出识别的古文字）

【现代汉字】
（逐行列出对应的现代汉字）

【解读】
（给出整体含义和背景解释）

如果图片中没有文字，请直接输出：未识别到文字。"""}
                ]
            }
        ]
        
        # 调用API
        try:
            response = MultiModalConversation.call(
                model="qwen-vl-plus",
                messages=messages
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content[0]["text"]
            else:
                return f"API调用失败: {response.status_code} - {response.message}"
        except Exception as e:
            return f"API异常: {str(e)}"
    
    def predict(self, image_path):
        """完整预测流程：检测 → 千问识别"""
        boxes = self.detect(image_path)
        
        if boxes:
            # 如果有检测框，对每个框进行识别
            results = []
            for i, box in enumerate(boxes):
                print(f"正在识别第 {i+1} 个文字区域...")
                text = self.recognize_with_qwen(image_path, box)
                results.append({
                    "box": box,
                    "text": text
                })
            return {
                "boxes": boxes,
                "results": results,
                "total_boxes": len(boxes)
            }
        else:
            # 如果没有检测框，整图识别
            print("未检测到文字区域，进行整图识别...")
            text = self.recognize_with_qwen(image_path, None)
            return {
                "boxes": [],
                "results": [{"box": "整图", "text": text}],
                "total_boxes": 0
            }

if __name__ == "__main__":
    skill = AncientTextSkillQwen(threshold=0.3)
    result = skill.predict("jiaguwentuopian.jpg")
    print(json.dumps(result, ensure_ascii=False, indent=2))