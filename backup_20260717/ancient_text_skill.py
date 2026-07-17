import sys
import os
import cv2
import json
import base64
import requests
sys.path.insert(0, 'ancient_text')
from train_det_full import TextDetector
from evaluate_v29 import predict_single_image

# 壁仞 API 配置
API_KEY = "sk-oCzZfSRljd6CFXSuOpf0i5Ls6jfemsRogq9LKdohqKgSpW0E"
API_URL = "https://fxb.supa.net.cn:6443/v1/chat/completions"

class AncientTextSkill:
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
    
    def recognize_with_api(self, image_path, box):
        """调用壁仞 API 识别检测框内的文字（支持图片）"""
        x1, y1, x2, y2 = map(int, box)
        
        # 裁剪检测框区域
        img = cv2.imread(image_path)
        crop = img[y1:y2, x1:x2]
        crop_path = "temp_crop.jpg"
        cv2.imwrite(crop_path, crop)
        
        # 将裁剪图片转为 base64
        with open(crop_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # 构建支持图片的消息
        prompt = """你是一位古文字专家。请识别这张图片中的文字。
要求：
1. 如果是甲骨文、金文或篆书，请识别并给出对应的现代汉字
2. 如果是现代文字，直接输出
3. 如果无法识别，请说明原因
4. 只输出识别结果，不要有其他解释"""
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 使用 OpenAI 兼容格式（支持图片）
        payload = {
            "model": "minimax-2.7",
            "messages": [
                {
                    "role": "system", 
                    "content": "你是古文字专家，擅长识别各种古代文字。"
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "识别失败")
            else:
                return f"[API调用失败: {response.status_code}] {response.text}"
        except Exception as e:
            return f"[API异常: {str(e)}]"
    
    def predict(self, image_path):
        """完整预测流程：检测 → API 识别"""
        boxes = self.detect(image_path)
        results = []
        
        for i, box in enumerate(boxes):
            print(f"正在识别第 {i+1} 个文字区域...")
            text = self.recognize_with_api(image_path, box)
            results.append({
                "box": box,
                "text": text
            })
        
        return {
            "boxes": boxes,
            "results": results,
            "total_boxes": len(boxes)
        }

if __name__ == "__main__":
    skill = AncientTextSkill(threshold=0.3)
    result = skill.predict("jiaguwentuopian.jpg")
    print(json.dumps(result, ensure_ascii=False, indent=2))