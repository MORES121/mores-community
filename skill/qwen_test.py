import dashscope
import base64
from dashscope import MultiModalConversation

# 配置API Key
dashscope.api_key = "sk-e8eda2b983714762a6ba49f8a7a2c238"

def recognize_image(image_path):
    """调用千问多模态API识别图片中的文字"""
    # 读取图片并转Base64
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # 构建请求
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{image_base64}"},
                {"text": "请识别这张图片中的所有文字，如果包含甲骨文或古文字，请给出对应的现代汉字。"}
            ]
        }
    ]
    
    # 调用API
    response = MultiModalConversation.call(
        model="qwen-vl-plus",  # 或 qwen-vl-max
        messages=messages
    )
    
    # 解析结果
    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    else:
        return f"API调用失败: {response.status_code} - {response.message}"

if __name__ == "__main__":
    result = recognize_image("jiaguwentuopian.jpg")
    print("识别结果:", result)