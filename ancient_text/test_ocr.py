from paddleocr import PaddleOCR
import os

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['GLOG_minloglevel'] = '3'

print('初始化 OCR...')
ocr = PaddleOCR()
print('初始化成功')

img_path = r'C:\test_saisdata\13\eval\images\0032.png'
print(f'\n识别图片: {img_path}')
result = ocr.ocr(img_path)

print('\n===== 识别结果 =====')
if result and result[0]:
    for line in result[0]:
        print(f'文字: {line[1][0]} , 置信度: {line[1][1]:.4f}')
else:
    print('未识别到文字')