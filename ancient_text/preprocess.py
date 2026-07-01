# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_xml_to_yolo(xml_path, output_dir):
    """将 XML 标注转换为 YOLO 格式"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获取图片尺寸
    width = int(root.get('width', 0))
    height = int(root.get('height', 0))
    
    img_id = root.get('id', '').replace('.jpg', '')
    
    annotations = []
    for text in root.findall('.//text'):
        for row in text.findall('.//text_row'):
            for chunk in row.findall('.//chunk'):
                for char in chunk.findall('.//char'):
                    # 获取字符和位置
                    char_text = char.text
                    pos = char.get('position', '').split(',')
                    if len(pos) == 4:
                        x1, y1, x2, y2 = map(int, pos)
                        # 转换为 YOLO 格式（中心点 + 宽高，归一化）
                        x_center = (x1 + x2) / 2 / width
                        y_center = (y1 + y2) / 2 / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height
                        annotations.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    
    # 保存标注文件
    if annotations:
        with open(output_dir / f"{img_id}.txt", 'w') as f:
            f.write('\n'.join(annotations))
    
    return len(annotations)

def main():
    # 数据路径
    xml_dir = Path(r"C:\Users\klidw\Downloads\train\train\out_of_domain")
    output_dir = Path(r"C:\mores_fusion\labels")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换所有 XML
    total = 0
    for xml_file in xml_dir.glob("*.xml"):
        try:
            count = convert_xml_to_yolo(xml_file, output_dir)
            total += count
            print(f"处理: {xml_file.name} -> {count} 个字符")
        except Exception as e:
            print(f"错误: {xml_file.name} - {e}")
    
    print(f"\n总计转换字符数: {total}")

if __name__ == '__main__':
    main()