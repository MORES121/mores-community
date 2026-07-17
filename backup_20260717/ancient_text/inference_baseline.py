import json
import random
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR

random.seed(42)
np.random.seed(42)

def main():
    input_dir = Path("/saisdata/13/eval/images")
    output_file = Path("/saisresult/prediction.json")

    ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True, show_log=False)

    results = {}
    for img_path in sorted(input_dir.glob("*.png")):
        image_id = img_path.stem
        result = ocr.ocr(str(img_path), cls=True)

        detections = []
        if result and result[0]:
            for line in result[0]:
                bbox = line[0]
                text = line[1][0]
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x = int(min(x_coords))
                y = int(min(y_coords))
                w = int(max(x_coords) - x)
                h = int(max(y_coords) - y)
                detections.append({"bbox": [x, y, w, h], "text": text})
        results[image_id] = detections

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()