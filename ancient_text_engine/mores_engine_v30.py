"""
墨睿思 MORES 古文字识别引擎 v30
可控 · 可解释 · 可决策

架构设计：
1. 可控层：检测参数可调、规则可配置
2. 可解释层：记录决策路径、输出可视化
3. 可决策层：多候选结果、置信度排序
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# ========== 数据结构定义 ==========

@dataclass
class DetectionBox:
    """检测框数据"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 0.0
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class RecognitionCandidate:
    """识别候选结果"""
    character: str          # 字符
    confidence: float       # 置信度
    reasoning_path: str     # 决策路径（可解释）


@dataclass
class EngineOutput:
    """引擎最终输出"""
    boxes: List[DetectionBox]                    # 检测框
    candidates: List[List[RecognitionCandidate]] # 每个框的候选字符
    decision_path: List[str]                     # 决策路径记录
    visualization: Optional[np.ndarray] = None   # 可视化图


# ========== 配置类（可控参数） ==========

@dataclass
class DetectionConfig:
    """检测器配置 - 可控参数"""
    prob_threshold: float = 0.3      # 概率阈值
    nms_threshold: float = 0.5       # NMS 阈值
    min_area: int = 20               # 最小面积
    max_aspect_ratio: float = 3.0    # 最大宽高比
    min_confidence: float = 0.2      # 最低置信度


@dataclass
class RecognitionConfig:
    """识别器配置 - 可控参数"""
    max_candidates: int = 3          # 最多返回几个候选
    min_confidence: float = 0.1      # 最低置信度
    use_knowledge_base: bool = True  # 是否使用知识库


# ========== 知识库（可扩展） ==========

class CharacterKnowledge:
    """古文字知识库"""
    
    def __init__(self):
        # 常用古文字映射（示例，可扩展）
        self.radical_map = {
            "口": ["曰", "囗"],
            "木": ["林", "森"],
            "水": ["氵", "淼"],
        }
        
        # 字符相似度矩阵（示例）
        self.similarity = {}
    
    def get_radicals(self, char: str) -> List[str]:
        """获取字符的偏旁部首"""
        return self.radical_map.get(char, [])
    
    def is_valid_character(self, char: str) -> bool:
        """验证是否为有效古文字"""
        # 扩展：连接真实字典
        return len(char) > 0


# ========== 主引擎 ==========

class MORESCharacterEngine:
    """
    墨睿思 MORES 古文字识别引擎
    可控 · 可解释 · 可决策
    """
    
    def __init__(self, 
                 detection_config: Optional[DetectionConfig] = None,
                 recognition_config: Optional[RecognitionConfig] = None):
        
        self.detection_config = detection_config or DetectionConfig()
        self.recognition_config = recognition_config or RecognitionConfig()
        self.knowledge_base = CharacterKnowledge()
        
        # 决策路径记录
        self.decision_path = []
        
        # 待集成：底层检测模型（后续添加）
        self._detector = None
        self._recognizer = None
        
        self._log("引擎初始化完成")
    
    def _log(self, message: str):
        """记录决策路径"""
        self.decision_path.append(f"[{len(self.decision_path)}] {message}")
        print(f"[MORES] {message}")
    
    def _log_decision(self, step: str, data: dict):
        """记录详细决策"""
        self.decision_path.append(f"  └─ {step}: {data}")
    
    def set_detector(self, detector):
        """设置底层检测器（复用 v27/v29 模型）"""
        self._detector = detector
        self._log("检测器已加载")
    
    def set_recognizer(self, recognizer):
        """设置底层识别器"""
        self._recognizer = recognizer
        self._log("识别器已加载")
    
    def _filter_boxes_by_rules(self, boxes: List[DetectionBox]) -> List[DetectionBox]:
        """
        规则过滤（可控层）
        解释：根据几何规则过滤不合理检测框
        """
        filtered = []
        for box in boxes:
            # 规则1：面积不能太小
            if box.area < self.detection_config.min_area:
                self._log_decision("过滤-面积太小", {"area": box.area, "min": self.detection_config.min_area})
                continue
            
            # 规则2：宽高比不能太离谱
            aspect_ratio = max(box.width / box.height, box.height / box.width)
            if aspect_ratio > self.detection_config.max_aspect_ratio:
                self._log_decision("过滤-宽高比异常", {"ratio": aspect_ratio, "max": self.detection_config.max_aspect_ratio})
                continue
            
            # 规则3：置信度不能太低
            if box.confidence < self.detection_config.min_confidence:
                self._log_decision("过滤-置信度过低", {"conf": box.confidence, "min": self.detection_config.min_confidence})
                continue
            
            filtered.append(box)
        
        self._log(f"规则过滤: {len(boxes)} → {len(filtered)} 个框")
        return filtered
    
    def _nms(self, boxes: List[DetectionBox]) -> List[DetectionBox]:
        """
        非极大值抑制（可控层）
        解释：去除重叠过多的检测框
        """
        if len(boxes) <= 1:
            return boxes
        
        # 按置信度排序
        boxes = sorted(boxes, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            
            # 移除与 best 重叠度过高的框
            remaining = []
            for box in boxes:
                iou = self._compute_iou(best, box)
                if iou < self.detection_config.nms_threshold:
                    remaining.append(box)
                else:
                    self._log_decision("NMS-去除重叠框", {"iou": iou, "threshold": self.detection_config.nms_threshold})
            boxes = remaining
        
        self._log(f"NMS: {len(boxes) + len(keep)} → {len(keep)} 个框")
        return keep
    
    def _compute_iou(self, box1: DetectionBox, box2: DetectionBox) -> float:
        """计算 IoU"""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1.area
        area2 = box2.area
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def process(self, image: np.ndarray) -> EngineOutput:
        """
        主处理流程
        可控 + 可解释 + 可决策
        """
        self.decision_path = []  # 重置决策路径
        self._log("开始处理图片")
        
        # 步骤1：检测（待集成模型）
        if self._detector is None:
            self._log("警告：检测器未加载，使用空检测")
            raw_boxes = []
        else:
            raw_boxes = self._run_detection(image)
        
        # 步骤2：规则过滤（可控层）
        filtered_boxes = self._filter_boxes_by_rules(raw_boxes)
        
        # 步骤3：NMS（可控层）
        final_boxes = self._nms(filtered_boxes)
        
        # 步骤4：识别（待集成）
        candidates = self._run_recognition(image, final_boxes) if self._recognizer else []
        
        self._log(f"处理完成: {len(final_boxes)} 个字符框")
        
        return EngineOutput(
            boxes=final_boxes,
            candidates=candidates,
            decision_path=self.decision_path,
            visualization=self._generate_visualization(image, final_boxes)
        )
    
    def _run_detection(self, image: np.ndarray) -> List[DetectionBox]:
        """运行底层检测器（待实现）"""
        # 这里后续集成 v27/v29 模型
        self._log("运行检测器...")
        # TODO: 调用 self._detector
        return []
    
    def _run_recognition(self, image: np.ndarray, boxes: List[DetectionBox]) -> List[List[RecognitionCandidate]]:
        """运行底层识别器（待实现）"""
        self._log("运行识别器...")
        # TODO: 调用 self._recognizer
        return []
    
    def _generate_visualization(self, image: np.ndarray, boxes: List[DetectionBox]) -> np.ndarray:
        """生成可视化图（可解释输出）"""
        vis = image.copy()
        for box in boxes:
            cv2.rectangle(vis, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{box.confidence:.2f}", (box.x1, box.y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis
    
    def explain_last(self) -> str:
        """输出最后一次处理的解释"""
        return "\n".join(self.decision_path)
    
    def get_controllable_params(self) -> Dict:
        """获取可调节的控制参数"""
        return {
            "detection": {
                "prob_threshold": self.detection_config.prob_threshold,
                "nms_threshold": self.detection_config.nms_threshold,
                "min_area": self.detection_config.min_area,
                "max_aspect_ratio": self.detection_config.max_aspect_ratio,
                "min_confidence": self.detection_config.min_confidence,
            },
            "recognition": {
                "max_candidates": self.recognition_config.max_candidates,
                "min_confidence": self.recognition_config.min_confidence,
                "use_knowledge_base": self.recognition_config.use_knowledge_base,
            }
        }
    
    def set_controllable_params(self, params: Dict):
        """动态调节控制参数"""
        if "detection" in params:
            for k, v in params["detection"].items():
                if hasattr(self.detection_config, k):
                    setattr(self.detection_config, k, v)
                    self._log(f"参数调节: detection.{k} = {v}")
        
        if "recognition" in params:
            for k, v in params["recognition"].items():
                if hasattr(self.recognition_config, k):
                    setattr(self.recognition_config, k, v)
                    self._log(f"参数调节: recognition.{k} = {v}")


# ========== 快速测试 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("墨睿思 MORES 古文字识别引擎 v30")
    print("可控 · 可解释 · 可决策")
    print("=" * 60)
    
    # 创建引擎
    engine = MORESCharacterEngine()
    
    # 查看可控参数
    print("\n【可控参数列表】")
    import json
    print(json.dumps(engine.get_controllable_params(), indent=2, ensure_ascii=False))
    
    # 创建测试图片（空白，仅验证流程）
    test_image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    
    # 处理
    print("\n【处理流程】")
    output = engine.process(test_image)
    
    print(f"\n【输出结果】")
    print(f"  检测框数量: {len(output.boxes)}")
    print(f"  决策路径: {len(output.decision_path)} 步")
    
    print("\n✅ 引擎骨架运行正常")
    print("\n下一步：集成底层检测模型")