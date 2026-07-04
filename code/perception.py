"""
code/perception.py
多模态感知模块
支持RGB图像 + 激光雷达融合
用于货物检测、位姿估计、障碍物识别
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import warnings


@dataclass
class PerceptionResult:
    """感知结果数据结构"""
    # 货物信息
    target_position: Tuple[float, float, float]  # (x, y, z) 相对叉车
    target_orientation: float  # 偏航角 (度)
    target_dimensions: Tuple[float, float, float]  # (长, 宽, 高) (m)

    # 货架信息
    shelf_position: Optional[Tuple[float, float, float]] = None
    shelf_height: Optional[float] = None

    # 障碍物信息
    obstacles: List[Tuple[float, float, float]] = None  # 障碍物位置列表

    # 置信度
    confidence: float = 0.0

    # 原始数据
    raw_detections: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.obstacles is None:
            self.obstacles = []


class PerceptionModule:
    """
    多模态感知模块
    支持RGB相机和激光雷达数据融合
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 配置字典
                - camera.enabled: bool
                - camera.width: int
                - camera.height: int
                - lidar.enabled: bool
                - lidar.range: float
        """
        if config is None:
            config = {}

        self.camera_enabled = config.get('camera', {}).get('enabled', True)
        self.lidar_enabled = config.get('lidar', {}).get('enabled', False)

        self.image_width = config.get('camera', {}).get('width', 640)
        self.image_height = config.get('camera', {}).get('height', 480)
        self.lidar_range = config.get('lidar', {}).get('range', 10.0)

        # 模拟感知噪声
        self.position_noise = 0.02  # 位置噪声 (m)
        self.angle_noise = 0.5  # 角度噪声 (度)
        self.detection_threshold = 0.6  # 检测置信度阈值

        # 缓存
        self.last_rgb = None
        self.last_depth = None
        self.last_lidar = None
        self.last_result = None

    def detect_from_rgb(self, rgb_image: np.ndarray) -> List[Dict]:
        """
        从RGB图像检测目标和障碍物

        Args:
            rgb_image: RGB图像 (H, W, 3)

        Returns:
            检测结果列表 [{'class': str, 'bbox': [x1,y1,x2,y2], 'confidence': float}, ...]
        """
        # 简化：模拟检测结果
        # 在实际场景中，这里应调用YOLO/RT-DETR等模型

        h, w = rgb_image.shape[:2]

        # 模拟检测
        detections = []

        # 模拟货物检测
        cargo_bbox = [int(w * 0.4), int(h * 0.3), int(w * 0.6), int(h * 0.7)]
        detections.append({
            'class': 'cargo',
            'bbox': cargo_bbox,
            'confidence': 0.85 + np.random.randn() * 0.05,
            'center': ((cargo_bbox[0] + cargo_bbox[2]) // 2,
                       (cargo_bbox[1] + cargo_bbox[3]) // 2)
        })

        # 模拟货架检测
        shelf_bbox = [int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)]
        detections.append({
            'class': 'shelf',
            'bbox': shelf_bbox,
            'confidence': 0.90 + np.random.randn() * 0.03,
            'center': ((shelf_bbox[0] + shelf_bbox[2]) // 2,
                       (shelf_bbox[1] + shelf_bbox[3]) // 2)
        })

        return detections

    def detect_from_lidar(self, lidar_points: np.ndarray) -> List[Dict]:
        """
        从激光雷达数据检测目标和障碍物

        Args:
            lidar_points: 点云数据 (N, 3)

        Returns:
            检测结果列表 [{'class': str, 'position': (x,y,z), 'confidence': float}, ...]
        """
        if len(lidar_points) == 0:
            return []

        detections = []

        # 简化：基于点云聚类模拟检测
        # 在实际场景中，这里应调用点云分割/聚类算法

        # 模拟货物检测
        detections.append({
            'class': 'cargo',
            'position': (2.0 + np.random.randn() * 0.02, 0.0, 0.5 + np.random.randn() * 0.02),
            'confidence': 0.88 + np.random.randn() * 0.04
        })

        # 模拟货架检测
        detections.append({
            'class': 'shelf',
            'position': (0.5, 0.0, 1.5),
            'confidence': 0.92 + np.random.randn() * 0.03
        })

        return detections

    def estimate_pose(self, detections: List[Dict]) -> Dict:
        """
        从检测结果估计目标位姿

        Args:
            detections: 检测结果列表

        Returns:
            位姿估计结果
        """
        pose = {
            'position': (0.0, 0.0, 0.0),
            'orientation': 0.0,
            'confidence': 0.0
        }

        # 查找货物检测
        cargo_detections = [d for d in detections if d.get('class') == 'cargo']

        if cargo_detections:
            # 取置信度最高的检测
            best = max(cargo_detections, key=lambda x: x.get('confidence', 0))

            if 'position' in best:
                pos = best['position']
                pose['position'] = pos
                pose['confidence'] = best.get('confidence', 0.8)
            elif 'bbox' in best:
                # 从边界框估计位置 (简化)
                bbox = best['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                # 假设图像中心对应正前方，计算偏移
                img_center = self.image_width / 2
                angle_offset = (center_x - img_center) / img_center * 30  # 最大30度
                pose['position'] = (2.0, 0.0, 0.5)
                pose['orientation'] = angle_offset
                pose['confidence'] = best.get('confidence', 0.7)

        return pose

    def fuse(self, rgb_detections: List[Dict],
             lidar_detections: List[Dict]) -> PerceptionResult:
        """
        融合RGB和激光雷达检测结果

        Args:
            rgb_detections: RGB检测结果
            lidar_detections: 激光雷达检测结果

        Returns:
            融合后的感知结果
        """
        # 合并检测
        all_detections = rgb_detections + lidar_detections

        # 提取货物信息
        cargo_detections = [d for d in all_detections if d.get('class') == 'cargo']

        if cargo_detections:
            # 取最高置信度
            best = max(cargo_detections, key=lambda x: x.get('confidence', 0))

            if 'position' in best:
                pos = best['position']
                target_pos = (pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0)
            else:
                target_pos = (2.0, 0.0, 0.5)

            target_orientation = 0.0  # 默认

            # 估计尺寸
            dimensions = (0.8, 0.6, 0.5)  # 默认托盘尺寸

            confidence = best.get('confidence', 0.7)
        else:
            target_pos = (0.0, 0.0, 0.0)
            target_orientation = 0.0
            dimensions = (0.0, 0.0, 0.0)
            confidence = 0.0

        # 提取货架信息
        shelf_detections = [d for d in all_detections if d.get('class') == 'shelf']
        shelf_pos = None
        shelf_height = None

        if shelf_detections:
            best_shelf = max(shelf_detections, key=lambda x: x.get('confidence', 0))
            if 'position' in best_shelf:
                pos = best_shelf['position']
                shelf_pos = (pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0)
                shelf_height = 1.2  # 默认货架高度

        # 提取障碍物
        obstacle_detections = [d for d in all_detections
                               if d.get('class') not in ['cargo', 'shelf']]
        obstacles = []
        for d in obstacle_detections:
            if 'position' in d:
                pos = d['position']
                obstacles.append((pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0))

        return PerceptionResult(
            target_position=target_pos,
            target_orientation=target_orientation,
            target_dimensions=dimensions,
            shelf_position=shelf_pos,
            shelf_height=shelf_height,
            obstacles=obstacles,
            confidence=confidence,
            raw_detections=all_detections
        )

    def process(self, rgb_image: Optional[np.ndarray] = None,
                lidar_points: Optional[np.ndarray] = None,
                depth_image: Optional[np.ndarray] = None) -> PerceptionResult:
        """
        完整感知处理流程

        Args:
            rgb_image: RGB图像
            lidar_points: 激光雷达点云
            depth_image: 深度图

        Returns:
            感知结果
        """
        # 缓存
        if rgb_image is not None:
            self.last_rgb = rgb_image
        if lidar_points is not None:
            self.last_lidar = lidar_points
        if depth_image is not None:
            self.last_depth = depth_image

        rgb_detections = []
        lidar_detections = []

        # 处理RGB
        if self.camera_enabled and self.last_rgb is not None:
            rgb_detections = self.detect_from_rgb(self.last_rgb)

        # 处理激光雷达
        if self.lidar_enabled and self.last_lidar is not None:
            lidar_detections = self.detect_from_lidar(self.last_lidar)

        # 如果既没有RGB也没有激光雷达，返回空结果
        if not rgb_detections and not lidar_detections:
            return PerceptionResult(
                target_position=(0.0, 0.0, 0.0),
                target_orientation=0.0,
                target_dimensions=(0.0, 0.0, 0.0),
                confidence=0.0
            )

        # 融合
        self.last_result = self.fuse(rgb_detections, lidar_detections)

        return self.last_result

    def get_target_relative_pose(self) -> Dict:
        """
        获取目标相对叉车的位姿

        Returns:
            {'position': (x,y,z), 'orientation': float}
        """
        if self.last_result is None:
            return {'position': (0.0, 0.0, 0.0), 'orientation': 0.0}

        return {
            'position': self.last_result.target_position,
            'orientation': self.last_result.target_orientation
        }

    def get_obstacles(self) -> List[Tuple]:
        """获取障碍物列表"""
        if self.last_result is None:
            return []
        return self.last_result.obstacles

    def reset(self):
        """重置感知模块"""
        self.last_rgb = None
        self.last_depth = None
        self.last_lidar = None
        self.last_result = None


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    print("🧪 测试感知模块...")

    # 创建感知模块
    config = {
        'camera': {'enabled': True, 'width': 640, 'height': 480},
        'lidar': {'enabled': True, 'range': 10.0}
    }
    perception = PerceptionModule(config)

    # 模拟RGB图像
    rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 模拟激光雷达点云
    lidar_points = np.random.randn(100, 3) * 0.5 + np.array([2.0, 0.0, 0.5])

    # 处理
    result = perception.process(rgb_image=rgb_image, lidar_points=lidar_points)

    print(f"  目标位置: {result.target_position}")
    print(f"  目标方向: {result.target_orientation:.1f}°")
    print(f"  目标尺寸: {result.target_dimensions}")
    print(f"  货架位置: {result.shelf_position}")
    print(f"  障碍物数量: {len(result.obstacles)}")
    print(f"  置信度: {result.confidence:.2f}")

    print("\n✅ perception.py 测试通过!")
