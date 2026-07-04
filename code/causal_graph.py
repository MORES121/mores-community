"""
code/causal_graph.py
因果图建模（专利核心）
基于有向无环图构建叉车作业因果结构
支持干预分析、因果路径查询
"""

import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class ForkliftCausalGraph:
    """
    叉车作业因果图
    定义各变量间的因果关系，支持干预分析
    """

    # 预定义节点（基于叉车作业域知识）
    DEFAULT_NODES = [
        "perception_error",   # 感知误差
        "path_quality",       # 路径质量
        "fork_alignment",     # 叉齿对准度
        "steering_angle",     # 转向角
        "speed",              # 速度
        "fork_height",        # 叉齿高度
        "load_stability",     # 负载稳定性
        "collision_risk",     # 碰撞风险
        "task_success",       # 任务成功
    ]

    # 预定义因果边（先验知识）
    DEFAULT_EDGES = [
        ("perception_error", "path_quality"),
        ("perception_error", "fork_alignment"),
        ("path_quality", "steering_angle"),
        ("path_quality", "speed"),
        ("fork_alignment", "fork_height"),
        ("steering_angle", "load_stability"),
        ("fork_height", "load_stability"),
        ("speed", "collision_risk"),
        ("load_stability", "task_success"),
        ("collision_risk", "task_success"),
    ]

    def __init__(self, nodes: List[str] = None, edges: List[Tuple] = None):
        """
        初始化因果图

        Args:
            nodes: 节点列表，默认使用预定义节点
            edges: 边列表 [(父节点, 子节点), ...]
        """
        self.graph = nx.DiGraph()

        if nodes is None:
            nodes = self.DEFAULT_NODES
        if edges is None:
            edges = self.DEFAULT_EDGES

        self._nodes = nodes
        self._edges = edges
        self._build_graph()

        # 干预状态记录
        self._interventions = {}

        # 节点状态值缓存
        self._node_values = {}

    def _build_graph(self):
        """构建因果图"""
        self.graph.add_nodes_from(self._nodes)
        self.graph.add_edges_from(self._edges)

        # 检查是否有环
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("因果图存在环，请检查边定义！")

    def get_nodes(self) -> List[str]:
        """获取所有节点"""
        return list(self.graph.nodes)

    def get_edges(self) -> List[Tuple]:
        """获取所有边"""
        return list(self.graph.edges)

    def get_parents(self, node: str) -> List[str]:
        """获取指定节点的父节点"""
        return list(self.graph.predecessors(node))

    def get_children(self, node: str) -> List[str]:
        """获取指定节点的子节点"""
        return list(self.graph.successors(node))

    def get_causal_paths(self, start: str, end: str) -> List[List[str]]:
        """
        获取两个节点间的所有因果路径

        Args:
            start: 起始节点
            end: 终止节点

        Returns:
            路径列表，每条路径为节点列表
        """
        return list(nx.all_simple_paths(self.graph, start, end))

    def get_shortest_causal_path(self, start: str, end: str) -> Optional[List[str]]:
        """
        获取最短因果路径

        Args:
            start: 起始节点
            end: 终止节点

        Returns:
            最短路径节点列表，若不存在返回None
        """
        try:
            return nx.shortest_path(self.graph, start, end)
        except nx.NetworkXNoPath:
            return None

    def intervene(self, node: str, value: float):
        """
        干预操作：固定节点值（反事实推理核心）

        Args:
            node: 目标节点
            value: 固定值
        """
        self._interventions[node] = value
        # 切断所有入边
        parents = self.get_parents(node)
        for parent in parents:
            if self.graph.has_edge(parent, node):
                self.graph.remove_edge(parent, node)

    def reset_interventions(self):
        """重置所有干预操作"""
        self._interventions = {}
        # 重建图
        self.graph = nx.DiGraph()
        self._build_graph()

    def get_intervention_state(self) -> Dict:
        """获取当前干预状态"""
        return self._interventions.copy()

    def set_node_value(self, node: str, value: float):
        """设置节点值（用于状态跟踪）"""
        self._node_values[node] = value

    def get_node_value(self, node: str) -> Optional[float]:
        """获取节点值"""
        return self._node_values.get(node)

    def get_all_values(self) -> Dict:
        """获取所有节点值"""
        return self._node_values.copy()

    def update_from_config(self, config: Dict):
        """
        从配置更新节点值
        config: {node_name: value, ...}
        """
        for node, value in config.items():
            if node in self._nodes:
                self._node_values[node] = value

    def explain_path(self, path: List[str]) -> str:
        """
        生成因果路径的文本解释

        Args:
            path: 因果路径节点列表

        Returns:
            可读的因果链描述
        """
        if not path:
            return "空路径"

        explanation = " → ".join(path)
        return explanation

    def query_causal_effect(self, cause: str, effect: str) -> Dict:
        """
        查询因果关系强度（基于结构）

        Args:
            cause: 原因节点
            effect: 结果节点

        Returns:
            {
                'has_path': bool,
                'paths': List[路径],
                'shortest_path': 最短路径,
                'explanation': 文本描述
            }
        """
        paths = self.get_causal_paths(cause, effect)
        shortest = self.get_shortest_causal_path(cause, effect)

        return {
            'has_path': len(paths) > 0,
            'paths': paths,
            'shortest_path': shortest,
            'explanation': f"{cause} → {effect}: {self.explain_path(shortest) if shortest else '无路径'}"
        }

    def visualize(self, output_file: str = None):
        """
        可视化因果图（需要matplotlib和networkx的绘图功能）

        Args:
            output_file: 输出文件路径，若None则显示
        """
        try:
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(self.graph, seed=42)
            plt.figure(figsize=(12, 8))

            nx.draw_networkx_nodes(self.graph, pos, node_size=3000,
                                   node_color='lightblue', alpha=0.8)
            nx.draw_networkx_labels(self.graph, pos, font_size=10)
            nx.draw_networkx_edges(self.graph, pos, arrows=True,
                                   arrowsize=20, edge_color='gray')

            plt.title("叉车作业因果图")
            plt.axis('off')

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"✅ 因果图已保存至: {output_file}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            print("⚠️ matplotlib未安装，跳过可视化")

    def __repr__(self) -> str:
        return (f"ForkliftCausalGraph(nodes={len(self._nodes)}, "
                f"edges={len(self._edges)})")


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    print("🧪 测试因果图建模...")

    # 1. 创建因果图
    causal_graph = ForkliftCausalGraph()
    print(f"  节点: {causal_graph.get_nodes()}")
    print(f"  边: {causal_graph.get_edges()}")

    # 2. 测试父节点/子节点查询
    print(f"\n  perception_error 的父节点: {causal_graph.get_parents('perception_error')}")
    print(f"  perception_error 的子节点: {causal_graph.get_children('perception_error')}")

    # 3. 测试因果路径查询
    paths = causal_graph.get_causal_paths('perception_error', 'task_success')
    print(f"\n  perception_error → task_success 的因果路径:")
    for i, path in enumerate(paths):
        print(f"   路径 {i+1}: {' → '.join(path)}")

    # 4. 测试最短路径
    shortest = causal_graph.get_shortest_causal_path('perception_error', 'task_success')
    print(f"\n  最短路径: {' → '.join(shortest)}")

    # 5. 测试因果路径解释
    explanation = causal_graph.explain_path(shortest)
    print(f"\n  解释: {explanation}")

    # 6. 测试干预
    print("\n  干预操作: 固定 path_quality = 1.0")
    causal_graph.intervene('path_quality', 1.0)
    print(f"  干预后 path_quality 的父节点: {causal_graph.get_parents('path_quality')}")

    # 7. 重置
    causal_graph.reset_interventions()
    print(f"\n  重置后 path_quality 的父节点: {causal_graph.get_parents('path_quality')}")

    # 8. 查询因果关系
    result = causal_graph.query_causal_effect('fork_alignment', 'task_success')
    print(f"\n  查询 fork_alignment → task_success:")
    print(f"  存在路径: {result['has_path']}")
    print(f"  解释: {result['explanation']}")

    print("\n✅ causal_graph.py 测试通过!")
