"""
MORES 决策引擎 - 可控核聚变赛道
等离子体位形控制
可控 · 可解释 · 可决策
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# ========== 数据结构定义 ==========

@dataclass
class PlasmaObservation:
    """等离子体观测状态"""
    # 位形参数
    plasma_current: float = 0.0      # 等离子体电流 (MA)
    beta_n: float = 0.0              # 归一化比压
    li: float = 0.0                  # 内感
    elongation: float = 1.0          # 拉长度
    triangularity: float = 0.0       # 三角变形度
    
    # 稳定性指标
    greenwald_fraction: float = 0.0  # 密度/Greenwald极限
    radiated_power_fraction: float = 0.0  # 辐射功率比例
    
    # 位置误差
    r_error: float = 0.0             # 水平位置误差 (m)
    z_error: float = 0.0             # 垂直位置误差 (m)
    
    # 温度/密度
    electron_temp: float = 0.0       # 电子温度 (keV)
    line_avg_density: float = 0.0    # 线平均密度 (1e19/m^3)
    
    def to_array(self) -> np.ndarray:
        """转换为向量（供模型输入）"""
        return np.array([
            self.plasma_current,
            self.beta_n,
            self.li,
            self.elongation,
            self.triangularity,
            self.greenwald_fraction,
            self.radiated_power_fraction,
            self.r_error,
            self.z_error,
            self.electron_temp,
            self.line_avg_density
        ])


@dataclass
class ControlAction:
    """控制动作"""
    pf_currents: np.ndarray   # 极向场线圈电流 (多组)
    heating_power: float      # 辅助加热功率 (MW)
    fueling_rate: float       # 加料速率
    
    @property
    def to_array(self) -> np.ndarray:
        return np.concatenate([self.pf_currents, [self.heating_power, self.fueling_rate]])


# ========== 可控层：规则约束 ==========

class ControllableFusion:
    """可控层 - 等离子体控制的规则约束"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.safety_margin = 2.0  # 安全因子目标
    
    def _default_config(self) -> Dict:
        return {
            # 物理极限
            "max_plasma_current": 2.0,      # MA
            "max_beta_n": 3.0,              # 归一化比压极限
            "max_greenwald": 1.2,           # 密度极限
            "max_r_error": 0.1,             # 水平位置误差极限 (m)
            "max_z_error": 0.05,            # 垂直位置误差极限 (m)
            
            # 控制参数
            "pf_current_limits": (-10.0, 10.0),  # 线圈电流范围 (kA)
            "max_heating_power": 30.0,           # MW
            "control_frequency": 1000,           # Hz
        }
    
    def filter_action(self, action: ControlAction, obs: PlasmaObservation) -> Tuple[ControlAction, List[str]]:
        """
        规则过滤（可控层核心）
        返回：过滤后的动作 + 决策路径
        """
        violations = []
        original_action = action
        
        # 规则1：密度极限保护
        if obs.greenwald_fraction > self.config["max_greenwald"]:
            action.fueling_rate *= 0.5
            violations.append(f"密度过高({obs.greenwald_fraction:.2f})，降低加料50%")
        
        # 规则2：垂直不稳定性保护
        if abs(obs.z_error) > self.config["max_z_error"]:
            # 紧急垂直控制
            action.pf_currents[0] *= 1.2  # 增强垂直场
            violations.append(f"垂直位移过大({obs.z_error:.3f}m)，增强垂直控制")
        
        # 规则3：比压极限保护
        if obs.beta_n > self.config["max_beta_n"]:
            action.heating_power *= 0.8
            violations.append(f"比压过高({obs.beta_n:.2f})，降低加热功率20%")
        
        # 规则4：位置控制
        action = self._position_control(action, obs)
        
        # 规则5：动作边界裁剪
        action = self._clip_action(action)
        
        return action, violations
    
    def _position_control(self, action: ControlAction, obs: PlasmaObservation) -> ControlAction:
        """位置控制 PID（简化）"""
        # 水平位置
        p_gain = 0.1
        action.pf_currents[1] += p_gain * obs.r_error
        
        # 垂直位置
        d_gain = 0.2
        action.pf_currents[0] += d_gain * obs.z_error
        
        return action
    
    def _clip_action(self, action: ControlAction) -> ControlAction:
        """动作边界裁剪"""
        action.pf_currents = np.clip(
            action.pf_currents,
            self.config["pf_current_limits"][0],
            self.config["pf_current_limits"][1]
        )
        action.heating_power = np.clip(action.heating_power, 0, self.config["max_heating_power"])
        action.fueling_rate = np.clip(action.fueling_rate, 0, 1)
        return action


# ========== 可解释层 ==========

class ExplainableFusion:
    """可解释层 - 记录决策路径和可视化"""
    
    def __init__(self):
        self.decision_log = []
    
    def record(self, step: str, data: Dict):
        """记录决策步骤"""
        entry = {"step": step, "data": data, "timestamp": len(self.decision_log)}
        self.decision_log.append(entry)
    
    def explain_last(self) -> str:
        """输出最近决策的解释"""
        if not self.decision_log:
            return "无决策记录"
        
        last = self.decision_log[-1]
        return f"决策: {last['step']} | {last['data']}"
    
    def get_full_log(self) -> List[Dict]:
        """获取完整决策日志"""
        return self.decision_log
    
    def clear(self):
        """清空日志"""
        self.decision_log = []


# ========== 可决策层 ==========

class DecisionFusion:
    """可决策层 - 多候选输出 + 置信度评估"""
    
    def __init__(self):
        self.candidates = []
    
    def generate_candidates(self, base_action: ControlAction, obs: PlasmaObservation, num_candidates=3) -> List[Dict]:
        """
        生成多个候选动作
        每个候选有不同策略方向
        """
        candidates = []
        
        # 候选1：保守策略（低功率）
        conservative = ControlAction(
            pf_currents=base_action.pf_currents * 0.8,
            heating_power=base_action.heating_power * 0.7,
            fueling_rate=base_action.fueling_rate * 0.8
        )
        candidates.append({"action": conservative, "strategy": "conservative", "risk": "low"})
        
        # 候选2：激进策略（高功率）
        aggressive = ControlAction(
            pf_currents=base_action.pf_currents * 1.2,
            heating_power=min(base_action.heating_power * 1.3, 30.0),
            fueling_rate=min(base_action.fueling_rate * 1.2, 1.0)
        )
        candidates.append({"action": aggressive, "strategy": "aggressive", "risk": "high"})
        
        # 候选3：平衡策略（基础动作）
        balanced = base_action
        candidates.append({"action": balanced, "strategy": "balanced", "risk": "medium"})
        
        # 基于观测状态评估每个候选的合适度
        for cand in candidates:
            cand["score"] = self._evaluate_candidate(cand["action"], obs)
        
        # 按分数排序
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        return candidates
    
    def _evaluate_candidate(self, action: ControlAction, obs: PlasmaObservation) -> float:
        """评估候选动作的合适度（0-1）"""
        score = 0.5  # 基础分
        
        # 根据等离子体状态调整
        if obs.beta_n > 2.5:
            # 高比压时，保守策略更好
            if action.heating_power < 10:
                score += 0.3
        
        if abs(obs.z_error) > 0.03:
            # 垂直不稳定时，需要强控制
            if abs(action.pf_currents[0]) > 5:
                score += 0.2
        
        return min(score, 1.0)
    
    def select_best(self, candidates: List[Dict]) -> Dict:
        """选择最佳候选"""
        if not candidates:
            return None
        return candidates[0]


# ========== MORES 核聚变决策引擎 ==========

class MORESFusionEngine:
    """
    MORES 决策引擎 - 可控核聚变赛道
    继承三大核心模块：可控层、可解释层、可决策层
    """
    
    def __init__(self, config: Dict = None):
        print("=" * 60)
        print("MORES 决策引擎 - 可控核聚变赛道 v1.0")
        print("=" * 60)
        
        # 三大核心模块
        self.controllable = ControllableFusion(config)
        self.explainable = ExplainableFusion()
        self.decision = DecisionFusion()
        
        # 底层AI模型（待集成）
        self.ai_model = None
        
        # 状态记录
        self.current_obs = None
        self.plasma_state = "init"
        
        print("[MORES] 引擎初始化完成")
        print("   可控层: ✅ 规则约束已加载")
        print("   可解释层: ✅ 决策日志已就绪")
        print("   可决策层: ✅ 多候选评估已启用")
    
    def act(self, observation: PlasmaObservation) -> Tuple[ControlAction, Dict]:
        """
        核心决策接口
        输入：等离子体观测
        输出：控制动作 + 可解释决策信息
        """
        self.explainable.clear()
        self.current_obs = observation
        
        # 步骤1：AI模型推理（如果有）
        self.explainable.record("开始推理", {"plasma_current": observation.plasma_current})
        
        if self.ai_model:
            ai_action = self._ai_inference(observation)
        else:
            # 无AI模型时的默认动作（基于规则）
            ai_action = self._rule_based_action(observation)
        
        # 步骤2：可控层过滤（规则约束）
        filtered_action, violations = self.controllable.filter_action(ai_action, observation)
        for v in violations:
            self.explainable.record("可控层-规则触发", {"rule": v})
        
        # 步骤3：可决策层（生成多候选）
        candidates = self.decision.generate_candidates(filtered_action, observation)
        best_candidate = self.decision.select_best(candidates)
        
        self.explainable.record("可决策层-选择", {
            "strategy": best_candidate["strategy"],
            "score": best_candidate["score"]
        })
        
        # 步骤4：输出最终动作 + 解释
        final_action = best_candidate["action"]
        
        explanation = {
            "final_action": {
                "pf_currents": final_action.pf_currents.tolist(),
                "heating_power": final_action.heating_power,
                "fueling_rate": final_action.fueling_rate
            },
            "decision_path": self.explainable.get_full_log(),
            "strategy_used": best_candidate["strategy"],
            "violations_applied": violations
        }
        
        return final_action, explanation
    
    def _rule_based_action(self, obs: PlasmaObservation) -> ControlAction:
        """基于规则的默认动作（无AI时）"""
        # 6个极向场线圈（示例）
        pf_currents = np.array([-5.0, 2.0, 3.0, -2.0, 1.0, -1.0])
        
        # 基于等离子体电流调节
        if obs.plasma_current < 0.5:
            pf_currents[0] *= 1.2  # 增加伏秒
        elif obs.plasma_current > 1.5:
            pf_currents[0] *= 0.8  # 减少伏秒
        
        # 基于垂直位置调节
        pf_currents[0] += 5.0 * obs.z_error
        
        return ControlAction(
            pf_currents=pf_currents,
            heating_power=15.0,
            fueling_rate=0.5
        )
    
    def _ai_inference(self, obs: PlasmaObservation) -> ControlAction:
        """AI模型推理（待实现）"""
        # 预留接口，后续集成神经网络模型
        return self._rule_based_action(obs)
    
    def load_ai_model(self, model_path: str):
        """加载AI模型"""
        # 待实现
        print(f"[MORES] 加载AI模型: {model_path}")
        self.ai_model = "loaded"
    
    def get_controllable_params(self) -> Dict:
        """获取可控参数"""
        return self.controllable.config
    
    def update_controllable_params(self, params: Dict):
        """更新可控参数"""
        for key, value in params.items():
            if key in self.controllable.config:
                self.controllable.config[key] = value
                print(f"[MORES] 参数更新: {key} = {value}")


# ========== 快速测试 ==========

def test_fusion_engine():
    """测试核聚变决策引擎"""
    print("\n" + "=" * 60)
    print("可控核聚变引擎测试")
    print("=" * 60)
    
    # 创建引擎
    engine = MORESFusionEngine()
    
    # 创建模拟观测（正常状态）
    normal_obs = PlasmaObservation(
        plasma_current=1.0,
        beta_n=1.5,
        li=0.8,
        elongation=1.6,
        triangularity=0.3,
        greenwald_fraction=0.7,
        radiated_power_fraction=0.2,
        r_error=0.02,
        z_error=0.01,
        electron_temp=5.0,
        line_avg_density=4.0
    )
    
    print("\n" + "-" * 40)
    print("场景1: 正常状态")
    print("-" * 40)
    
    action, explanation = engine.act(normal_obs)
    
    print(f"最终控制动作:")
    print(f"  极向场电流: {action.pf_currents}")
    print(f"  加热功率: {action.heating_power:.1f} MW")
    print(f"  加料速率: {action.fueling_rate:.2f}")
    print(f"  采用策略: {explanation['strategy_used']}")
    print(f"  触发的规则: {explanation['violations_applied']}")
    
    # 测试异常状态
    print("\n" + "-" * 40)
    print("场景2: 异常状态（高比压+垂直位移）")
    print("-" * 40)
    
    abnormal_obs = PlasmaObservation(
        plasma_current=1.5,
        beta_n=2.8,           # 接近极限
        li=1.2,
        elongation=1.7,
        triangularity=0.25,
        greenwald_fraction=0.9,
        radiated_power_fraction=0.3,
        r_error=0.05,
        z_error=0.08,         # 垂直位移过大
        electron_temp=6.0,
        line_avg_density=5.0
    )
    
    action2, explanation2 = engine.act(abnormal_obs)
    
    print(f"最终控制动作:")
    print(f"  极向场电流: {action2.pf_currents}")
    print(f"  加热功率: {action2.heating_power:.1f} MW")
    print(f"  采用策略: {explanation2['strategy_used']}")
    print(f"  触发的规则: {explanation2['violations_applied']}")
    
    print("\n" + "=" * 60)
    print("✅ 可控核聚变引擎测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_fusion_engine()