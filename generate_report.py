import torch
import torch.nn as nn
import numpy as np
import csv
import json
from pathlib import Path

# ========== 1. 模型定义 ==========
class LightMORES(nn.Module):
    def __init__(self, protein_dim=237, rna_dim=34, hidden_dim=64):
        super().__init__()
        self.protein_encoder = nn.Embedding(protein_dim, hidden_dim)
        self.rna_encoder = nn.Embedding(rna_dim, hidden_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, p_idx, r_idx):
        p = self.protein_encoder(p_idx)
        r = self.rna_encoder(r_idx)
        combined = torch.cat([p.mean(dim=0), r.mean(dim=0)], dim=0)
        return torch.sigmoid(self.edge_predictor(combined.unsqueeze(0))).item()

# ========== 2. 序列编码 ==========
def encode_sequence(seq: str, max_len: int = 200):
    uniq = list(set(seq[:max_len]))
    return torch.tensor([uniq.index(ch) for ch in seq[:max_len]], dtype=torch.long)

# ========== 3. 单个评估 ==========
def evaluate_pair(protein_seq: str, rna_seq: str):
    model = LightMORES()
    model.load_state_dict(torch.load("mores_light.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        score = model(encode_sequence(protein_seq), encode_sequence(rna_seq))
    length_penalty = np.log(len(protein_seq) + len(rna_seq)) / 50
    stability = max(0.0, min(1.0, score - length_penalty))
    return {
        "raw_score": round(score, 4),
        "stability": round(stability, 4),
        "risk": "HIGH" if stability < 0.4 else ("MEDIUM" if stability < 0.7 else "LOW")
    }

# ========== 4. 批量评估并生成报告 ==========
def generate_report(targets, output_format="txt"):
    results = []
    for t in targets:
        res = evaluate_pair(t["protein"], t["rna"])
        results.append({
            "name": t["name"],
            "stability": res["stability"],
            "risk": res["risk"],
            "raw_score": res["raw_score"]
        })
    ranked = sorted(results, key=lambda x: x["stability"], reverse=True)

    # 报告头
    report = []
    report.append("=" * 50)
    report.append("MORES 靶点早期评估报告")
    report.append("=" * 50)
    report.append("")
    report.append("评估结论（按优先顺序）")
    report.append("")

    for idx, r in enumerate(ranked, 1):
        recommend = "✅ 推荐推进" if r["risk"] == "MEDIUM" else "❌ 建议暂缓"
        report.append(f"{idx}. {r['name']}")
        report.append(f"   稳定度: {r['stability']} | 风险: {r['risk']}")
        report.append(f"   建议: {recommend}")
        report.append("")

    # 一句话结论
    best = ranked[0]
    report.append("=" * 50)
    report.append("一句话结论")
    report.append("=" * 50)
    report.append(f"在本次评估的 {len(targets)} 个候选靶点中，")
    if best["risk"] == "MEDIUM":
        report.append(f"「{best['name']}」是唯一具备中等稳定度、值得继续推进的靶点。")
    else:
        report.append(f"所有靶点风险均偏高，建议重新筛选。")
    report.append("其余建议暂缓或放弃。")
    report.append("")
    report.append("=" * 50)

    # 保存文本报告
    if output_format == "txt":
        with open("靶点评估报告.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print("✅ 报告已生成：靶点评估报告.txt")

    # 同时保存 JSON（便于程序调用）
    with open("靶点评估结果.json", "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2)
    print("✅ 数据已保存：靶点评估结果.json")

    # 打印到屏幕
    print("\n".join(report))

# ========== 5. 示例（替换为你的真实靶点）==========
if __name__ == "__main__":
    target_list = [
        {"name": "Target_A", "protein": "SEREELKALLDEYEQAMKELMKYKNQLLALERGTDLYDPEFAKYLKELLK", "rna": "GCUUUUUUAUCGAAGAUGACUCCAAAGGCAGAAC"},
        {"name": "Target_B", "protein": "MGSSHHHHHHSSGLVPRGSHMSGKIQHKAVVPAPSRIPLTLSEIED", "rna": "GGUGAUUCUCAGGUACAGCUAGUCUCAACUGUGAGGCG"},
        {"name": "Target_C", "protein": "SEREELKALLDEYEQAMKELMKYKNQLLALERGTDLYDPEFAK", "rna": "CCUGCAUCGAUCGAUCGACUAGCUAGCUAGCUAGCUAG"}
    ]
    generate_report(target_list, output_format="txt")
