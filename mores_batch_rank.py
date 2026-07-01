import torch
import torch.nn as nn
import numpy as np
import csv
from typing import List, Dict

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

# ========== 4. 批量排序 ==========
def batch_rank(pairs: List[Dict]) -> List[Dict]:
    results = []
    for p in pairs:
        res = evaluate_pair(p["protein"], p["rna"])
        results.append({
            "name": p.get("name", "unknown"),
            "stability": res["stability"],
            "risk": res["risk"],
            "raw_score": res["raw_score"]
        })
    return sorted(results, key=lambda x: x["stability"], reverse=True)

# ========== 5. 示例 ==========
if __name__ == "__main__":
    print("\n[MORES 批量排序 · 靶点优先级]\n")

    target_list = [
        {"name": "Target_A", "protein": "SEREELKALLDEYEQAMKELMKYKNQLLALERGTDLYDPEFAKYLKELLK", "rna": "GCUUUUUUAUCGAAGAUGACUCCAAAGGCAGAAC"},
        {"name": "Target_B", "protein": "MGSSHHHHHHSSGLVPRGSHMSGKIQHKAVVPAPSRIPLTLSEIED", "rna": "GGUGAUUCUCAGGUACAGCUAGUCUCAACUGUGAGGCG"},
        {"name": "Target_C", "protein": "SEREELKALLDEYEQAMKELMKYKNQLLALERGTDLYDPEFAK", "rna": "CCUGCAUCGAUCGAUCGACUAGCUAGCUAGCUAGCUAG"}
    ]

    ranked = batch_rank(target_list)

    print("排序结果（最值得优先推进）\n")
    for idx, t in enumerate(ranked, 1):
        print(f"{idx}. {t['name']}")
        print(f"   稳定度: {t['stability']} | 风险: {t['risk']}")
        print("")
