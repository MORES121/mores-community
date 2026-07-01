import torch
import torch.nn as nn
import numpy as np

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

def encode_sequence(seq: str, max_len: int = 200):
    uniq = list(set(seq[:max_len]))
    return torch.tensor([uniq.index(ch) for ch in seq[:max_len]], dtype=torch.long)

def evaluate(protein_seq, rna_seq):
    model = LightMORES()
    model.load_state_dict(torch.load("mores_light.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        score = model(encode_sequence(protein_seq), encode_sequence(rna_seq))
    length_penalty = np.log(len(protein_seq) + len(rna_seq)) / 50
    stability = max(0.0, min(1.0, score - length_penalty))
    return {
        "score": round(score, 4),
        "stability": round(stability, 4),
        "risk": "HIGH" if stability < 0.4 else ("MEDIUM" if stability < 0.7 else "LOW")
    }

if __name__ == "__main__":
    rna = "GCUUUUUUAUCGAAGAUGACUCCAAAGGCAGAAC"
    prot = "SEREELKALLDEYEQAMKELMKYKNQLLALERGTDLYDPEFAKYLKELLKLTEEYLNKILKKLKELIENSKDPLIQALLSVKGVGPITAAYLYAYVDLTKATSASALWAYLGIDKPSHKRYKKGEAGGGNKKLRTAVWNQARSMIKRRDSPYRKVYLEEKKRLSKSKKVTKSRNTQGELVKVKWSKAKPSHKHGAALRAVMKTFLADVWFVGHKIAGLPTRPLYVGIVDPEKRGFKY"
    res = evaluate(prot, rna)
    print("=== MORES 本地推理结果 ===")
    print(f"响应值   : {res['score']}")
    print(f"稳定度   : {res['stability']}")
    print(f"风险等级 : {res['risk']}")
