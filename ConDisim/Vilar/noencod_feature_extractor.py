import torch
import torch.nn as nn

class VilarFeatureExtractor(nn.Module):
    """
    Entrée  : (B, 3, 200)           ← 3 espèces, 200 pas de temps
    Sortie  : (B, 15)               ← 15 features par trajectoire
    Méthode : 3 conv1d + BatchNorm + SiLU + GAP.
    """
    def __init__(self):
        super().__init__()

        def block(c_in, c_out, k=5, p=2):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=p),
                nn.BatchNorm1d(c_out),
                nn.SiLU(inplace=True)
            )

        self.backbone = nn.Sequential(
            block(3, 32),            # (B,32,200)
            block(32, 64),           # (B,64,200)
            block(64, 15, k=1, p=0)  # (B,15,200)
        )

        self.gap = nn.AdaptiveAvgPool1d(1)    # (B,15,1)

    def forward(self, x):                      # x : (B,3,200)
        h = self.backbone(x)                  # (B,15,200)
        h = self.gap(h).squeeze(-1)           # (B,15,1) → (B,15)
        return h

    # ---------- utilitaire hors-gradient ----------
    @torch.no_grad()
    def encode_dataset(self, loader, device="cpu"):
        self.eval()
        feats = []
        for batch in loader:                  # batch = (tensor,)
            x = batch[0].to(device)
            feats.append(self(x).cpu())
        return torch.cat(feats, dim=0)        # (N,15)

class SimpleStatsFeatureExtractor(nn.Module):
    """
    Baseline trivial : pour chaque canal (espèce) on calcule
    [mean, std, min, max, last] → vecteur de taille 5,
    concaténé sur les 3 espèces → (B, 15).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x : (B, 3, T)
        means = x.mean(dim=2)              # (B,3)
        stds  = x.std(dim=2)               # (B,3)
        mins  = x.min(dim=2)[0]            # (B,3)
        maxs  = x.max(dim=2)[0]            # (B,3)
        last  = x[:, :, -1]                # (B,3)
        # concatène en (B, 15)
        feats = torch.cat([means, stds, mins, maxs, last], dim=1)
        return feats                       # (B,15)