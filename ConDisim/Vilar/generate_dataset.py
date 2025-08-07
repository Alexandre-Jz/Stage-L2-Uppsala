import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from vilar_autoencoder import train_autoencoder, encode_dataset
from vilar_dataset import simulator, generate_data_parallel, Vilar_Oscillator
from gillespy2 import SSACSolver

# 1) On reprend les utilitaires de noencod_gen_dataset
def simulate_batch(n, model, solver):
    ts_raw, theta = generate_data_parallel(n, model, solver)
    ts_raw = ts_raw.squeeze(1).astype(np.float32)      # (n,3,200)
    theta  = theta.astype(np.float32)                  # (n,15)

    # filtre NaN / ±Inf
    flat   = ts_raw.reshape(ts_raw.shape[0], -1)
    valid  = np.isfinite(flat).all(axis=1) & ~np.isnan(flat).any(axis=1)
    return ts_raw[valid], theta[valid]

def collect_clean_dataset(budget, model, solver, batch_size=256):
    ts_list, th_list = [], []
    total_valid = 0

    while total_valid < budget:
        ts_batch, th_batch = simulate_batch(batch_size, model, solver)
        need = min(batch_size, budget - total_valid)
        ts_list.append(ts_batch[:need])
        th_list.append(th_batch[:need])
        total_valid += need
        print(f"  · récupéré {need} trajectoires valides — total : {total_valid}/{budget}")

    ts_all = np.concatenate(ts_list, axis=0)[:budget]
    th_all = np.concatenate(th_list, axis=0)[:budget]
    return ts_all, th_all

# 2) Script principal
def main():
    os.system('clear')
    # dossiers
    base_dir   = os.path.dirname(__file__)
    ds_dir     = os.path.join(base_dir, 'datasets')
    os.makedirs(ds_dir, exist_ok=True)

    # modèle & solveur (chaque worker compilera son binaire la 1ʳᵉ fois)
    model  = Vilar_Oscillator()
    solver = SSACSolver(model=model)

    # trajectoire « vraie »
    true_theta = np.array([50,500,0.01,50,50,5,10,0.5,1,0.2,1,1,2,50,100], dtype=np.float32)
    print("● Génération de la trajectoire de référence…")
    true_ts = simulator(true_theta, model, solver).squeeze(0).astype(np.float32)  # (3,200)
    print("  → OK", true_ts.shape)

    budgets = [30000]
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for budget in budgets:
        print(f"\n=== Budget {budget} ===")
        # on collecte proprement `budget` trajectoires
        ts_data, theta = collect_clean_dataset(budget, model, solver, batch_size=256)

        # mise en forme
        N, S, T = ts_data.shape
        flat     = ts_data.reshape(N, -1)
        data_scaler  = MinMaxScaler().fit(flat)
        ts_norm      = data_scaler.transform(flat).reshape(N,S,T)
        theta_norm   = MinMaxScaler().fit_transform(theta)

        # apprentissage AE
        ae_model, _    = train_autoencoder(ts_norm, device=device)
        summary_stats  = encode_dataset(ae_model, ts_norm, device=device)

        # encodage de la donnée vraie
        true_flat  = true_ts.reshape(1, -1)
        true_norm  = data_scaler.transform(true_flat).reshape(1,S,T)
        with torch.no_grad():
            true_emb = ae_model.encode(torch.FloatTensor(true_norm).to(device)).cpu().numpy()

        # sauvegarde
        out = os.path.join(ds_dir, f'vilar_dataset_{budget}.npz')
        np.savez_compressed(
            out,
            theta_norm=theta_norm,
            ts_embeddings=summary_stats,
            true_theta=true_theta,
            true_ts=true_ts,
            true_ts_embedding=true_emb,
            data_scaler=data_scaler,
            theta_scaler=MinMaxScaler().fit(theta)
        )
        print(f"✔ Sauvegardé → {out}")

if __name__ == '__main__':
    main()