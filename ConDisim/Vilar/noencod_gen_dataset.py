import os
import numpy as np
from vilar_dataset import simulator, generate_data_parallel, Vilar_Oscillator
from gillespy2 import SSACSolver


# ---------------------------------------------------------------------
# ---  UTILITAIRES -----------------------------------------------------
# ---------------------------------------------------------------------
def simulate_batch(n, model, solver, n_procs=4):
    """
    Simule un lot de n trajectoires brutes (1,3,200) et renvoie :
        ts  : (n, 3, 200)  float32
        th  : (n, 15)      float32
    Les trajectoires invalides (NaN / Inf) sont retirées.
    """
    ts_raw, theta = generate_data_parallel(n, model, solver)
    ts_raw = ts_raw.squeeze(1).astype(np.float32)      # (n,3,200)
    theta  = theta.astype(np.float32)                  # (n,15)

    # Masque valable ↦ pas de NaN / Inf
    flat   = ts_raw.reshape(ts_raw.shape[0], -1)          # (n, 3×200)
    finite = np.isfinite(flat).all(axis=1)                # True si pas de ±Inf
    notnan = ~np.isnan(flat).any(axis=1)                  # True si pas de NaN
    valid  = finite & notnan
    return ts_raw[valid], theta[valid]


def collect_clean_dataset(budget, model, solver, batch_size=256):
    """
    Boucle jusqu’à obtenir `budget` trajectoires propres.
    Retourne ts_data (budget,3,200) et theta (budget,15).
    """
    ts_list, th_list = [], []
    total_valid, total_bad = 0, 0

    while total_valid < budget:
        good = min(batch_size, budget - total_valid)
        bad  = batch_size - good
        ts_batch, th_batch = simulate_batch(batch_size, model, solver)
        # Split des bons et des mauvais
        ts_good = ts_batch[:good]
        th_good = th_batch[:good]
        ts_bad  = ts_batch[good:]
        th_bad  = th_batch[good:]

        ts_list.append(ts_good)
        th_list.append(th_good)

        total_valid += ts_good.shape[0]
        total_bad   += ts_bad.shape[0]

        print(f"    · {ts_good.shape[0]:4d} valides  |  {ts_bad.shape[0]:4d} rejetées  |  total = {total_valid}/{budget}")

    # Concaténation pile la taille voulue
    ts_all = np.concatenate(ts_list, axis=0)[:budget]
    th_all = np.concatenate(th_list, axis=0)[:budget]
    assert ts_all.shape[0] == budget

    print(f"  ✔  Collecte terminée : {total_valid} valides  |  {total_bad} rejetées")
    return ts_all, th_all


# ---------------------------------------------------------------------
# ---  SCRIPT PRINCIPAL -----------------------------------------------
# ---------------------------------------------------------------------
def main():
    budgets_train = [20000]
    dataset_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    os.makedirs(dataset_dir, exist_ok=True)

    # Instanciation unique (réutilisée partout)
    model  = Vilar_Oscillator()
    solver = SSACSolver(model=model)

    # Trajectoire de référence (paramètres “vrais”)
    true_theta = np.array(
        [50, 500, 0.01, 50, 50, 5, 10, 0.5, 1, 0.2, 1, 1, 2, 50, 100],
        dtype=np.float32
    )
    print("Génération de la trajectoire de référence…")
    true_ts = simulator(true_theta, model, solver).squeeze(0).astype(np.float32)  # (3,200)
    print("  → OK :", true_ts.shape)

    # Boucle sur les budgets pour le training
    for budget in budgets_train:
        print(f"\n=== Train dataset budget {budget} ===")
        ts_data, theta = collect_clean_dataset(budget, model, solver, batch_size=256)
        # Sauvegarde
        file_name = os.path.join(dataset_dir, f'vilar_dataset_{budget}_noencod.npz')
        np.savez_compressed(
            file_name,
            ts_data=ts_data,           # (N,3,200)
            theta=theta,               # (N,15)
            true_ts=true_ts,           # (3,200)
            true_theta=true_theta      # (15,)
        )
        print(f"  → Training dataset sauvegardé dans  {file_name}")

if __name__ == '__main__':
    main()