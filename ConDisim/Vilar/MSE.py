import argparse
import numpy as np
import sys

def compute_mse(npz_path, samples_key='posterior_samples', true_key='true_theta'):
    # Chargement du fichier
    data = np.load(npz_path)
    if samples_key not in data or true_key not in data:
        raise KeyError(f"Clés attendues '{samples_key}' et '{true_key}' non trouvées dans {npz_path}. "
                       f"Clés disponibles : {list(data.keys())}")

    samples   = data[samples_key]   # shape (N_samples, D)
    true_vals = data[true_key]      # shape (D,) ou (N_datasets, D)

    # Si true_vals a une dimension supplémentaire (ex. (1, D)), on l'aplatie
    true_vals = np.reshape(true_vals, (-1, true_vals.shape[-1]))
    if true_vals.shape[0] == 1:
        true_vals = true_vals[0]    # devient (D,)

    # Vérification de compatibilité
    N, D = samples.shape
    if true_vals.shape != (D,):
        raise ValueError(f"Incompatibilité de forme : samples a shape {(N,D)}, "
                         f"true_theta a shape {true_vals.shape}")

    # Calculs MSE
    # MSE par dimension
    mse_per_dim = np.mean((samples - true_vals[np.newaxis, :])**2, axis=0)
    # MSE globale
    mse_global  = np.mean(mse_per_dim)

    return mse_per_dim, mse_global

def main():
    parser = argparse.ArgumentParser(
        description="Calcule le MSE entre true_theta et posterior_samples dans un fichier .npz.")
    parser.add_argument("npz_file", help="Chemin vers le fichier .npz")
    parser.add_argument("--samples_key", default="posterior_samples",
                        help="Nom de la clé contenant les samples (default: posterior_samples)")
    parser.add_argument("--true_key", default="true_theta",
                        help="Nom de la clé contenant les paramètres vrais (default: true_theta)")
    args = parser.parse_args()

    try:
        mse_per_dim, mse_global = compute_mse(
            args.npz_file, args.samples_key, args.true_key
        )
    except Exception as e:
        print(f"Erreur : {e}", file=sys.stderr)
        sys.exit(1)

    D = mse_per_dim.shape[0]
    print(f"MSE global : {mse_global:.6e}")
    print("MSE par dimension :")
    for d in range(D):
        print(f"  dim {d:2d}: {mse_per_dim[d]:.6e}")

if __name__ == "__main__":
    main()