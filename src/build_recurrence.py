
import numpy as np, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    data = np.load("outputs/data.npz")
    traj = data["traj"]  # [T, nx, ny]
    T = traj.shape[0]
    flat = traj.reshape(T, -1)
    # normalize
    flat = (flat - flat.mean(0, keepdims=True)) / (flat.std(0, keepdims=True) + 1e-6)

    # simple L2 distances (use chunking to keep memory small)
    knn_idx = np.zeros((T, args.k), dtype=int)
    for i in range(T):
        d = np.sum((flat - flat[i])**2, axis=1)
        order = np.argsort(d)
        # exclude self at index 0
        knn_idx[i] = order[1:args.k+1]

    out = Path("outputs"); out.mkdir(exist_ok=True, parents=True)
    np.savez(out/"recurrence.npz", knn=knn_idx)
    print(f"Saved {out/'recurrence.npz'} with k={args.k}")

if __name__ == "__main__":
    main()
