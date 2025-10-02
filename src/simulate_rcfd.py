
import numpy as np, argparse, time
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    data = np.load("outputs/data.npz")
    traj, u, v = data["traj"], data["u"], data["v"]
    dt = float(data["dt"]); diff = float(data["diff"])
    knn = np.load("outputs/recurrence.npz")["knn"]

    start_idx = 50
    gt = traj[start_idx:start_idx+args.horizon]  # ground truth slice
    cur = traj[start_idx].copy()

    # RCFD rollout: at each step, jump to the nearest neighbor of current state's index and blend small step
    state_id = start_idx
    preds = []
    t0 = time.time()
    for t in range(args.horizon):
        cand = knn[state_id][0]  # best neighbor
        # linear blend between current and candidate next state (cheap "local linear model")
        alpha = 0.85
        nxt = alpha*traj[cand] + (1-alpha)*cur
        preds.append(nxt.copy())
        # pick new state id as cand+1 to approximate forward time
        state_id = min(cand+1, traj.shape[0]-1)
        cur = nxt
    t1 = time.time()
    print(f"RCFD rollout {args.horizon} steps in {t1-t0:.4f}s")

    preds = np.array(preds)

    out = Path("outputs"); out.mkdir(exist_ok=True, parents=True)
    np.savez(out/"rcfd_rollout.npz", preds=preds, gt=gt)

    # quick plots
    plt.figure()
    plt.imshow(gt[-1], origin="lower", aspect="auto")
    plt.colorbar(); plt.title("Ground Truth (final)")
    plt.savefig(out/"gt_final.png", dpi=150); plt.close()

    plt.figure()
    plt.imshow(preds[-1], origin="lower", aspect="auto")
    plt.colorbar(); plt.title("RCFD (final)")
    plt.savefig(out/"rcfd_final.png", dpi=150); plt.close()

if __name__ == "__main__":
    main()
