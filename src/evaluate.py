
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def psnr(a, b):
    mse = np.mean((a-b)**2)
    if mse == 0: return 99.0
    mx = max(a.max(), b.max())
    return 20*np.log10(mx / np.sqrt(mse + 1e-12))

def main():
    d = np.load("outputs/rcfd_rollout.npz")
    preds, gt = d["preds"], d["gt"]
    T = preds.shape[0]
    scores = [psnr(preds[t], gt[t]) for t in range(T)]
    print(f"Mean PSNR over horizon={T}: {np.mean(scores):.2f} dB")

    # plot metric vs time
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(T), scores)
    plt.xlabel("t"); plt.ylabel("PSNR (dB)")
    plt.title("RCFD fidelity vs horizon")
    out = Path("outputs"); out.mkdir(exist_ok=True, parents=True)
    plt.savefig(out/"psnr_vs_time.png", dpi=150); plt.close()

if __name__ == "__main__":
    main()
