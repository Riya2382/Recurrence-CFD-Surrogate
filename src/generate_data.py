
import numpy as np
import argparse
from pathlib import Path

def double_gyre_velocity(nx, ny, Lx=2.0, Ly=1.0, A=1.0):
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    # classic steady double-gyre-like streamfunction-inspired field
    u =  A * np.sin(np.pi * X / Lx) * np.cos(np.pi * Y / Ly)
    v = -A * np.cos(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
    return u, v

def advect_diffuse(c, u, v, dx, dy, dt, diff):
    # upwind advection + diffusion (explicit)
    cx = np.zeros_like(c)
    cy = np.zeros_like(c)
    # upwind in x
    cx[:,1:-1] = np.where(u[:,1:-1] > 0,
                          (c[:,1:-1]-c[:,0:-2])/dx,
                          (c[:,2:]-c[:,1:-1])/dx)
    # upwind in y
    cy[1:-1,:] = np.where(v[1:-1,:] > 0,
                          (c[1:-1,:]-c[0:-2,:])/dy,
                          (c[2:,:]-c[1:-1,:])/dy)
    lap = (np.roll(c,1,0)-2*c+np.roll(c,-1,0))/dx**2 + (np.roll(c,1,1)-2*c+np.roll(c,-1,1))/dy**2
    c_new = c - dt*(u*cx + v*cy) + dt*diff*lap
    # periodic boundaries
    return c_new

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--diff", type=float, default=1e-3)
    args = ap.parse_args()

    nx, ny = args.nx, args.ny
    Lx, Ly = 2.0, 1.0
    dx, dy = Lx/(nx-1), Ly/(ny-1)
    u, v = double_gyre_velocity(nx, ny, Lx, Ly)

    # initial blob
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    c = np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.02)

    traj = [c.copy()]
    for _ in range(args.steps):
        c = advect_diffuse(c, u, v, dx, dy, args.dt, args.diff)
        traj.append(c.copy())
    traj = np.array(traj)  # [T+1, nx, ny]

    out = Path("outputs"); out.mkdir(parents=True, exist_ok=True)
    np.savez(out/"data.npz", traj=traj, u=u, v=v, dx=dx, dy=dy, dt=args.dt, diff=args.diff)
    print(f"Saved {out/'data.npz'} with shape {traj.shape}")

if __name__ == "__main__":
    main()
