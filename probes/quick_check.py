"""
quick_check.py  (model env; numpy + h5py only)

Sanity check for one activations file: shapes, NaNs, feature spread, and a
rough object-depth probe with and without the (task, object) baseline removed.
Fixed ridge strength, one 80/20 split by episode. For the real protocol use
run_probes.py.

  python probes/quick_check.py data/acts/acts_B.h5
"""

import argparse

import h5py
import numpy as np


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ridge_fit_predict(Xtr, ytr, Xte, lam=1e3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (ytr - ytr.mean()))
    return Xte @ w + ytr.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acts")
    ap.add_argument("--frames", default="data/libero_spatial_frames.h5")
    args = ap.parse_args()

    a, f = h5py.File(args.acts, "r"), h5py.File(args.frames, "r")
    layers = list(a.attrs["layers"])
    print(f"{args.acts}  model={a.attrs['model_id']}")

    print("\n[1] shapes / NaNs / spread")
    for L in layers:
        x = a[L]["obj_pool"][::50].astype(np.float32)
        print(f"  {L:8s} {a[L]['obj_pool'].shape}  nan={np.isnan(x).any()}  std={x.std():.3f}")

    o = f["objects"]
    fr, oid, z = o["frame"][()], o["obj_id"][()], o["cam"][:, 2]
    task = f["task_id"][()][fr]
    vis = a["patch_w"][()].astype(np.float32).sum(-1)[fr, oid] > 0.5
    test = f["episode_id"][()][fr] % 5 == 0
    tr, te = vis & ~test, vis & test

    key = task * 100 + oid
    means = {k: z[tr & (key == k)].mean() for k in np.unique(key[tr])}
    obj_means = {k: z[tr & (oid == k)].mean() for k in np.unique(oid[tr])}
    print("\n[2] baselines (no activations), R2 on raw depth")
    print(f"  per-object mean           {r2(z[te], np.array([obj_means[k] for k in oid[te]])):.3f}")
    print(f"  per-(task, object) mean   {r2(z[te], np.array([means.get(k, z[tr].mean()) for k in key[te]])):.3f}")

    resid = z - np.array([means.get(k, 0.0) for k in key])
    print(f"\n[3] probes  (residual depth std {resid[vis].std() * 100:.1f} cm)")
    print(f"  {'layer':8s} {'raw R2':>7s} {'resid R2':>9s}")
    for L in layers:
        X = a[L]["obj_pool"][()][fr, oid].astype(np.float32)
        raw = r2(z[te], ridge_fit_predict(X[tr], z[tr], X[te]))
        # remove the (task, object) mean from the FEATURES too, so the probe only
        # sees within-group variation on both sides (otherwise it has to undo a
        # task x object layout term that a linear map can't represent)
        Xr = X.copy()
        for k in np.unique(key):
            m = key == k
            Xr[m] -= X[tr & m].mean(0) if (tr & m).any() else X[m].mean(0)
        res = r2(resid[te], ridge_fit_predict(Xr[tr], resid[tr], Xr[te]))
        print(f"  {L:8s} {raw:7.3f} {res:9.3f}")


if __name__ == "__main__":
    main()
