"""
run_probes.py  (model env; uses the GPU via torch if available)

Linear (ridge) probes from saved activations to 3D ground truth, for every
(checkpoint, layer, feature, target, baseline mode, split). Writes one long
CSV plus R2-vs-layer plots.

  python probes/run_probes.py data/acts/acts_C.h5 data/acts/acts_B.h5

Targets (see TARGETS):
  obj_depth      object camera-frame z                     [object-pooled features]
  obj_xyz        object camera-frame x, y, z (mean R2)     [object-pooled]
  obj_pix        object pixel row, col: the 2D control     [object-pooled]
  grip_obj_dist  gripper-to-object distance (world)        [object-pooled]
  grip_xyz       gripper camera-frame x, y, z              [mean_img, last_tok]

Baseline modes. A baseline uses NO activations and is fit on the training
fold only. It is removed from BOTH the target and the features (Frisch-Waugh /
fixed effects), so the probe is scored only on within-group variation:
  none         subtract the training mean (= plain R2 on the raw target)
  group        subtract the group mean. Groups: (task, object) for object
               targets, task for gripper targets. Under the task split the
               held-out task is unseen, so groups are object only / global.
  group_phase  subtract a per-group cubic in episode phase t / T, i.e. what
               "which object, which task, how far into the demo" predicts
The layer="baseline" rows in the CSV give the R2 of the baseline alone on the
raw target, to read the residual R2s against.

Splits (r2 = predictions pooled over held-out folds, then one R2;
r2_fold_mean / r2_fold_std = R2 computed per fold, to judge how much a
difference between checkpoints could be split noise; r2_folds = the per-fold
values. Folds are identical across checkpoints, so compare checkpoints by the
per-fold DIFFERENCE, which is less noisy than comparing two spreads):
  episode   5 folds by episode id
  task      leave one task out (10 folds)

Ridge strength is chosen per column from a grid by an inner held-out set that
mirrors the outer split (other episodes, or another task).
"""

import argparse
import csv
import os
import time

import h5py
import numpy as np

try:
    import torch
except ImportError:  # numpy fallback, much slower
    torch = None

TARGETS = {
    "obj_depth": dict(level="obj", feats=["obj_pool"]),
    "obj_xyz": dict(level="obj", feats=["obj_pool"]),
    "obj_pix": dict(level="obj", feats=["obj_pool"]),
    "grip_obj_dist": dict(level="obj", feats=["obj_pool"]),
    "grip_xyz": dict(level="frame", feats=["mean_img", "last_tok"]),
}
MODES = ["none", "group", "group_phase"]
SPLITS = ["episode", "task"]
ALPHAS = np.logspace(-6, 1, 15)  # lambda = alpha * n_train on standardised features


# ---------------------------------------------------------------- data
def load_rows(f, patch_w, min_cov):
    """Row tables for object-level and frame-level targets."""
    F = {k: f[k][()] for k in ["task_id", "episode_id", "t", "eef_world", "eef_cam"]}
    O = {k: f["objects"][k][()] for k in ["frame", "obj_id", "world", "cam", "pix"]}
    ep = F["episode_id"]
    T = np.zeros(ep.max() + 1)
    np.maximum.at(T, ep, F["t"])
    phase = F["t"] / np.maximum(T[ep], 1)

    cov = patch_w.sum(-1)[O["frame"], O["obj_id"]]
    keep = cov >= min_cov
    fr, oid = O["frame"][keep], O["obj_id"][keep]
    obj = dict(
        frame=fr, obj=oid, task=F["task_id"][fr], episode=ep[fr], phase=phase[fr],
        group=F["task_id"][fr] * 100 + oid, level="obj",
        y=dict(
            obj_depth=O["cam"][keep][:, 2:3],
            obj_xyz=O["cam"][keep],
            obj_pix=O["pix"][keep].astype(np.float64),
            grip_obj_dist=np.linalg.norm(F["eef_world"][fr] - O["world"][keep], axis=1, keepdims=True),
        ),
    )
    n = len(ep)
    frame = dict(
        frame=np.arange(n), obj=np.full(n, -1), task=F["task_id"], episode=ep, phase=phase,
        group=F["task_id"].copy(), level="frame", y=dict(grip_xyz=F["eef_cam"]),
    )
    return obj, frame


def folds(rows, split):
    """Yield (name, train, inner_train, inner_val, test) boolean masks."""
    ep, task = rows["episode"], rows["task"]
    tasks = np.unique(task)
    ks = range(5) if split == "episode" else tasks
    for k in ks:
        if split == "episode":
            test = ep % 5 == k
            val_in = (ep // 5) % 4 == 0           # other episodes, independent of the outer fold
        else:
            test = task == k
            val_in = task == tasks[(np.searchsorted(tasks, k) + 1) % len(tasks)]  # another task
        train = ~test
        yield k, train, train & ~val_in, train & val_in, test


def groups_for(rows, split):
    """Baseline groups. Episode split: (task, object), or task for gripper
    targets. Task split: the held-out task is unseen, so object only (gripper
    targets: one global group)."""
    if split == "episode":
        return rows["group"]
    return rows["obj"] if rows["level"] == "obj" else np.zeros_like(rows["group"])


def residualize(rows, M, train, mode, split):
    """M minus its baseline fit on train rows. Works on numpy arrays or torch
    tensors, and is applied to BOTH targets and features (Frisch-Waugh): the
    probe then only sees within-group variation on either side."""
    lib = torch if (torch and isinstance(M, torch.Tensor)) else np
    def T(x):  # values
        return torch.as_tensor(x, dtype=M.dtype, device=M.device) if lib is torch else x

    def I(i):  # row indices
        return torch.as_tensor(i, dtype=torch.long, device=M.device) if lib is torch else i
    if mode == "none":
        return M - M[I(np.nonzero(train)[0])].mean(0)
    g = groups_for(rows, split)
    ph = rows["phase"]
    out = M.clone() if lib is torch else M.copy()
    gm = M[I(np.nonzero(train)[0])].mean(0)
    for k in np.unique(g):
        all_i, tr_i = np.nonzero(g == k)[0], np.nonzero(train & (g == k))[0]
        if len(tr_i) == 0:
            out[I(all_i)] = M[I(all_i)] - gm
        elif mode == "group" or len(tr_i) < 12:
            out[I(all_i)] = M[I(all_i)] - M[I(tr_i)].mean(0)
        else:
            A = lambda i: T(np.stack([np.ones(len(i)), ph[i], ph[i] ** 2, ph[i] ** 3], 1))
            if lib is torch:
                coef = torch.linalg.lstsq(A(tr_i), M[I(tr_i)]).solution
            else:
                coef = np.linalg.lstsq(A(tr_i), M[tr_i], rcond=None)[0]
            out[I(all_i)] = M[I(all_i)] - A(all_i) @ coef
    return out


# ---------------------------------------------------------------- ridge
class Ridge:
    """Eigen-decomposition ridge: one eig per training set, any number of
    target columns and lambdas for free."""

    def __init__(self, dev):
        self.dev = dev

    def t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.dev) if torch else np.asarray(x, np.float32)

    def idx(self, mask):
        i = np.nonzero(mask)[0]
        return torch.as_tensor(i, device=self.dev) if torch else i

    def fit(self, X, Y):
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs = (X - mu) / sd
        ym = Y.mean(0)
        G = Xs.T @ Xs
        if torch:
            s, V = torch.linalg.eigh(G)
            s = s.clamp_min(0)
        else:
            s, V = np.linalg.eigh(G)
            s = np.maximum(s, 0)
        fit = Ridge(self.dev)
        fit.mu, fit.sd, fit.ym, fit.V, fit.s = mu, sd, ym, V, s
        fit.A = V.T @ (Xs.T @ (Y - ym))      # (d, m)
        fit.n = X.shape[0]
        return fit

    def predict(self, X, lam):
        """lam: (m,) per-column lambda, or a scalar."""
        Z = ((X - self.mu) / self.sd) @ self.V
        return Z @ (self.A / (self.s[:, None] + lam)) + self.ym

    def proj(self, X):
        return ((X - self.mu) / self.sd) @ self.V


def r2_cols(Y, P):
    ss = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ((Y - P) ** 2).sum(0) / ss


def to_np(x):
    return x.detach().cpu().numpy() if torch and isinstance(x, torch.Tensor) else np.asarray(x)


# ---------------------------------------------------------------- main loop
def run_feature(R, X, rows, tgts, modes, splits):
    """X (n,d) features on device. For each split and baseline mode, residualize
    targets and features on the training fold, pick lambda per target column on
    an inner held-out set, refit, predict the test fold. Returns
    {(split, mode): (Y_true, Y_pred)} with test predictions pooled over folds."""
    Yraw = np.concatenate([rows["y"][t] for t in tgts], 1)
    out = {}
    for split in splits:
        for mode in modes:
            Yt, Yp = np.zeros(Yraw.shape), np.full(Yraw.shape, np.nan)
            fold_id = np.full(len(Yraw), -1)
            for k, train, itr, ival, test in folds(rows, split):
                if test.sum() == 0 or ival.sum() == 0:
                    continue
                Xr = residualize(rows, X, train, mode, split)
                Y = residualize(rows, Yraw, train, mode, split)
                Yd = R.t(Y)
                ix = {n: R.idx(m) for n, m in dict(itr=itr, ival=ival, train=train, test=test).items()}
                inner = R.fit(Xr[ix["itr"]], Yd[ix["itr"]])
                Zv = inner.proj(Xr[ix["ival"]])
                yv = to_np(Yd[ix["ival"]])
                best, best_r2 = np.full(Y.shape[1], ALPHAS[0]), np.full(Y.shape[1], -np.inf)
                for a in ALPHAS:
                    r = r2_cols(yv, to_np(Zv @ (inner.A / (inner.s[:, None] + a * inner.n)) + inner.ym))
                    better = r > best_r2
                    best[better], best_r2[better] = a, r[better]
                outer = R.fit(Xr[ix["train"]], Yd[ix["train"]])
                Yp[test] = to_np(outer.predict(Xr[ix["test"]], R.t(best * outer.n)))
                Yt[test] = Y[test]
                fold_id[test] = k
                del Xr
            out[(split, mode)] = (Yt, Yp, fold_id)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acts", nargs="+", help="data/acts/acts_<ckpt>.h5 files")
    ap.add_argument("--frames", default="data/libero_spatial_frames.h5")
    ap.add_argument("--out", default="results")
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--modes", nargs="*", default=MODES)
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--layers", nargs="*", default=None, help="subset of layer names")
    ap.add_argument("--min-cov", type=float, default=0.5, help="min object coverage, in patches")
    args = ap.parse_args()

    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    R = Ridge(dev)
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "probe_results.csv")
    fields = ["ckpt", "layer", "feature", "target", "mode", "split", "scope", "r2", "n",
              "r2_fold_mean", "r2_fold_std", "n_folds", "r2_folds"]
    w = csv.DictWriter(open(csv_path, "w", newline=""), fieldnames=fields)
    w.writeheader()

    f = h5py.File(args.frames, "r")
    names = list(f.attrs["object_names"])
    for path in args.acts:
        a = h5py.File(path, "r")
        ckpt = os.path.basename(path).replace("acts_", "").replace(".h5", "")
        obj_rows, frame_rows = load_rows(f, a["patch_w"][()].astype(np.float32), args.min_cov)
        layers = [L for L in a.attrs["layers"] if args.layers is None or L in args.layers]
        print(f"\n== {ckpt}: {len(obj_rows['frame'])} object rows, {len(frame_rows['frame'])} frames, "
              f"{len(layers)} layers, device {dev}")

        # baseline-only R2 on the raw target (no activations); same for every ckpt
        for rows in (obj_rows, frame_rows):
            for t in [t for t in args.targets if t in rows["y"]]:
                y = rows["y"][t]
                for m in args.modes:
                    for split in args.splits:
                        P = np.zeros_like(y)
                        for k, train, _, _, test in folds(rows, split):
                            if test.sum():
                                P[test] = (y - residualize(rows, y, train, m, split))[test]
                        w.writerow(dict(ckpt=ckpt, layer="baseline", feature="none", target=t, mode=m,
                                        split=split, scope="all", r2=round(float(r2_cols(y, P).mean()), 5),
                                        n=len(y)))

        for L in layers:
            t0 = time.time()
            for feat in ["obj_pool", "mean_img", "last_tok"]:
                if feat not in a[L]:
                    continue
                tgts = [t for t in args.targets if feat in TARGETS[t]["feats"]]
                if not tgts:
                    continue
                rows = obj_rows if TARGETS[tgts[0]]["level"] == "obj" else frame_rows
                raw = a[L][feat][()]
                X = raw[rows["frame"], rows["obj"]] if feat == "obj_pool" else raw[rows["frame"]]
                X = R.t(X.astype(np.float32))
                dims = np.cumsum([0] + [rows["y"][t].shape[1] for t in tgts])
                res = run_feature(R, X, rows, tgts, args.modes, args.splits)
                for (split, m), (Yt, Yp, fid) in res.items():
                    for j, t in enumerate(tgts):
                        idx = slice(dims[j], dims[j + 1])
                        scopes = [("all", np.ones(len(Yt), bool))]
                        if rows["level"] == "obj":
                            scopes += [(names[o], rows["obj"] == o) for o in np.unique(rows["obj"])]
                        for sc, msk in scopes:
                            if msk.sum() < 20:
                                continue
                            r = r2_cols(Yt[msk][:, idx], Yp[msk][:, idx]).mean()
                            # spread across held-out folds: how much R2 moves with the split
                            per = [r2_cols(Yt[msk & (fid == k)][:, idx], Yp[msk & (fid == k)][:, idx]).mean()
                                   for k in np.unique(fid[msk]) if (msk & (fid == k)).sum() >= 20]
                            w.writerow(dict(ckpt=ckpt, layer=L, feature=feat, target=t, mode=m,
                                            split=split, scope=sc, r2=round(float(r), 5), n=int(msk.sum()),
                                            r2_fold_mean=round(float(np.mean(per)), 5) if per else "",
                                            r2_fold_std=round(float(np.std(per)), 5) if per else "",
                                            n_folds=len(per), r2_folds=";".join(f"{x:.5f}" for x in per)))
                del X
            print(f"  {L:8s} done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nwrote {csv_path}")
    plot(csv_path, args.out)


def plot(csv_path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path)))
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    probe = [r for r in rows if r["layer"] != "baseline" and r["scope"] == "all"]
    keys = sorted({(r["target"], r["mode"], r["split"], r["feature"]) for r in probe})
    for t, m, s, feat in keys:
        sel = [r for r in probe if (r["target"], r["mode"], r["split"], r["feature"]) == (t, m, s, feat)]
        fig, ax = plt.subplots(figsize=(7, 4))
        for ckpt in sorted({r["ckpt"] for r in sel}):
            rs = [r for r in sel if r["ckpt"] == ckpt]
            xs, ys = [r["layer"] for r in rs], np.array([float(r["r2"]) for r in rs])
            ax.plot(xs, ys, marker="o", label=ckpt)
            sd = np.array([float(r["r2_fold_std"] or 0) for r in rs])
            ax.fill_between(range(len(xs)), ys - sd, ys + sd, alpha=0.2)
        base = [r for r in rows if r["layer"] == "baseline" and (r["target"], r["mode"], r["split"]) == (t, m, s)]
        title = f"{t} | baseline removed: {m} | split: {s} | {feat}"
        if m != "none" and base:
            title += f"\n(baseline alone explains R2 = {float(base[0]['r2']):.3f} of the raw target)"
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("held-out R2")
        ax.set_xlabel("layer")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "plots", f"{t}__{m}__{s}__{feat}.png"), dpi=120)
        plt.close(fig)
    print(f"plots in {os.path.join(out_dir, 'plots')}")


if __name__ == "__main__":
    main()
