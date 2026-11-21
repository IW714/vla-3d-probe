"""
transfer_probes.py  (model env)

Train probes on one suite (LIBERO-Spatial), test them on another (LIBERO-Goal),
for the objects both suites share (matched by name). Answers: does the depth
readout learned on scenes C was fine-tuned on still work on scenes it never saw,
and does C still beat B there?

  python probes/transfer_probes.py \
      --train-frames data/libero_spatial_frames.h5 --train-acts data/acts \
      --test-frames  data/libero_goal_frames.h5    --test-acts  data/acts_goal \
      --ckpts C B --out results_transfer

Modes. In "group" and "group_phase", each suite is centred on its OWN group
means (or per-group phase curves), features and targets alike, using all rows
of that suite. That removes layout offsets, which can't transfer between
different scenes, and scores whether the probe direction learned on Spatial
reads within-group 3D variation in Goal. The test suite's target means are used
only to define that within-group target, never to fit the probe.
  none         raw transfer: train-set standardisation applied to the test set
  group        centre by (task, object) means within each suite
  group_phase  centre by per-(task, object) cubic in episode phase

Ridge strength is chosen per target column on held-out Spatial episodes, then
the probe is refit on all Spatial rows. r2_folds: R2 on 5 disjoint groups of
Goal episodes, identical across checkpoints, for paired C-vs-B comparison.
"""

import argparse
import csv
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_probes import ALPHAS, TARGETS, Ridge, load_rows, r2_cols, residualize, to_np, torch  # noqa: E402

MODES = ["none", "group", "group_phase"]


def filter_rows(rows, keep):
    out = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in rows.items() if k != "y"}
    out["y"] = {t: y[keep] for t, y in rows["y"].items()}
    return out


def suite(frames_path, acts_path, shared):
    f = h5py.File(frames_path, "r")
    a = h5py.File(acts_path, "r")
    names = list(f.attrs["object_names"])
    obj, frame = load_rows(f, a["patch_w"][()].astype(np.float32), 0.5)
    keep = np.isin(np.array(names)[obj["obj"]], shared)
    obj = filter_rows(obj, keep)
    obj["name"] = np.array(names)[obj["obj"]]
    return a, obj, frame


def features(a, L, feat, rows):
    raw = a[L][feat][()]
    X = raw[rows["frame"], rows["obj"]] if feat == "obj_pool" else raw[rows["frame"]]
    return X.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frames", default="data/libero_spatial_frames.h5")
    ap.add_argument("--train-acts", default="data/acts", help="folder with acts_<ckpt>.h5")
    ap.add_argument("--test-frames", default="data/libero_goal_frames.h5")
    ap.add_argument("--test-acts", default="data/acts_goal")
    ap.add_argument("--ckpts", nargs="+", default=["C", "B"])
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--modes", nargs="*", default=MODES)
    ap.add_argument("--layers", nargs="*", default=None)
    ap.add_argument("--out", default="results_transfer")
    args = ap.parse_args()

    tr_names = list(h5py.File(args.train_frames, "r").attrs["object_names"])
    te_names = list(h5py.File(args.test_frames, "r").attrs["object_names"])
    shared = [n for n in tr_names if n in te_names]
    print("shared objects:", shared)

    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    R = Ridge(dev)
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "transfer_results.csv")
    fields = ["ckpt", "layer", "feature", "target", "mode", "scope", "r2", "n",
              "r2_fold_mean", "r2_fold_std", "n_folds", "r2_folds"]
    w = csv.DictWriter(open(csv_path, "w", newline=""), fieldnames=fields)
    w.writeheader()

    for ckpt in args.ckpts:
        a_tr, obj_tr, frm_tr = suite(args.train_frames, os.path.join(args.train_acts, f"acts_{ckpt}.h5"), shared)
        a_te, obj_te, frm_te = suite(args.test_frames, os.path.join(args.test_acts, f"acts_{ckpt}.h5"), shared)
        layers = [L for L in a_tr.attrs["layers"] if args.layers is None or L in args.layers]
        print(f"\n== {ckpt}: train {len(obj_tr['frame'])} object rows / {len(frm_tr['frame'])} frames, "
              f"test {len(obj_te['frame'])} / {len(frm_te['frame'])}, device {dev}")

        for L in layers:
            for feat in ["obj_pool", "mean_img", "last_tok"]:
                if feat not in a_tr[L]:
                    continue
                tgts = [t for t in args.targets if feat in TARGETS[t]["feats"]]
                if not tgts:
                    continue
                lvl = TARGETS[tgts[0]]["level"]
                rtr, rte = (obj_tr, obj_te) if lvl == "obj" else (frm_tr, frm_te)
                Xtr0, Xte0 = R.t(features(a_tr, L, feat, rtr)), R.t(features(a_te, L, feat, rte))
                Ytr0 = np.concatenate([rtr["y"][t] for t in tgts], 1)
                Yte0 = np.concatenate([rte["y"][t] for t in tgts], 1)
                dims = np.cumsum([0] + [rtr["y"][t].shape[1] for t in tgts])
                all_tr, all_te = np.ones(len(Ytr0), bool), np.ones(len(Yte0), bool)

                for mode in args.modes:
                    if mode == "none":
                        Xtr, Xte, Ytr, Yte = Xtr0, Xte0, Ytr0, Yte0
                    else:  # each suite centred on its own groups
                        Xtr = residualize(rtr, Xtr0, all_tr, mode, "episode")
                        Xte = residualize(rte, Xte0, all_te, mode, "episode")
                        Ytr = residualize(rtr, Ytr0, all_tr, mode, "episode")
                        Yte = residualize(rte, Yte0, all_te, mode, "episode")
                    Ytr_d = R.t(Ytr)
                    # choose lambda on held-out Spatial episodes
                    val = (rtr["episode"] // 5) % 4 == 0
                    inner = R.fit(Xtr[R.idx(~val)], Ytr_d[R.idx(~val)])
                    Zv, yv = inner.proj(Xtr[R.idx(val)]), Ytr[val]
                    best, best_r2 = np.full(Ytr.shape[1], ALPHAS[0]), np.full(Ytr.shape[1], -np.inf)
                    for al in ALPHAS:
                        r = r2_cols(yv, to_np(Zv @ (inner.A / (inner.s[:, None] + al * inner.n)) + inner.ym))
                        better = r > best_r2
                        best[better], best_r2[better] = al, r[better]
                    fit = R.fit(Xtr, Ytr_d)
                    P = to_np(fit.predict(Xte, R.t(best * fit.n)))

                    chunk = rte["episode"] % 5
                    scopes = [("all", np.ones(len(Yte), bool))]
                    if lvl == "obj":
                        scopes += [(n, rte["name"] == n) for n in shared]
                    for j, t in enumerate(tgts):
                        idx = slice(dims[j], dims[j + 1])
                        for sc, m in scopes:
                            if m.sum() < 20:
                                continue
                            r = r2_cols(Yte[m][:, idx], P[m][:, idx]).mean()
                            per = [r2_cols(Yte[m & (chunk == k)][:, idx], P[m & (chunk == k)][:, idx]).mean()
                                   for k in range(5) if (m & (chunk == k)).sum() >= 20]
                            w.writerow(dict(ckpt=ckpt, layer=L, feature=feat, target=t, mode=mode, scope=sc,
                                            r2=round(float(r), 5), n=int(m.sum()),
                                            r2_fold_mean=round(float(np.mean(per)), 5) if per else "",
                                            r2_fold_std=round(float(np.std(per)), 5) if per else "",
                                            n_folds=len(per), r2_folds=";".join(f"{x:.5f}" for x in per)))
            print(f"  {L:8s} done", flush=True)
    print(f"\nwrote {csv_path}")
    plot(csv_path, args.out)


def plot(csv_path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in csv.DictReader(open(csv_path)) if r["scope"] == "all"]
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    for key in sorted({(r["target"], r["mode"], r["feature"]) for r in rows}):
        sel = [r for r in rows if (r["target"], r["mode"], r["feature"]) == key]
        fig, ax = plt.subplots(figsize=(7, 4))
        for ckpt in sorted({r["ckpt"] for r in sel}):
            rs = [r for r in sel if r["ckpt"] == ckpt]
            ys = np.array([float(r["r2"]) for r in rs])
            sd = np.array([float(r["r2_fold_std"] or 0) for r in rs])
            ax.plot([r["layer"] for r in rs], ys, marker="o", label=ckpt)
            ax.fill_between(range(len(rs)), ys - sd, ys + sd, alpha=0.2)
        ax.set_title(f"TRANSFER train Spatial -> test Goal | {key[0]} | centred: {key[1]} | {key[2]}", fontsize=9)
        ax.set_ylabel("R2 on Goal")
        ax.set_xlabel("layer")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "plots", f"transfer__{key[0]}__{key[1]}__{key[2]}.png"), dpi=120)
        plt.close(fig)
    print(f"plots in {os.path.join(out_dir, 'plots')}")


if __name__ == "__main__":
    main()
