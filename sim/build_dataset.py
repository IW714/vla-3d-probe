"""
build_dataset.py  (sim env, CPU only)

Replays LIBERO demos by *setting* the saved MuJoCo state at a subsample of
timesteps, renders the agentview camera, and records ground truth for probing.
No policy is run. One output file is reused by every later experiment.

Everything image-shaped is stored in the orientation OpenVLA sees at LIBERO
eval time: the raw robosuite render rotated 180 degrees (img[::-1, ::-1]).
So rgb, depth, seg, and every pixel coordinate line up with the model input.
(The raw render is upside down; the 180 degree rotation fixes that and also
mirrors left/right. That mirrored view is what OpenVLA was trained on.)

Output (HDF5, one row per frame unless noted):
  rgb        (N,H,W,3) uint8
  depth      (N,H,W)   float16, metres along the camera axis
  seg        (N,H,W)   int16, global object id per pixel, -1 = not an object
  task_id, episode_id, demo_idx, t, instr_id          (N,)
  eef_world, eef_cam (N,3)   eef_pix (N,2) [row, col]
  gripper_qpos (N,2)
  K (3,3), cam_T_world (N,4,4)
  /objects/*  long format, one row per (frame, object):
     frame, obj_id, world (3), cam (3), pix (2), area (pixels in mask)
  attrs: object_names, instructions, task_names, orientation note
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np


def ensure_libero_config(datasets_dir):
    """LIBERO asks an interactive question on first import unless a config
    file exists. Write one that points at the repo's own bddl/init/assets."""
    import importlib.util
    import yaml

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("Cannot import `libero`. See README: add the LIBERO repo to a .pth file.")
    root = os.path.join(list(spec.submodule_search_locations)[0], "libero")
    cfg_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    cfg = os.path.join(cfg_dir, "config.yaml")
    if not os.path.exists(cfg):
        os.makedirs(cfg_dir, exist_ok=True)
        with open(cfg, "w") as f:
            yaml.safe_dump(
                {
                    "benchmark_root": root,
                    "bddl_files": os.path.join(root, "bddl_files"),
                    "init_states": os.path.join(root, "init_files"),
                    "datasets": os.path.abspath(datasets_dir),
                    "assets": os.path.join(root, "assets"),
                },
                f,
            )
        print(f"[setup] wrote {cfg}")


def to_model_frame(a):
    """Raw robosuite render -> OpenVLA input orientation (rotate 180)."""
    return a[::-1, ::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", required=True, help="folder of libero_spatial *_demo.hdf5 files")
    ap.add_argument("--out", default="data/libero_spatial_frames.h5")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--stride", type=int, default=10, help="keep every k-th timestep")
    ap.add_argument("--res", type=int, default=256, help="render size (OpenVLA renders 256, resizes to 224)")
    ap.add_argument("--max-demos", type=int, default=None, help="per task; small numbers for a smoke test")
    ap.add_argument("--tasks", type=int, nargs="*", default=None, help="task indices to include")
    args = ap.parse_args()

    ensure_libero_config(os.path.dirname(os.path.abspath(args.demo_dir)))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils as cu

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = args.tasks if args.tasks is not None else list(range(suite.n_tasks))
    H = W = args.res
    cam = "agentview"

    # Buffers. Small enough for a few thousand frames at 256px (~2 GB peak);
    # switch to incremental writes if you scale up a lot.
    F = {k: [] for k in ["rgb", "depth", "seg", "task_id", "episode_id", "demo_idx", "t",
                         "instr_id", "eef_world", "eef_cam", "eef_pix", "gripper_qpos", "cam_T_world"]}
    O = {k: [] for k in ["frame", "obj_id", "world", "cam", "pix", "area"]}
    object_names, instructions, task_names = [], [], []
    K = None
    episode_counter = 0

    for tid in task_ids:
        task = suite.get_task(tid)
        stem = os.path.splitext(task.bddl_file)[0]
        demo_path = os.path.join(args.demo_dir, f"{stem}_demo.hdf5")
        if not os.path.exists(demo_path):
            print(f"[skip] task {tid}: no {demo_path}")
            continue
        task_names.append(stem)
        instructions.append(task.language)
        instr_id = len(instructions) - 1

        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_names=[cam],
            camera_heights=H,
            camera_widths=W,
            camera_depths=True,
            camera_segmentations="element",  # per-pixel geom id
        )
        env.seed(0)
        env.reset()
        e, sim = env.env, env.env.sim
        m = sim.model

        # Map every geom to the LIBERO object that owns it, by walking up the
        # body tree until we hit one of the object root bodies.
        root_to_name = {b: n for n, b in e.obj_body_id.items()}
        for n in e.obj_body_id:
            if n not in object_names:
                object_names.append(n)
        geom_to_gid = np.full(m.ngeom + 1, -1, dtype=np.int16)  # +1 slot for background (-1)
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            while b != 0 and b not in root_to_name:
                b = m.body_parentid[b]
            if b in root_to_name:
                geom_to_gid[g] = object_names.index(root_to_name[b])

        with h5py.File(demo_path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
            if args.max_demos:
                demos = demos[: args.max_demos]
            for d_i, dk in enumerate(demos):
                states = f[f"data/{dk}/states"][()]
                expected = sim.get_state().flatten().shape[0]
                if states.shape[1] != expected:
                    sys.exit(f"State size {states.shape[1]} != env's {expected} for {demo_path}:{dk}")
                for t in range(0, len(states), args.stride):
                    obs = env.set_init_state(states[t])
                    frame = len(F["rgb"])

                    rgb = to_model_frame(obs[f"{cam}_image"])
                    depth = cu.get_real_depth_map(sim, obs[f"{cam}_depth"])[..., 0]
                    depth = to_model_frame(depth)
                    seg_geom = to_model_frame(obs[f"{cam}_segmentation_element"][..., 0])
                    seg = geom_to_gid[np.clip(seg_geom, -1, m.ngeom - 1)]

                    # Camera matrices. world_T_cam is the camera pose; invert for world->camera.
                    if K is None:
                        K = cu.get_camera_intrinsic_matrix(sim, cam, H, W)
                    world_T_cam = cu.get_camera_extrinsic_matrix(sim, cam)
                    cam_T_world = np.linalg.inv(world_T_cam)
                    full_P = cu.get_camera_transform_matrix(sim, cam, H, W)

                    def to_cam(p):
                        return (cam_T_world @ np.r_[p, 1.0])[:3]

                    def to_pix(p):
                        # robosuite returns [row, col] in the *upright* image;
                        # mirror the column to match the rotated model frame.
                        r, c = cu.project_points_from_world_to_camera(p[None], full_P, H, W)[0]
                        return np.array([r, W - 1 - c], dtype=np.int16)

                    eef = obs["robot0_eef_pos"]
                    F["rgb"].append(np.ascontiguousarray(rgb))
                    F["depth"].append(depth.astype(np.float16))
                    F["seg"].append(seg.astype(np.int16))
                    F["task_id"].append(tid)
                    F["episode_id"].append(episode_counter)
                    F["demo_idx"].append(d_i)
                    F["t"].append(t)
                    F["instr_id"].append(instr_id)
                    F["eef_world"].append(eef)
                    F["eef_cam"].append(to_cam(eef))
                    F["eef_pix"].append(to_pix(eef))
                    F["gripper_qpos"].append(obs["robot0_gripper_qpos"])
                    F["cam_T_world"].append(cam_T_world)

                    for name, b in e.obj_body_id.items():
                        gid = object_names.index(name)
                        p = sim.data.body_xpos[b].copy()
                        O["frame"].append(frame)
                        O["obj_id"].append(gid)
                        O["world"].append(p)
                        O["cam"].append(to_cam(p))
                        O["pix"].append(to_pix(p))
                        O["area"].append(int((seg == gid).sum()))
                episode_counter += 1
                print(f"task {tid} demo {d_i}: {len(F['rgb'])} frames so far", flush=True)
        env.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.out, "w") as out:
        img_kw = dict(compression="gzip", compression_opts=4, chunks=True)
        for k, v in F.items():
            arr = np.stack(v) if k in ("rgb", "depth", "seg", "eef_world", "eef_cam", "eef_pix",
                                      "gripper_qpos", "cam_T_world") else np.asarray(v)
            out.create_dataset(k, data=arr, **(img_kw if arr.ndim >= 3 else {}))
        g = out.create_group("objects")
        for k, v in O.items():
            g.create_dataset(k, data=np.asarray(v))
        out.create_dataset("K", data=K)
        out.attrs["object_names"] = object_names
        out.attrs["instructions"] = instructions
        out.attrs["task_names"] = task_names
        out.attrs["orientation"] = "OpenVLA LIBERO input frame: raw robosuite render rotated 180 deg"
        out.attrs["stride"] = args.stride
    print(f"wrote {args.out}: {len(F['rgb'])} frames, {len(O['frame'])} object rows, "
          f"{len(object_names)} objects")


if __name__ == "__main__":
    main()
