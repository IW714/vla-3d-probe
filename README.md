# vla-3d-probe

Does learning to act teach a model 3D? Linear probes for 3D information in
OpenVLA before and after robot training (COMP 790).

Checkpoints: **A** base Prismatic VLM, **B** `openvla/openvla-7b`,
**C** `openvla/openvla-7b-finetuned-libero-spatial`.

## Layout

| Folder | Runs where | What |
| --- | --- | --- |
| `sim/` | Mac, CPU | LIBERO demo replay to a frames + ground-truth file |
| `models/` | GPU box | One forward pass per frame, save activations per layer |
| `probes/` | anywhere | Ridge probes per (layer, target), R² vs layer plots |
| `data/` | (gitignored) | Demos, frames file, activations |

Two separate Python environments on purpose: the simulator stack and the model
stack have conflicting pins, and they only talk through files in `data/`.

## Sim env setup (Mac)

```bash
cd vla-3d-probe
uv venv -p 3.10 .venv-sim && source .venv-sim/bin/activate   # or: conda create -n libero python=3.10
uv pip install -r sim/requirements.txt                        # or: pip install -r ...

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
# LIBERO's `pip install -e .` is broken on current setuptools (no top-level
# __init__.py), so put it on the path directly instead:
echo "$PWD/third_party/LIBERO" > "$(python -c 'import site; print(site.getsitepackages()[0])')/libero.pth"
python -c "import libero.libero; print('libero ok')"
```

## Build the frames file

```bash
python sim/download_data.py                      # -> data/libero/libero_spatial/*.hdf5

# smoke test: 2 demos per task (~a few hundred frames), check it looks right
python sim/build_dataset.py --demo-dir data/libero/libero_spatial \
    --out data/frames_smoke.h5 --max-demos 2

# full: 10 tasks x 50 demos, every 10th step (~5k frames)
python sim/build_dataset.py --demo-dir data/libero/libero_spatial \
    --out data/libero_spatial_frames.h5 --stride 10
```

Rendering: on macOS leave `MUJOCO_GL` unset (robosuite picks the native
backend). On a headless Linux box use `MUJOCO_GL=egl`.

## Frames file (`data/*_frames.h5`)

All images are in **OpenVLA's LIBERO input orientation** (raw render rotated
180°), 256×256. OpenVLA resizes to 224 at inference; `models/` will do the same.

| Key | Shape | Notes |
| --- | --- | --- |
| `rgb` | N×256×256×3 uint8 | model input |
| `depth` | N×256×256 f16 | metres along camera axis |
| `seg` | N×256×256 i16 | object id per pixel, -1 = background/robot/table |
| `task_id`, `episode_id`, `demo_idx`, `t`, `instr_id` | N | split by `episode_id`, never by frame |
| `eef_world`, `eef_cam`, `eef_pix` | N×3, N×3, N×2 | gripper position; pix is [row, col] |
| `K`, `cam_T_world` | 3×3, N×4×4 | camera intrinsics, world→camera |
| `objects/{frame, obj_id, world, cam, pix, area}` | one row per (frame, object) | `area` = mask pixels, 0 if occluded |
| attrs `object_names`, `instructions`, `task_names` | | |

Object positions are body origins (inside or under the object), so `cam[:, 2]`
runs a few cm deeper than the surface depth at `pix`. That's expected.

## Known gotchas

- LIBERO prompts for a dataset path on first import; `sim/` scripts write
  `~/.libero/config.yaml` first so it never does.
- `torch>=2.6` refuses LIBERO's `.pruned_init` files (`weights_only`). The
  frames builder doesn't use them; if you need them, `torch.load(..., weights_only=False)`.
