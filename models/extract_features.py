"""
extract_features.py  (model env, GPU)

One forward pass per frame through a frozen checkpoint. Saves, per layer:
  obj_pool  (N, K, D)  activations averaged over the image patches each object covers
  mean_img  (N, D)     average over all 256 image-patch tokens
  last_tok  (N, D)     final prompt token (where the first action token is predicted)

Layers saved:
  vision    output of the fused DINOv2 + SigLIP encoder (2nd-to-last ViT block each), D = 2176
  proj      after the MLP projector, D = 4096 (== LLM layer 0 at the image positions)
  llm_L     output of Llama-2 decoder layer L (1..32), D = 4096

Also saves model-independent patch targets (same for every checkpoint):
  patch_w      (N, K, 256)  fraction of each 14x14 patch covered by object k
  patch_depth  (N, 256)     mean depth (m) over each patch

Usage
  # Mac, no model: check preprocessing + patch mapping, writes a PNG
  python models/extract_features.py --frames data/libero_spatial_frames.h5 --check

  # GPU box
  python models/extract_features.py --frames data/libero_spatial_frames.h5 --ckpt C
  python models/extract_features.py --frames data/libero_spatial_frames.h5 --ckpt B
"""

import argparse
import io
import math
import os
import time

import h5py
import numpy as np
from PIL import Image

CKPTS = {
    "B": "openvla/openvla-7b",
    "C": "openvla/openvla-7b-finetuned-libero-spatial",
    "A": "prism-dinosiglip-224px+7b",   # base Prismatic VLM OpenVLA started from; needs prismatic-vlms + Llama-2 access
}
PROMPT = "In: What action should the robot take to {}?\nOut:"
IMG = 224        # model input size
PATCH = 14       # ViT patch size -> 16 x 16 = 256 patches
GRID = IMG // PATCH
CROP_SCALE = 0.9  # area kept by OpenVLA's LIBERO center crop


# --------------------------------------------------------------------------
# Preprocessing. Mirrors openvla/experiments/robot (libero_utils.resize_image
# + openvla_utils center crop), with PIL in place of TensorFlow, so it is
# close to but not bit-identical with the original.
# --------------------------------------------------------------------------
def crop_box(size):
    side = math.sqrt(CROP_SCALE) * size
    off = (size - side) / 2
    return (off, off, off + side, off + side)


def preprocess_rgb(rgb, center_crop=True):
    """256x256 uint8 (already rotated 180) -> 224x224 PIL image the model sees."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)   # JPEG round trip, as in RLDS
    img = Image.open(buf).convert("RGB").resize((IMG, IMG), Image.LANCZOS)
    if center_crop:
        img = img.resize((IMG, IMG), Image.BILINEAR, box=crop_box(IMG))
    return img


def preprocess_label(arr, center_crop=True):
    """Apply the same geometry to a per-pixel map (seg or depth), nearest-neighbour."""
    img = Image.fromarray(arr).resize((IMG, IMG), Image.NEAREST)
    if center_crop:
        img = img.resize((IMG, IMG), Image.NEAREST, box=crop_box(IMG))
    return np.asarray(img)


def patch_targets(seg, depth, n_obj, center_crop=True):
    """Per-patch object coverage and mean depth on the 16x16 token grid.
    Patch p = (row r, col c) covers pixels [14r, 14r+14) x [14c, 14c+14) and is
    token index p = 16 r + c, the same row-major order the ViT uses."""
    s = preprocess_label(seg.astype(np.int16), center_crop).reshape(GRID, PATCH, GRID, PATCH)
    d = preprocess_label(depth.astype(np.float32), center_crop).reshape(GRID, PATCH, GRID, PATCH)
    w = np.stack([(s == k).mean(axis=(1, 3)) for k in range(n_obj)])  # (K,16,16)
    return w.reshape(n_obj, GRID * GRID), d.mean(axis=(1, 3)).reshape(-1)


def pool(h, w):
    """h (B,256,D) patch activations, w (B,K,256) coverage -> (B,K,D) weighted mean.
    Objects with no visible pixels come out as zeros; check patch_w.sum(-1) > 0."""
    mass = w.sum(-1, keepdims=True)
    return (w / np.maximum(mass, 1e-8)) @ h


# --------------------------------------------------------------------------
def run_check(f, args):
    names = list(f.attrs["object_names"])
    idx = np.linspace(0, len(f["rgb"]) - 1, 4).astype(int)
    tiles = []
    for i in idx:
        img = np.asarray(preprocess_rgb(f["rgb"][i], args.center_crop)).copy()
        w, pd = patch_targets(f["seg"][i], f["depth"][i], len(names), args.center_crop)
        img[::PATCH, :] = 255
        img[:, ::PATCH] = 255                                     # token grid
        cov = w.max(0).reshape(GRID, GRID)                         # any-object coverage
        heat = np.kron(cov, np.ones((PATCH, PATCH)))[..., None] * np.array([255, 60, 0])
        over = (0.55 * img + 0.45 * heat).astype(np.uint8)
        dimg = np.kron(pd.reshape(GRID, GRID), np.ones((PATCH, PATCH)))
        dimg = (255 * (1 - np.clip((dimg - 0.5) / 1.5, 0, 1)))[..., None].repeat(3, -1).astype(np.uint8)
        tiles.append(np.concatenate([img, over, dimg], 1))
        vis = [f"{names[k].rsplit('_', 1)[0]}={w[k].sum():.1f}" for k in range(len(names)) if w[k].sum() > 0]
        print(f"frame {i}: patch mass per object: {', '.join(vis)}")
    out = os.path.join(os.path.dirname(args.out_dir.rstrip("/")) or ".", "check_patches.png")
    Image.fromarray(np.concatenate(tiles, 0)).save(out)
    print(f"wrote {out}  (columns: model input + 16x16 grid | object coverage | patch depth)")


def load_openvla(model_id, dev, cap):
    """B and C: OpenVLA checkpoints through Hugging Face remote code."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True, trust_remote_code=True).to(dev).eval()
    vla.vision_backbone.register_forward_hook(lambda m, i, o: cap.__setitem__("vision", o))

    def forward(prompt, imgs):
        inputs = proc([prompt] * len(imgs), imgs).to(dev, dtype=torch.bfloat16)
        ids = inputs["input_ids"]
        if not torch.all(ids[:, -1] == 29871):   # predict_action appends this empty token
            ids = torch.cat([ids, torch.full((len(imgs), 1), 29871, device=dev, dtype=ids.dtype)], 1)
        with torch.no_grad():
            o = vla(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=inputs["pixel_values"],
                    output_hidden_states=True, use_cache=False, return_dict=True)
        return o.hidden_states

    return forward, vla.language_model.config.num_hidden_layers


def load_prismatic(model_id, dev, cap):
    """A: the base Prismatic VLM (no robot training) through the prismatic-vlms
    package. Same layout as OpenVLA: fused DINOv2+SigLIP patches projected and
    inserted right after BOS, Llama-2 7B on top. Fed the same prompt text and the
    same preprocessed 224px image as B and C; no trailing action token, since A
    was never trained to emit actions."""
    import torch
    from prismatic import load

    token = os.environ.get("HF_TOKEN")
    if token is None and os.path.exists(".hf_token"):
        token = open(".hf_token").read().strip()
    vlm = load(model_id, hf_token=token)
    vlm.to(dev, dtype=torch.bfloat16).eval()
    vlm.vision_backbone.register_forward_hook(lambda m, i, o: cap.__setitem__("vision", o))
    tf, tok = vlm.vision_backbone.image_transform, vlm.llm_backbone.tokenizer

    def forward(prompt, imgs):
        pv = [tf(im) for im in imgs]
        if isinstance(pv[0], dict):              # fused backbones return {"dino": ..., "siglip": ...}
            pv = {k: torch.stack([p[k] for p in pv]).to(dev, dtype=torch.bfloat16) for k in pv[0]}
        else:
            pv = torch.stack(pv).to(dev, dtype=torch.bfloat16)
        ids = tok(prompt, truncation=True, return_tensors="pt").input_ids.repeat(len(imgs), 1).to(dev)
        with torch.no_grad():
            o = vlm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv,
                    output_hidden_states=True, return_dict=True)
        return o.hidden_states

    llm = getattr(vlm.llm_backbone, "llm", None)
    n_llm = llm.config.num_hidden_layers if llm is not None else 32
    return forward, n_llm


def run_extract(f, args):
    import torch

    model_id = CKPTS[args.ckpt]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    # The vision encoder output is captured with a hook; projector and LLM
    # outputs come back in hidden_states.
    cap = {}
    loader = load_prismatic if args.ckpt == "A" else load_openvla
    forward, n_llm = loader(model_id, dev, cap)

    names = list(f.attrs["object_names"])
    instrs = list(f.attrs["instructions"])
    K, N = len(names), len(f["rgb"])
    llm_layers = list(range(args.layer_stride, n_llm + 1, args.layer_stride))
    if n_llm not in llm_layers:
        llm_layers.append(n_llm)
    layers = ["vision", "proj"] + [f"llm_{l}" for l in llm_layers]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"acts_{args.ckpt}.h5")
    out = h5py.File(out_path, "w")
    out.attrs.update(model_id=model_id, layers=layers, object_names=names,
                     center_crop=args.center_crop, frames_file=os.path.abspath(args.frames))
    pw = out.create_dataset("patch_w", (N, K, GRID * GRID), "f2")
    pdp = out.create_dataset("patch_depth", (N, GRID * GRID), "f2")
    dsets = {}

    # Group frames by instruction: same prompt -> same token length -> no padding.
    instr_id = f["instr_id"][()]
    t0, done = time.time(), 0
    for iid in np.unique(instr_id):
        prompt = PROMPT.format(instrs[iid].lower())
        rows = np.nonzero(instr_id == iid)[0]
        for b0 in range(0, len(rows), args.batch):
            r = rows[b0:b0 + args.batch]
            imgs, W = [], []
            for i in r:
                imgs.append(preprocess_rgb(f["rgb"][i], args.center_crop))
                w, d = patch_targets(f["seg"][i], f["depth"][i], K, args.center_crop)
                W.append(w)
                pw[i], pdp[i] = w, d
            W = np.stack(W).astype(np.float32)

            hs = forward(prompt, imgs)               # tuple: [embeddings, layer 1, ..., layer 32]
            if b0 == 0 and iid == instr_id.min():
                assert len(hs) == n_llm + 1 and hs[0].shape[1] > GRID * GRID, \
                    f"unexpected hidden states: {len(hs)} x {tuple(hs[0].shape)}"
                print(f"[{args.ckpt}] seq len {hs[0].shape[1]}, {len(hs) - 1} LLM layers, "
                      f"vision feats {tuple(cap['vision'].shape)}", flush=True)
            # hs[0] at the image positions is exactly the projector output
            feats = {"vision": cap["vision"], "proj": hs[0][:, 1:1 + GRID * GRID]}
            for l in llm_layers:
                feats[f"llm_{l}"] = hs[l]
            for name, h in feats.items():
                h = h.float().cpu().numpy()
                if name.startswith("llm_"):
                    img_tok, last = h[:, 1:1 + GRID * GRID], h[:, -1]   # image tokens follow BOS
                else:
                    img_tok, last = h, None
                if name not in dsets:
                    D = h.shape[-1]
                    g = out.create_group(name)
                    dsets[name] = (g.create_dataset("obj_pool", (N, K, D), "f2"),
                                   g.create_dataset("mean_img", (N, D), "f2"),
                                   g.create_dataset("last_tok", (N, D), "f2") if last is not None else None)
                dp, dm, dl = dsets[name]
                for j, i in enumerate(r):           # rows within one task are contiguous but write per row to be safe
                    dp[i] = pool(img_tok[j:j + 1], W[j:j + 1])[0]
                    dm[i] = img_tok[j].mean(0)
                    if dl is not None:
                        dl[i] = last[j]
            done += len(r)
            print(f"[{args.ckpt}] {done}/{N} frames  {done / (time.time() - t0):.1f} fr/s", flush=True)
    out.close()
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/libero_spatial_frames.h5")
    ap.add_argument("--ckpt", choices=sorted(CKPTS), default="C", help="A needs the prismatic-vlms env (see README)")
    ap.add_argument("--out-dir", default="data/acts")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--layer-stride", type=int, default=2, help="keep every k-th LLM layer (1 = all 32)")
    ap.add_argument("--no-center-crop", dest="center_crop", action="store_false")
    ap.add_argument("--check", action="store_true", help="preprocessing + patch mapping only, no model")
    args = ap.parse_args()
    with h5py.File(args.frames, "r") as f:
        run_check(f, args) if args.check else run_extract(f, args)


if __name__ == "__main__":
    main()
