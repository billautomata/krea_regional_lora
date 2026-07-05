r"""
RegionalCharacterLora  —  ComfyUI custom node (Krea2 / Flux-2 single-stream DiT)
================================================================================
Place two trained character LoRAs into ONE coherent image, each concentrated in
its own spatial region, WITHOUT identity blend — and without compositing or
inpainting. The base model still generates a single image with full attention
across the whole token sequence; this node injects each character LoRA's
*activation delta* (up @ down * scale) into the image tokens that fall inside
that character's region, and (at text_strength) into the shared text tokens —
the trigger-word pathway a global LoRA load would also transform.

WHY THIS WORKS (vs. a LoRA stack)
  A normal LoRA load MERGES a low-rank weight delta into the model globally, so
  every pixel carries both identities -> blend. Here the deltas are never merged;
  they are added at forward time and masked to a token region, so identity A only
  reaches region A's tokens.

INSTALL
  <ComfyUI>/custom_nodes/regional_character_lora/__init__.py   (this file)
  Restart ComfyUI -> Add Node -> conditioning/regional -> "Krea2 Regional Character LoRA".
  Run recon_krea2.py FIRST (same folder) to confirm your LoRA/UNet key stems line up.

WIRING (replace the character LoRAs in your stack)
  UNETLoader -> Krea2T-Enhancer -> NAG -> ModelSamplingFlux
     -> LoraLoaderStack (global style/NSFW LoRAs, stay here)
     -> Krea2 Regional Character LoRA (two character LoRAs, per-region)
     -> KSampler

SELF-DISCOVERING (does not hardcode recon answers)
  * LoRA target layers are read from the file; matched to live model modules by
    normalised name (collapses '_' vs '.', strips lora_unet_/diffusion_model_).
  * Fused-qkv vs separate q/k/v is irrelevant: we patch whatever Linear the LoRA
    actually targets.
  * Text-token offset is measured from the real activation at hook time
    (n_text = seq_len - n_image_tokens); image tokens assumed to be the trailing
    block ([text | image], per Krea2). Nothing about 512 is hardcoded.
"""

import os
import re
import math
import json

import torch
import safetensors.torch

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    import comfy.patcher_extension as _pext
    _WRAPPER_ENUM = _pext.WrappersMP.DIFFUSION_MODEL
except Exception:
    _pext = None
    _WRAPPER_ENUM = "diffusion_model"

WRAPPER_KEY = "regional_character_lora"
__version__ = "1.0.0-beta"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _lora_dir_list():
    if folder_paths is not None:
        try:
            return folder_paths.get_filename_list("loras")
        except Exception:
            pass
    return []


def _resolve_lora_path(name):
    if folder_paths is not None:
        try:
            p = folder_paths.get_full_path("loras", name)
            if p:
                return p
        except Exception:
            pass
    return name  # assume an absolute path was given


# Diffusers-style Krea2 LoRA names -> live module names (mirrors
# comfy.utils.krea2_to_diffusers, applied on collapsed signatures).
# Order matters: longer patterns first.
_SIG_RENAMES = (
    ("transformerblocks", "blocks"),
    ("textfusion", "txtfusion"),
    ("attntoout0", "attnwo"),
    ("attntoout", "attnwo"),
    ("attntogate", "attngate"),
    ("attntoq", "attnwq"),
    ("attntok", "attnwk"),
    ("attntov", "attnwv"),
    ("ffgate", "mlpgate"),
    ("ffup", "mlpup"),
    ("ffdown", "mlpdown"),
    ("imgin", "first"),
    ("finallayerlinear", "lastlinear"),
    ("timeembedlinear1", "tmlp0"),
    ("timeembedlinear2", "tmlp2"),
    ("timemodproj", "tproj1"),
    ("txtinlinear1", "txtmlp1"),
    ("txtinlinear2", "txtmlp3"),
)


def _norm(s):
    """Collapse a key/module name to a comparison signature."""
    s = s.lower()
    for pre in ("lora_unet_", "lora_te_", "lora_", "diffusion_model.",
                "diffusion_model_", "transformer.", "model.diffusion_model.",
                "model.", "base_model."):
        if s.startswith(pre):
            s = s[len(pre):]
    s = s.replace(".", "").replace("_", "")
    for old, new in _SIG_RENAMES:
        if old in s:
            s = s.replace(old, new)
    return s


def _load_lora_matrices(path):
    """Return { module_sig: {'down':T, 'up':T, 'alpha':float} } in fp32 on CPU.
    Handles kohya (lora_down/lora_up + alpha) and diffusers (lora_A/lora_B)."""
    sd = safetensors.torch.load_file(path)
    groups = {}
    alphas = {}
    for k, v in sd.items():
        if k.endswith(".alpha") or k.endswith("alpha"):
            base = re.sub(r"\.?alpha$", "", k)
            alphas[base] = float(v.flatten()[0].item())
            continue
        m = re.search(r"(.*?)\.(lora_down|lora_A)\.weight$", k)
        if m:
            groups.setdefault(m.group(1), {})["down"] = v.float()
            continue
        m = re.search(r"(.*?)\.(lora_up|lora_B)\.weight$", k)
        if m:
            groups.setdefault(m.group(1), {})["up"] = v.float()
            continue

    out = {}
    for base, mats in groups.items():
        if "down" not in mats or "up" not in mats:
            continue
        down, up = mats["down"], mats["up"]
        rank = down.shape[0]
        alpha = alphas.get(base, alphas.get(base + ".alpha", float(rank)))
        out[_norm(base)] = {
            "down": down,                       # [rank, in]
            "up": up,                           # [out, rank]
            "scale": float(alpha) / float(rank),
            "_dbg": base,
        }
    if not out:
        print(f"[RegionalCharacterLora] !! '{os.path.basename(path)}' has no "
              f"lora_down/up or lora_A/B weight pairs ({len(sd)} keys; LoHa/"
              f"LyCORIS/diff formats are unsupported) - it will have NO effect.")
    return out


def _load_optional_lora(name):
    """"None" slot -> None; otherwise same loading as a required lora widget."""
    if not name or name == "None":
        return None
    return _load_lora_matrices(_resolve_lora_path(name))


def _iter_named_linears(module):
    for name, sub in module.named_modules():
        if isinstance(sub, torch.nn.Linear) or hasattr(sub, "weight"):
            yield name, sub


def _build_token_grid(w, h):
    # Krea2: VAE f8 (/8) then patch=2 (/2) -> /16 total. Row-major raster.
    cols = max(1, w // 16)
    rows = max(1, h // 16)
    return rows, cols


def _smoothstep_ramp(n, lo, hi):
    """1.0 left of lo, 0.0 right of hi, smooth ramp between (indices 0..n-1)."""
    xs = torch.arange(n, dtype=torch.float32)
    if hi <= lo:
        return (xs < lo).float()
    t = ((xs - lo) / (hi - lo)).clamp(0.0, 1.0)
    s = t * t * (3 - 2 * t)          # smoothstep
    return 1.0 - s


def _build_masks(split_mode, w, h, feather, blend, bboxes):
    rows, cols = _build_token_grid(w, h)
    n = rows * cols

    if split_mode == "horizontal_auto":
        ramp_rows = _smoothstep_ramp(
            rows, rows / 2 - feather * rows, rows / 2 + feather * rows)
        a = ramp_rows.unsqueeze(1).expand(rows, cols).reshape(-1)
    elif split_mode == "bbox" and bboxes:
        a = _mask_from_bbox(bboxes, 0, rows, cols, w, h, feather)
        b = _mask_from_bbox(bboxes, 1, rows, cols, w, h, feather)
        return _apply_blend(a, b, blend)
    else:  # vertical_auto (default)
        ramp_cols = _smoothstep_ramp(
            cols, cols / 2 - feather * cols, cols / 2 + feather * cols)
        a = ramp_cols.unsqueeze(0).expand(rows, cols).reshape(-1)

    b = 1.0 - a
    return _apply_blend(a, b, blend)


def _mask_from_bbox(bboxes, idx, rows, cols, w, h, feather):
    n = rows * cols
    if idx >= len(bboxes):
        return torch.zeros(n)
    x0, y0, x1, y1 = _coerce_bbox(bboxes[idx], w, h)
    # to token-grid coords
    c0, c1 = x0 / w * cols, x1 / w * cols
    r0, r1 = y0 / h * rows, y1 / h * rows
    fc = max(1e-3, feather * cols)
    fr = max(1e-3, feather * rows)
    cc = torch.arange(cols).float().unsqueeze(0)       # [1,cols]
    rr = torch.arange(rows).float().unsqueeze(1)       # [rows,1]
    in_x = (torch.sigmoid((cc - c0) / fc) * torch.sigmoid((c1 - cc) / fc))
    in_y = (torch.sigmoid((rr - r0) / fr) * torch.sigmoid((r1 - rr) / fr))
    m = (in_y * in_x).reshape(-1)
    peak = m.max()
    if peak > 0:                 # sigmoid product tops out below 1 for small boxes
        m = m / peak             # -> full LoRA strength inside the box
    return m.clamp(0.0, 1.0)


def _coerce_bbox(box, w, h):
    vals = list(box) if not isinstance(box, dict) else [
        box.get("x", box.get("x0", 0)), box.get("y", box.get("y0", 0)),
        box.get("x1", box.get("x", 0) + box.get("w", box.get("width", 0))),
        box.get("y1", box.get("y", 0) + box.get("h", box.get("height", 0)))]
    x0, y0, x1, y1 = [float(v) for v in vals[:4]]
    if max(x0, y0, x1, y1) <= 1.0:          # normalised 0..1 -> pixels
        x0, x1 = x0 * w, x1 * w
        y0, y1 = y0 * h, y1 * h
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _apply_blend(a, b, blend):
    # blend 0 -> pure regional ; blend 1 -> both at 0.5 everywhere (controlled merge)
    a = (1.0 - blend) * a + blend * 0.5
    b = (1.0 - blend) * b + blend * 0.5
    return a, b


# ----------------------------------------------------------------------------
# the forward-time delta injection  (optimized: prepare-once + bf16)
# ----------------------------------------------------------------------------
_COMPUTE_DTYPE = torch.bfloat16


def _stack_delta(xf, items):
    """Sum the (x @ down.T) @ up.T delta of every stacked LoRA at this layer/side."""
    total = (xf @ items[0]["down_d"].t()) @ items[0]["up_d"].t()
    for d in items[1:]:
        total = total + (xf @ d["down_d"].t()) @ d["up_d"].t()
    return total


def _make_hook(session, entry):
    """Forward hook: out += mask_a*delta_a + mask_b*delta_b on the image-token tail.
    delta_a/delta_b are each the SUM of every LoRA stacked on that side, masked once.
    All heavy tensors (LoRA up/down matrices, masks) are pre-moved to device + bf16
    in session._prepare_weights()/_prepare_grid(), so the hook only does matmuls —
    no per-call .to()/cast, and `out` is never up-cast to fp32."""
    a_list = entry.get("a") or ()
    b_list = entry.get("b") or ()

    def hook(module, inp, out):
        if not torch.is_tensor(out) or out.dim() < 2:
            return out
        x = inp[0]
        if not torch.is_tensor(x) or x.dim() < 2:
            return out
        seq = x.shape[-2]
        xf = x.to(_COMPUTE_DTYPE)
        res = None
        if a_list:
            res = session._full_mask("a", seq, out.dim()) * _stack_delta(xf, a_list)
        if b_list:
            mb = session._full_mask("b", seq, out.dim()) * _stack_delta(xf, b_list)
            res = mb if res is None else res + mb
        if res is None:
            return out
        return out + res.to(out.dtype)

    return hook


def _resolve_auto_split(rows, cols):
    """landscape -> vertical (L/R, side-by-side across the width);
    portrait/square -> horizontal (T/B, stacked down the height)."""
    return "vertical_auto" if cols > rows else "horizontal_auto"


def _masks_from_grid(split_mode, rows, cols, feather, blend):
    if split_mode == "horizontal_auto":
        ramp = _smoothstep_ramp(rows, rows / 2 - feather * rows, rows / 2 + feather * rows)
        a = ramp.unsqueeze(1).expand(rows, cols).reshape(-1)
    else:  # vertical_auto
        ramp = _smoothstep_ramp(cols, cols / 2 - feather * cols, cols / 2 + feather * cols)
        a = ramp.unsqueeze(0).expand(rows, cols).reshape(-1)
    b = 1.0 - a
    return _apply_blend(a, b, blend)


def _parse_bbox_str(s, w, h):
    """'x0,y0,x1,y1' -> normalized (0..1) tuple. Accepts normalized or pixel coords
    (auto-detected: any value >1 => pixels, divided by canvas w/h). None if empty/bad."""
    if not s or not str(s).strip():
        return None
    try:
        parts = [float(v) for v in str(s).replace(";", ",").split(",") if v.strip() != ""]
    except Exception:
        return None
    if len(parts) < 4:
        return None
    x0, y0, x1, y1 = parts[:4]
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.0:
        x0, x1 = x0 / max(1, w), x1 / max(1, w)
        y0, y1 = y0 / max(1, h), y1 / max(1, h)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1))


def _rect_token_mask(rows, cols, nx0, ny0, nx1, ny1, feather):
    """Soft-edged rectangle (normalized coords) rendered onto the rows x cols grid."""
    c0, c1 = nx0 * cols, nx1 * cols
    r0, r1 = ny0 * rows, ny1 * rows
    fc = max(1e-3, feather * cols)
    fr = max(1e-3, feather * rows)
    cc = torch.arange(cols, dtype=torch.float32).unsqueeze(0)
    rr = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
    in_x = torch.sigmoid((cc - c0) / fc) * torch.sigmoid((c1 - cc) / fc)
    in_y = torch.sigmoid((rr - r0) / fr) * torch.sigmoid((r1 - rr) / fr)
    m = (in_y * in_x).reshape(-1)
    peak = m.max()
    if peak > 0:                 # sigmoid product tops out below 1 for small boxes
        m = m / peak             # -> full LoRA strength inside the box
    return m.clamp(0.0, 1.0)


def _masks_from_regions(regions, rows, cols, feather, blend):
    """Parse the JS editor's regions JSON: [{x,y,w,h,char:'a'|'b'}, ...] in
    normalized 0-1 coords. Union of each character's rects -> two token masks."""
    try:
        items = json.loads(regions) if isinstance(regions, str) else regions
    except Exception:
        return None
    if not isinstance(items, list) or not items:
        return None
    n = rows * cols
    ma = torch.zeros(n)
    mb = torch.zeros(n)
    na = nb = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            x = float(it["x"]); y = float(it["y"])
            w = float(it.get("w", it.get("width", 0)))
            h = float(it.get("h", it.get("height", 0)))
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        ch = str(it.get("char", "a")).lower()
        m = _rect_token_mask(rows, cols, x, y, x + w, y + h, feather)
        if ch == "b":
            mb = torch.maximum(mb, m); nb += 1
        else:
            ma = torch.maximum(ma, m); na += 1
    if na == 0 and nb == 0:
        return None
    if na == 0:
        ma = 1.0 - mb
    if nb == 0:
        mb = 1.0 - ma
    return _apply_blend(ma, mb, blend)


def _flatten_bboxes(bboxes):
    """KJ/BoundingBox output is per-frame nested list[list[dict]]; unwrap to a flat
    list of box dicts (frame 0). Accepts an already-flat list too."""
    if not bboxes:
        return []
    try:
        first = bboxes[0]
    except Exception:
        return []
    if isinstance(first, (list, tuple)):
        return list(first)
    return list(bboxes)


def _masks_from_boxlist(boxes, rows, cols, w, h, feather, blend):
    """box[0] -> A, box[1] -> B (draw order). One box -> B is the complement of A."""
    ma = _mask_from_bbox(boxes, 0, rows, cols, w, h, feather)
    mb = _mask_from_bbox(boxes, 1, rows, cols, w, h, feather) if len(boxes) > 1 else (1.0 - ma)
    return _apply_blend(ma, mb, blend)


class _RegionalSession:
    """Holds per-apply config; builds masks at RUNTIME from the real latent grid
    (no reliance on typed canvas dims), then installs/removes hooks each forward.
    lora_a_stack/lora_b_stack: list of (matrices_dict, strength) per side — one
    entry per stacked LoRA. A single-LoRA node just passes a 1-item list."""
    def __init__(self, patcher, lora_a_stack, lora_b_stack,
                 split_mode, seam_feather, blend_override, bboxes,
                 mask_a_in, mask_b_in, regions_str="", text_strength=1.0):
        self.patcher = patcher
        self.lora_a_stack, self.lora_b_stack = lora_a_stack, lora_b_stack
        self.split_mode = split_mode
        self.seam_feather = seam_feather
        self.blend_override = blend_override
        self.text_strength = text_strength
        self.bboxes = bboxes
        self.mask_a_in, self.mask_b_in = mask_a_in, mask_b_in
        self.regions_str = regions_str
        self.n_img = 0
        self.mask_a = None
        self.mask_b = None
        self._layer_map = None
        self._weights_prepared = False
        self._mask_a_d = None
        self._mask_b_d = None
        self._last_shape = "unset"
        self._full_mask_cache = {}

    def _diffusion_model(self):
        m = self.patcher.model
        return getattr(m, "diffusion_model", m)

    @staticmethod
    def _side_maps(stack):
        """[(matrices_dict, strength), ...] -> [{sig: entry_with_baked_scale}, ...]"""
        return [
            {sig: {**d, "scale": d["scale"] * strength} for sig, d in mats.items()}
            for mats, strength in stack
        ]

    def _build_layer_map(self, dm):
        amaps = self._side_maps(self.lora_a_stack)
        bmaps = self._side_maps(self.lora_b_stack)
        layer_map = {}
        matched = 0
        for name, mod in _iter_named_linears(dm):
            sig = _norm(name)
            entry = {}
            a_entries = [m[sig] for m in amaps if sig in m]
            b_entries = [m[sig] for m in bmaps if sig in m]
            if a_entries:
                entry["a"] = a_entries
            if b_entries:
                entry["b"] = b_entries
            if entry:
                layer_map[name] = (mod, entry)
                matched += 1
        a_targets = sum(len(m) for m in amaps)
        b_targets = sum(len(m) for m in bmaps)
        hit = sum(len(e.get("a") or ()) + len(e.get("b") or ())
                  for _, e in layer_map.values())
        print(f"[RegionalCharacterLora] matched {matched} layers "
              f"(A:{len(amaps)} lora(s)/{a_targets} targets, "
              f"B:{len(bmaps)} lora(s)/{b_targets} targets).")
        if 0 < hit < a_targets + b_targets:
            print(f"[RegionalCharacterLora] !! {a_targets + b_targets - hit} LoRA "
                  f"target(s) matched no model layer - those weights are inert. "
                  f"Run recon_krea2.py to compare key stems.")
        if matched == 0:
            print("[RegionalCharacterLora] !! 0 layers matched - run recon_krea2.py "
                  "and compare LoRA stems vs UNet module names.")
        return layer_map

    def _infer_device(self, dm, args):
        x0 = args[0] if args else None
        if torch.is_tensor(x0):
            return x0.device
        try:
            return next(dm.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _resolve_grid(self, x):
        """Token grid (rows, cols) from the runtime latent [B,C,(T,)H,W]; falls back
        to typed canvas dims only if the latent is unreadable. Krea2 patch size = 2."""
        if torch.is_tensor(x) and x.dim() >= 4:
            H, W = int(x.shape[-2]), int(x.shape[-1])
            rows, cols = H // 2, W // 2
            if rows > 0 and cols > 0:
                return rows, cols, "latent"
        rows, cols = _build_token_grid(1024, 1536)   # default if latent unreadable
        return rows, cols, "canvas-fallback"

    def _build_masks_now(self, rows, cols):
        feather, blend = self.seam_feather, self.blend_override
        pw, ph = cols * 16, rows * 16   # pixel reference from the real latent grid

        # painted MASK sockets are an always-on advanced override
        if self.mask_a_in is not None or self.mask_b_in is not None:
            ma = _mask_to_token_grid(self.mask_a_in, rows, cols) if self.mask_a_in is not None else None
            mb = _mask_to_token_grid(self.mask_b_in, rows, cols) if self.mask_b_in is not None else None
            if ma is None:
                ma = 1.0 - mb
            if mb is None:
                mb = 1.0 - ma
            a, b = _apply_blend(ma, mb, blend)
            return a, b, "mask-socket"

        mode = self.split_mode

        # manual = the on-node visual editor (regions JSON)
        if mode == "manual":
            res = _masks_from_regions(self.regions_str, rows, cols, feather, blend)
            if res is not None:
                a, b = res
                return a, b, "manual"
            mode = "auto"   # nothing drawn yet -> sensible fallback

        # bbox = KJ/BoundingBox wire (visual editor in KJ's node)
        if mode == "bbox":
            wire = _flatten_bboxes(self.bboxes)
            if wire:
                a, b = _masks_from_boxlist(wire, rows, cols, pw, ph, feather, blend)
                return a, b, "bbox-wire(%d)" % len(wire)
            mode = "auto"   # bbox selected but nothing connected

        # geometric auto / vertical / horizontal
        if mode == "auto":
            mode = _resolve_auto_split(rows, cols)
        a, b = _masks_from_grid(mode, rows, cols, feather, blend)
        return a, b, mode

    def _prepare_weights(self, dev):
        """LoRA up/down matrices only — resolution-independent, so this runs once."""
        cdt = _COMPUTE_DTYPE
        for name, (mod, entry) in self._layer_map.items():
            for side in ("a", "b"):
                for d in entry.get(side) or ():
                    if "down_d" in d:
                        continue
                    d["down_d"] = d["down"].to(dev, cdt)
                    d["up_d"] = d["up"].to(dev, cdt) * d["scale"]

    def _prepare_grid(self, dev, x):
        """Rebuild the token grid + masks from the CURRENT latent. This is cheap
        (a few thousand-element tensor op), so instead of caching it once we just
        skip the rebuild when the latent shape hasn't changed since last call —
        keeps auto/vertical/horizontal splits correct across resolution changes
        (e.g. a hires-fix 2nd pass reusing this same patched model) with no need
        to re-trigger the node. Returns True if the grid was (re)built."""
        shape_key = tuple(x.shape) if torch.is_tensor(x) else None
        if shape_key == self._last_shape:
            return False
        cdt = _COMPUTE_DTYPE
        self._dev = dev
        rows, cols, src = self._resolve_grid(x)
        self.n_img = rows * cols
        a, b, used = self._build_masks_now(rows, cols)
        self.mask_a, self.mask_b = a, b
        self._mask_a_d = a.to(dev, cdt)
        self._mask_b_d = b.to(dev, cdt)
        self._full_mask_cache = {}
        self._grid_info = (rows, cols, src, used)
        self._last_shape = shape_key
        return True

    def _full_mask(self, side, seq, ndim):
        """Cached full-sequence mask: text_strength over the text prefix (the
        trigger-token pathway a global LoRA load would also transform), regional
        mask over the image tail. Layers whose sequence has no image tail
        (txtfusion / tmlp / txtmlp - text-conditioning stack) run entirely at
        text_strength; a global LoRA load runs them at 1.0."""
        key = (side, seq, ndim)
        fm = self._full_mask_cache.get(key)
        if fm is None:
            mv = self._mask_a_d if side == "a" else self._mask_b_d
            base = torch.full((seq,), self.text_strength,
                              device=self._dev, dtype=_COMPUTE_DTYPE)
            n_img = self.n_img
            if n_img <= 0 or n_img > seq:
                print(f"[RegionalCharacterLora] note: hooked layer with seq={seq} "
                      f"has no image-token tail (n_img={n_img}); side '{side}' "
                      f"applied uniformly at text_strength={self.text_strength}.")
            else:
                base[seq - n_img:] = mv
            fm = base.view(*([1] * (ndim - 2)), seq, 1)
            self._full_mask_cache[key] = fm
        return fm

    def run(self, executor, *args, **kwargs):
        dm = self._diffusion_model()
        if self._layer_map is None:
            self._layer_map = self._build_layer_map(dm)
        x0 = args[0] if args else None
        dev = self._infer_device(dm, args)
        if not self._weights_prepared:
            self._prepare_weights(dev)
            self._weights_prepared = True
        if self._prepare_grid(dev, x0):
            rows, cols, src, used = self._grid_info
            shp = tuple(x0.shape) if torch.is_tensor(x0) else None
            print(f"[RegionalCharacterLora] grid ready on {dev} | latent={shp} "
                  f"grid={rows}x{cols} ({src}) n_img={self.n_img} split={used}")
            if src == "canvas-fallback":
                print("[RegionalCharacterLora] !! WARNING: latent shape was unreadable, "
                      "fell back to a hardcoded 1024x1536 grid. Masks do NOT match your "
                      "actual resolution — this should not happen in normal use; "
                      "check what's calling the model wrapper.")
        handles = []
        try:
            for name, (mod, entry) in self._layer_map.items():
                handles.append(mod.register_forward_hook(_make_hook(self, entry)))
            return executor(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()


def _mask_to_token_grid(mask, rows, cols):
    """Resize a full-canvas MASK [.,H,W] down to the token grid (rows x cols) and
    flatten row-major to [n_image_tokens], matching image-token order."""
    import torch.nn.functional as F
    m = mask
    if m.dim() == 2:
        m = m.unsqueeze(0)
    if m.dim() == 3:
        m = m.unsqueeze(1)
    m = m.float()
    m = F.interpolate(m, size=(rows, cols), mode="bilinear", align_corners=False)
    return m[0, 0].reshape(-1).clamp(0.0, 1.0)


# ----------------------------------------------------------------------------
# the node
# ----------------------------------------------------------------------------
class Krea2RegionalCharacterLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        loras = _lora_dir_list() or ["<put .safetensors in models/loras>"]
        return {
            "required": {
                "model": ("MODEL",),
                "lora_a": (loras,),
                "strength_a": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "lora_b": (loras,),
                "strength_b": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "split_mode": (["manual", "auto", "vertical_auto", "horizontal_auto", "bbox"],),
                "seam_feather": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.3, "step": 0.01}),
                "blend_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "text_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                            "tooltip": "How strongly each LoRA also transforms the text/"
                                                       "conditioning tokens (both regions share them). "
                                                       "1.0 = like a global LoRA load; 0 = old behavior "
                                                       "(image tokens only - much weaker identity). "
                                                       "Lower it if the characters bleed into each other."}),
            },
            "optional": {
                "regions": ("STRING", {"default": "", "tooltip": "managed by the visual editor widget"}),
                "bboxes": ("BOUNDINGBOX",),
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/regional"

    def apply(self, model, lora_a, strength_a, lora_b, strength_b, split_mode,
              seam_feather, blend_override, text_strength=1.0,
              regions="", bboxes=None, mask_a=None, mask_b=None):
        la = _load_lora_matrices(_resolve_lora_path(lora_a))
        lb = _load_lora_matrices(_resolve_lora_path(lora_b))

        patched = model.clone()
        session = _RegionalSession(
            patched, [(la, strength_a)], [(lb, strength_b)],
            split_mode, seam_feather, blend_override, bboxes,
            mask_a, mask_b, regions, text_strength)

        def wrapper(executor, *args, **kwargs):
            return session.run(executor, *args, **kwargs)

        if hasattr(patched, "add_wrapper_with_key"):
            patched.add_wrapper_with_key(_WRAPPER_ENUM, WRAPPER_KEY, wrapper)
        elif hasattr(patched, "add_wrapper"):
            patched.add_wrapper(_WRAPPER_ENUM, wrapper)
        else:
            raise RuntimeError(
                "This ComfyUI build lacks model wrapper support "
                "(add_wrapper_with_key). Update ComfyUI.")
        return (patched,)


_STACK_SLOTS = 3   # LoRAs per zone; unused slots left as "None"


class Krea2RegionalCharacterLoRAStack:
    """Same regional injection as Krea2RegionalCharacterLoRA, but each zone (A/B)
    takes up to _STACK_SLOTS LoRAs instead of exactly one — e.g. a character LoRA
    + an outfit LoRA in the same region. Leave a slot's lora as "None" to skip it."""
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + (_lora_dir_list() or [])
        required = {"model": ("MODEL",)}
        for side in ("a", "b"):
            for i in range(1, _STACK_SLOTS + 1):
                required[f"lora_{side}{i}"] = (loras,)
                required[f"strength_{side}{i}"] = (
                    "FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05})
        required["split_mode"] = (["manual", "auto", "vertical_auto", "horizontal_auto", "bbox"],)
        required["seam_feather"] = ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.3, "step": 0.01})
        required["blend_override"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05})
        required["text_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                               "tooltip": "How strongly each LoRA also transforms the text/"
                                                          "conditioning tokens (both regions share them). "
                                                          "1.0 = like a global LoRA load; 0 = old behavior "
                                                          "(image tokens only - much weaker identity). "
                                                          "Lower it if the characters bleed into each other."})
        return {
            "required": required,
            "optional": {
                "regions": ("STRING", {"default": "", "tooltip": "managed by the visual editor widget"}),
                "bboxes": ("BOUNDINGBOX",),
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/regional"

    def apply(self, model, split_mode, seam_feather, blend_override, text_strength=1.0,
              regions="", bboxes=None, mask_a=None, mask_b=None, **kwargs):
        def build_stack(side):
            stack = []
            for i in range(1, _STACK_SLOTS + 1):
                mats = _load_optional_lora(kwargs.get(f"lora_{side}{i}"))
                if mats is not None:
                    stack.append((mats, kwargs.get(f"strength_{side}{i}", 1.0)))
            return stack

        a_stack, b_stack = build_stack("a"), build_stack("b")
        if not a_stack and not b_stack:
            print("[RegionalCharacterLoraStack] !! no LoRAs selected in either "
                  "zone - model passed through unchanged.")

        patched = model.clone()
        session = _RegionalSession(
            patched, a_stack, b_stack,
            split_mode, seam_feather, blend_override, bboxes,
            mask_a, mask_b, regions, text_strength)

        def wrapper(executor, *args, **wkwargs):
            return session.run(executor, *args, **wkwargs)

        if hasattr(patched, "add_wrapper_with_key"):
            patched.add_wrapper_with_key(_WRAPPER_ENUM, WRAPPER_KEY + "_stack", wrapper)
        elif hasattr(patched, "add_wrapper"):
            patched.add_wrapper(_WRAPPER_ENUM, wrapper)
        else:
            raise RuntimeError(
                "This ComfyUI build lacks model wrapper support "
                "(add_wrapper_with_key). Update ComfyUI.")
        return (patched,)


WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {
    "Krea2RegionalCharacterLoRA": Krea2RegionalCharacterLoRA,
    "RegionalCharacterLora": Krea2RegionalCharacterLoRA,   # legacy id, keeps old graphs loading
    "Krea2RegionalCharacterLoRAStack": Krea2RegionalCharacterLoRAStack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionalCharacterLoRA": "Krea2 Regional Character LoRA",
    "RegionalCharacterLora": "Krea2 Regional Character LoRA (legacy id)",
    "Krea2RegionalCharacterLoRAStack": "Krea2 Regional Character LoRA (Stack)",
}
