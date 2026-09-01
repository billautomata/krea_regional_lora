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
__version__ = "1.2.0"   # keep in sync with pyproject.toml


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


def _apply_blend(masks, blend):
    """blend 0 -> pure regional ; blend 1 -> every zone at 1/N everywhere
    (controlled merge). `masks` is the per-zone list, in zone order."""
    even = 1.0 / max(1, len(masks))
    return [(1.0 - blend) * m + blend * even for m in masks]


def _complement_the_one_gap(masks, missing):
    """Convenience shared by every mask source: if exactly ONE zone was left
    undefined, it gets whatever the other zones don't cover. Two or more gaps is
    ambiguous, so those zones just get an empty mask (that LoRA stays inert)."""
    if len(missing) != 1:
        return
    i = missing[0]
    union = None
    for j, m in enumerate(masks):
        if j == i:
            continue
        union = m if union is None else torch.maximum(union, m)
    if union is not None:
        masks[i] = (1.0 - union).clamp(0.0, 1.0)


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


def _make_hook(session, entry, uniform=False):
    """Forward hook: out += sum over zones of mask_z * delta_z on the image-token
    tail. delta_z is the SUM of every LoRA stacked on that zone, masked once.
    All heavy tensors (LoRA up/down matrices, masks) are pre-moved to device + bf16
    in session._prepare_weights()/_prepare_grid(), so the hook only does matmuls —
    no per-call .to()/cast, and `out` is never up-cast to fp32.
    uniform=True is the isolation path: the zone's LoRA runs globally (like a
    plain LoRA load) because containment happens later, in latent space."""
    sides = [(z, items) for z, items in entry.items() if items]

    def hook(module, inp, out):
        if not torch.is_tensor(out) or out.dim() < 2:
            return out
        x = inp[0]
        if not torch.is_tensor(x) or x.dim() < 2:
            return out
        seq = x.shape[-2]
        xf = x.to(_COMPUTE_DTYPE)
        res = None
        for z, items in sides:
            m = (session._uniform_mask(seq, out.dim()) if uniform
                 else session._full_mask(z, seq, out.dim()))
            d = m * _stack_delta(xf, items)
            res = d if res is None else res + d
        if res is None:
            return out
        return out + res.to(out.dtype)

    return hook


def _upsample_weight(mask_flat, rows, cols, spatial):
    """Token-grid mask -> latent-resolution weight [1,1,H,W]; broadcasts over
    [B,C,H,W] and [B,C,T,H,W] model outputs."""
    import torch.nn.functional as F
    m = mask_flat.view(1, 1, rows, cols).float()
    return F.interpolate(m, size=spatial, mode="bilinear", align_corners=False)


def _blend_isolated(zone_outs, weights, clean):
    """Composite per-zone full-strength model outputs in latent space. With full
    coverage (weights sum to ~1 everywhere) it's a normalized weighted sum; with
    gaps or overlaps `clean` (a no-LoRA pass) is the baseline and each zone
    contributes weight * (its output - clean)."""
    if clean is not None:
        res = clean
        for z, o in zone_outs.items():
            res = res + weights[z].to(o.dtype) * (o - clean)
        return res
    res, total = None, None
    for z, o in zone_outs.items():
        w = weights[z].to(o.dtype)
        res = w * o if res is None else res + w * o
        total = w if total is None else total + w
    return res / total.clamp(min=1e-3)


def _resolve_auto_split(rows, cols):
    """landscape -> vertical (L/R, side-by-side across the width);
    portrait/square -> horizontal (T/B, stacked down the height)."""
    return "vertical_auto" if cols > rows else "horizontal_auto"


def _masks_from_grid(split_mode, rows, cols, feather, blend, n_zones=2):
    """Cut the token grid into n_zones equal contiguous slices along one axis:
    top -> bottom for horizontal_auto, left -> right for vertical_auto, so zone
    order is reading order (A top/left, then B, then C).

    Each slice is the plateau between its two smoothstep boundaries, so at N=2
    this is identical to the old ramp/1-ramp pair and the slices sum to 1.0.
    seam_feather stays in its existing units (a fraction of the whole axis, not
    of a slice) — so at 3 zones a large feather makes neighbouring seams overlap
    and the slices sum to slightly under 1.0 near the seams."""
    axis_n = rows if split_mode == "horizontal_auto" else cols
    step = axis_n / float(n_zones)
    # edge[i] = 1.0 left of boundary i, 0.0 right of it. edge[0] is the grid's
    # left/top wall (nothing before it), edge[n] the far wall (everything).
    edge = [torch.zeros(axis_n)]
    for i in range(1, n_zones):
        edge.append(_smoothstep_ramp(
            axis_n, i * step - feather * axis_n, i * step + feather * axis_n))
    edge.append(torch.ones(axis_n))

    masks = []
    for i in range(n_zones):
        band = (edge[i + 1] - edge[i]).clamp(0.0, 1.0)
        if split_mode == "horizontal_auto":
            masks.append(band.unsqueeze(1).expand(rows, cols).reshape(-1))
        else:
            masks.append(band.unsqueeze(0).expand(rows, cols).reshape(-1))
    return _apply_blend(masks, blend)


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


def _masks_from_regions(regions, rows, cols, feather, blend, zones=("a", "b")):
    """Parse the JS editor's regions JSON: [{x,y,w,h,char:'a'|'b'|'c'}, ...] in
    normalized 0-1 coords. Union of each character's rects -> one token mask per
    zone. A rect whose char isn't a known zone falls back to the first zone,
    matching the old 'anything that isn't b is a' behaviour."""
    try:
        items = json.loads(regions) if isinstance(regions, str) else regions
    except Exception:
        return None
    if not isinstance(items, list) or not items:
        return None
    n = rows * cols
    masks = [torch.zeros(n) for _ in zones]
    counts = [0] * len(zones)
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
        i = zones.index(ch) if ch in zones else 0
        masks[i] = torch.maximum(masks[i], _rect_token_mask(
            rows, cols, x, y, x + w, y + h, feather))
        counts[i] += 1
    if not any(counts):
        return None
    _complement_the_one_gap(masks, [i for i, c in enumerate(counts) if c == 0])
    return _apply_blend(masks, blend)


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


def _masks_from_boxlist(boxes, rows, cols, w, h, feather, blend, n_zones=2):
    """box[i] -> zone i, in draw order. One box short of the zone count -> the
    last zone is the complement of the others (2-zone: one box -> B = not-A)."""
    masks = [_mask_from_bbox(boxes, i, rows, cols, w, h, feather)
             for i in range(n_zones)]
    _complement_the_one_gap(masks, list(range(len(boxes), n_zones)))
    return _apply_blend(masks, blend)


def _bake_stacks(stacks):
    """{zone: [(matrices, strength)]} -> {zone: [{sig: entry_with_baked_scale}]}.

    Called ONCE per node execution and handed to every session, so `model` and
    `model_b` share the same entry dicts — which means they also share the device
    tensors `_prepare_weights` hangs off them, and the LoRA weights are moved to
    VRAM once rather than per model."""
    return {
        z: [{sig: {**d, "scale": d["scale"] * strength} for sig, d in mats.items()}
            for mats, strength in stack]
        for z, stack in stacks.items()
    }


class _RegionalSession:
    """Holds per-apply config; builds masks at RUNTIME from the real latent grid
    (no reliance on typed canvas dims), then installs/removes hooks each forward.
    zone_maps: the output of _bake_stacks — {zone: [{sig: entry}, ...]}, one dict
    per LoRA stacked on that zone. Zone order is the dict's insertion order
    ('a','b'[,'c']) and defines the geometric slice order. Pass the SAME object to
    every session that shares one node's settings (model and model_b).
    mask_ins: {zone: MASK or None} for the painted-mask override sockets."""
    def __init__(self, patcher, zone_maps,
                 split_mode, seam_feather, blend_override, bboxes,
                 mask_ins, regions_str="", text_strength=1.0, isolate=False):
        self.isolate = isolate
        self.patcher = patcher
        self.zone_maps = zone_maps
        self.zones = tuple(zone_maps)
        self.split_mode = split_mode
        self.seam_feather = seam_feather
        self.blend_override = blend_override
        self.text_strength = text_strength
        self.bboxes = bboxes
        self.mask_ins = {z: mask_ins.get(z) for z in self.zones}
        self.regions_str = regions_str
        self.n_img = 0
        self.masks = {}
        self._layer_map = None
        self._weights_prepared = None   # the device the matrices were moved to
        self._masks_d = {}
        self._last_shape = "unset"
        self._full_mask_cache = {}

    def _diffusion_model(self):
        m = self.patcher.model
        return getattr(m, "diffusion_model", m)

    def _build_layer_map(self, dm):
        zmaps = self.zone_maps
        layer_map = {}
        matched = 0
        for name, mod in _iter_named_linears(dm):
            sig = _norm(name)
            entry = {}
            for z in self.zones:
                hits = [m[sig] for m in zmaps[z] if sig in m]
                if hits:
                    entry[z] = hits
            if entry:
                layer_map[name] = (mod, entry)
                matched += 1
        targets = {z: sum(len(m) for m in zmaps[z]) for z in self.zones}
        total_targets = sum(targets.values())
        hit = sum(len(v) for _, e in layer_map.values() for v in e.values())
        print(f"[RegionalCharacterLora] matched {matched} layers ("
              + ", ".join(f"{z.upper()}:{len(zmaps[z])} lora(s)/{targets[z]} targets"
                          for z in self.zones) + ").")
        if 0 < hit < total_targets:
            print(f"[RegionalCharacterLora] !! {total_targets - hit} LoRA "
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
        nz = len(self.zones)
        pw, ph = cols * 16, rows * 16   # pixel reference from the real latent grid

        # painted MASK sockets are an always-on advanced override
        if any(m is not None for m in self.mask_ins.values()):
            masks = [_mask_to_token_grid(self.mask_ins[z], rows, cols)
                     if self.mask_ins[z] is not None else torch.zeros(rows * cols)
                     for z in self.zones]
            _complement_the_one_gap(
                masks, [i for i, z in enumerate(self.zones)
                        if self.mask_ins[z] is None])
            return _apply_blend(masks, blend), "mask-socket"

        mode = self.split_mode

        # manual = the on-node visual editor (regions JSON)
        if mode == "manual":
            res = _masks_from_regions(self.regions_str, rows, cols, feather,
                                      blend, self.zones)
            if res is not None:
                return res, "manual"
            mode = "auto"   # nothing drawn yet -> sensible fallback

        # bbox = KJ/BoundingBox wire (visual editor in KJ's node)
        if mode == "bbox":
            wire = _flatten_bboxes(self.bboxes)
            if wire:
                return (_masks_from_boxlist(wire, rows, cols, pw, ph, feather,
                                            blend, nz),
                        "bbox-wire(%d)" % len(wire))
            mode = "auto"   # bbox selected but nothing connected

        # geometric auto / vertical / horizontal
        if mode == "auto":
            mode = _resolve_auto_split(rows, cols)
        return _masks_from_grid(mode, rows, cols, feather, blend, nz), mode

    def _prepare_weights(self, dev):
        """LoRA up/down matrices only — resolution-independent, so this runs once."""
        cdt = _COMPUTE_DTYPE
        for name, (mod, entry) in self._layer_map.items():
            for items in entry.values():
                for d in items:
                    # Two sessions (model + model_b) share these dicts, so the
                    # second one reuses the first's device tensors — VRAM is paid
                    # once. Keyed on device so a model on a *different* device
                    # re-moves them instead of silently using the wrong ones.
                    if d.get("_dev") == dev:
                        continue
                    d["down_d"] = d["down"].to(dev, cdt)
                    d["up_d"] = d["up"].to(dev, cdt) * d["scale"]
                    d["_dev"] = dev

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
        masks, used = self._build_masks_now(rows, cols)
        self.masks = dict(zip(self.zones, masks))
        self._masks_d = {z: m.to(dev, cdt) for z, m in self.masks.items()}
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
            mv = self._masks_d[side]
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

    def _uniform_mask(self, seq, ndim):
        """Isolation-pass mask: image tail at 1.0 everywhere, text prefix at
        text_strength (1.0 = exactly a global LoRA load). Spatial containment
        happens after the pass, in latent space, not here."""
        key = ("*", seq, ndim)
        fm = self._full_mask_cache.get(key)
        if fm is None:
            base = torch.full((seq,), self.text_strength,
                              device=self._dev, dtype=_COMPUTE_DTYPE)
            n_img = self.n_img
            if 0 < n_img <= seq:
                base[seq - n_img:] = 1.0
            fm = base.view(*([1] * (ndim - 2)), seq, 1)
            self._full_mask_cache[key] = fm
        return fm

    def _run_isolated(self, executor, args, kwargs):
        """Two-memory-slots mode: one full forward per zone with only that zone's
        LoRAs hooked, applied GLOBALLY (its own private text tokens included),
        then the outputs are composited with the zone masks in latent space —
        where every value has coordinates, so nothing can leak through the shared
        prompt. Costs one extra model pass per active zone (+1 clean pass when
        the masks don't cover the canvas).
        # ponytail: zones run sequentially; batch them into one call if step time matters
        """
        x0 = args[0] if args else None
        rows, cols = self._grid_info[0], self._grid_info[1]
        spatial = tuple(x0.shape[-2:])
        active, weights = [], {}
        for z in self.zones:
            if float(self.masks[z].max()) <= 0:
                continue
            layers = [(mod, {z: entry[z]})
                      for mod, entry in self._layer_map.values() if z in entry]
            if not layers:
                continue
            active.append((z, layers))
            weights[z] = _upsample_weight(self.masks[z], rows, cols,
                                          spatial).to(x0.device)
        if not active:
            return executor(*args, **kwargs)
        outs = {}
        for z, layers in active:
            handles = [mod.register_forward_hook(_make_hook(self, e, uniform=True))
                       for mod, e in layers]
            try:
                outs[z] = executor(*args, **kwargs)
            finally:
                for h in handles:
                    h.remove()
        total = weights[active[0][0]]
        for z, _ in active[1:]:
            total = total + weights[z]
        clean = executor(*args, **kwargs) if (total - 1.0).abs().max() > 1e-3 else None
        return _blend_isolated(outs, weights, clean)

    def run(self, executor, *args, **kwargs):
        dm = self._diffusion_model()
        if self._layer_map is None:
            self._layer_map = self._build_layer_map(dm)
        x0 = args[0] if args else None
        dev = self._infer_device(dm, args)
        if self._weights_prepared != dev:
            self._prepare_weights(dev)
            self._weights_prepared = dev
        if self._prepare_grid(dev, x0):
            rows, cols, src, used = self._grid_info
            shp = tuple(x0.shape) if torch.is_tensor(x0) else None
            print(f"[RegionalCharacterLora] grid ready on {dev} | latent={shp} "
                  f"grid={rows}x{cols} ({src}) n_img={self.n_img} split={used} "
                  f"isolate={self.isolate}")
            if src == "canvas-fallback":
                print("[RegionalCharacterLora] !! WARNING: latent shape was unreadable, "
                      "fell back to a hardcoded 1024x1536 grid. Masks do NOT match your "
                      "actual resolution — this should not happen in normal use; "
                      "check what's calling the model wrapper.")
        if self.isolate:
            return self._run_isolated(executor, args, kwargs)
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
# the nodes
# ----------------------------------------------------------------------------
_SPLIT_MODES = ["manual", "auto", "vertical_auto", "horizontal_auto", "bbox"]
_TEXT_STRENGTH_TIP = (
    "How strongly each LoRA also transforms the text/conditioning tokens (every "
    "zone shares them). 1.0 = like a global LoRA load; 0 = image tokens only, "
    "much weaker identity. MORE ZONES WANT MORE, not less: each seam feathers "
    "away part of a zone's full-strength area (2 zones keep ~43% of the image at "
    "full strength each, 3 zones only ~26/18/25%), so each identity leans harder "
    "on the shared text pathway. Measured: 1.0 is right for 2 zones, ~2.0 for 3. "
    "Lower it only if characters actually bleed into each other.")
_ISOLATE_TIP = (
    "Hard containment. ON: the model runs once per zone with that zone's LoRA "
    "applied globally (its own private copy of the text tokens included), and the "
    "outputs are composited with the zone masks in latent space — nothing can "
    "leak through the shared prompt, even for LoRAs that live entirely in the "
    "text-conditioning layers. Costs one extra full model pass per zone (2 zones "
    "~2x step time; +1 clean pass if your boxes don't cover the canvas). "
    "OFF: the old single-pass activation blend, where text_strength is the "
    "leak-vs-likeness tradeoff.")
_SHOW_REF_TIP = (
    "Draw the last image your graph produced as the backdrop of the region editor, "
    "so you can drag the boxes onto where the characters actually landed. NOTHING "
    "NEEDS WIRING - the editor picks up your SaveImage/PreviewImage output after "
    "the run, in the browser. There is deliberately no image input: that image is "
    "made downstream of the model this node outputs, so a link back in would be a "
    "dependency cycle and ComfyUI would reject the graph.")
_MODEL_B_TIP = (
    "Optional second model for a two-pass workflow (e.g. Krea2Raw then Turbo). "
    "It gets the SAME zones, boxes, LoRAs and strengths as the first model - one "
    "node driving both passes. For different settings per pass, use two nodes.")
_STACK_SLOTS = 3   # LoRAs per zone on the Stack nodes; unused slots left as "None"


def _common_widgets(required):
    """The config widgets every variant shares, appended after the LoRA slots."""
    required["split_mode"] = (_SPLIT_MODES,)
    required["seam_feather"] = ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.3, "step": 0.01})
    required["blend_override"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05})
    required["text_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0,
                                           "step": 0.05, "tooltip": _TEXT_STRENGTH_TIP})
    required["isolate_zones"] = ("BOOLEAN", {"default": True, "tooltip": _ISOLATE_TIP})
    required["show_reference"] = ("BOOLEAN", {"default": False, "tooltip": _SHOW_REF_TIP})
    return required


def _optional_widgets(zones):
    opt = {
        "model_b": ("MODEL", {"tooltip": _MODEL_B_TIP}),
        "regions": ("STRING", {"default": "", "tooltip": "managed by the visual editor widget"}),
        "bboxes": ("BOUNDINGBOX",),
    }
    for z in zones:
        opt[f"mask_{z}"] = ("MASK",)
    return opt


def _install(patched, session, key):
    def wrapper(executor, *args, **kwargs):
        return session.run(executor, *args, **kwargs)

    if hasattr(patched, "add_wrapper_with_key"):
        patched.add_wrapper_with_key(_WRAPPER_ENUM, key, wrapper)
    elif hasattr(patched, "add_wrapper"):
        patched.add_wrapper(_WRAPPER_ENUM, wrapper)
    else:
        raise RuntimeError(
            "This ComfyUI build lacks model wrapper support "
            "(add_wrapper_with_key). Update ComfyUI.")
    return patched


class Krea2RegionalCharacterLoRA:
    """One LoRA per zone. ZONES is both the widget set (lora_a, lora_b, ...) and
    the geometric order of the auto-split slices: A is the left/top slice, then B,
    then C.

    Two MODEL inputs / outputs: wire a second model (e.g. a Turbo pass after a
    Krea2Raw pass) and it gets the identical zones, boxes, LoRAs and strengths -
    one node, one set of controls, both passes. Each model gets its own session,
    because a session holds direct references to one model's nn.Linear modules;
    the loaded LoRA matrices themselves are shared, so VRAM is paid once."""
    ZONES = ("a", "b")
    WRAPPER_SUFFIX = ""

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + (_lora_dir_list() or [])
        required = {"model": ("MODEL",)}
        for z in cls.ZONES:
            required[f"lora_{z}"] = (loras,)
            required[f"strength_{z}"] = (
                "FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05})
        return {"required": _common_widgets(required),
                "optional": _optional_widgets(cls.ZONES)}

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("model", "model_b")
    FUNCTION = "apply"
    CATEGORY = "conditioning/regional"

    def _build_stacks(self, kw):
        """{zone: [(matrices, strength)]} - one LoRA per zone, "None" = empty
        zone (keeps its geometric slot)."""
        stacks = {}
        for z in self.ZONES:
            mats = _load_optional_lora(kw.get(f"lora_{z}"))
            stacks[z] = [] if mats is None else [(mats, kw.get(f"strength_{z}", 1.0))]
        return stacks

    def apply(self, model, split_mode, seam_feather, blend_override,
              text_strength=1.0, isolate_zones=True, show_reference=False,
              model_b=None, regions="", bboxes=None, **kw):
        zone_maps = _bake_stacks(self._build_stacks(kw))   # read + baked once...
        mask_ins = {z: kw.get(f"mask_{z}") for z in self.ZONES}

        out = []
        for m in (model, model_b):
            if m is None:
                out.append(None)
                continue
            patched = m.clone()
            session = _RegionalSession(                    # ...and shared here
                patched, zone_maps, split_mode, seam_feather, blend_override,
                bboxes, mask_ins, regions, text_strength, isolate_zones)
            out.append(_install(patched, session, WRAPPER_KEY + self.WRAPPER_SUFFIX))

        # show_reference is read by the editor in the browser; nothing to do here
        return tuple(out)


class Krea2RegionalCharacterLoRA3(Krea2RegionalCharacterLoRA):
    """Three zones (A/B/C) instead of two - identical in every other respect.
    auto still uses the same landscape/portrait heuristic as the 2-zone node, so
    a portrait latent gives top / middle / bottom and a landscape one gives
    left / middle / right."""
    ZONES = ("a", "b", "c")
    WRAPPER_SUFFIX = "_3"


class Krea2RegionalCharacterLoRAStack(Krea2RegionalCharacterLoRA):
    """Same regional injection, but each zone takes up to _STACK_SLOTS LoRAs
    instead of exactly one - e.g. a character LoRA + an outfit LoRA in the same
    region. Leave a slot's lora as "None" to skip it."""
    ZONES = ("a", "b")
    WRAPPER_SUFFIX = "_stack"

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + (_lora_dir_list() or [])
        required = {"model": ("MODEL",)}
        for z in cls.ZONES:
            for i in range(1, _STACK_SLOTS + 1):
                required[f"lora_{z}{i}"] = (loras,)
                required[f"strength_{z}{i}"] = (
                    "FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05})
        return {"required": _common_widgets(required),
                "optional": _optional_widgets(cls.ZONES)}

    def _build_stacks(self, kw):
        stacks = {}
        for z in self.ZONES:
            stack = []
            for i in range(1, _STACK_SLOTS + 1):
                mats = _load_optional_lora(kw.get(f"lora_{z}{i}"))
                if mats is not None:
                    stack.append((mats, kw.get(f"strength_{z}{i}", 1.0)))
            stacks[z] = stack   # an empty zone keeps its slot, so the other
                                # zones' slices don't shift
        if not any(stacks.values()):
            print("[RegionalCharacterLoraStack] !! no LoRAs selected in any "
                  "zone - model passed through unchanged.")
        return stacks


class Krea2RegionalCharacterLoRAStack3(Krea2RegionalCharacterLoRAStack):
    """Three zones, up to _STACK_SLOTS LoRAs each: a1/a2/a3, b1/b2/b3, c1/c2/c3."""
    ZONES = ("a", "b", "c")
    WRAPPER_SUFFIX = "_stack3"


WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {
    "Krea2RegionalCharacterLoRA": Krea2RegionalCharacterLoRA,
    "Krea2RegionalCharacterLoRA3": Krea2RegionalCharacterLoRA3,
    "Krea2RegionalCharacterLoRAStack": Krea2RegionalCharacterLoRAStack,
    "Krea2RegionalCharacterLoRAStack3": Krea2RegionalCharacterLoRAStack3,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionalCharacterLoRA": "Krea2 Regional Character LoRA",
    "Krea2RegionalCharacterLoRA3": "Krea2 Regional Character LoRA (3 zones)",
    "Krea2RegionalCharacterLoRAStack": "Krea2 Regional Character LoRA (Stack)",
    "Krea2RegionalCharacterLoRAStack3": "Krea2 Regional Character LoRA (3 zones, Stack)",
}
