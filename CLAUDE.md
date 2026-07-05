# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A ComfyUI custom node package for the Krea2 / Flux.2-Klein single-stream DiT. It places multiple trained character LoRAs into one generated image, each confined to its own spatial region, without merging their weights (which would blend identities together). It works around a crash in ComfyUI's native LoRA-patch path on fp8 Krea2 models (`'Linear' object has no attribute 'weight_scale'`) by injecting LoRA deltas as forward-hook activations instead of patching quantized weights.

Two nodes are exposed, both under `conditioning/regional`:
- `Krea2RegionalCharacterLoRA` — one LoRA per zone (A/B).
- `Krea2RegionalCharacterLoRAStack` — up to 3 LoRAs per zone (e.g. character + outfit LoRA stacked in the same region); unused slots are left as `"None"`.

## Commands

There is no build, lint, or automated test setup in this repo — it's a pure-Python ComfyUI plugin loaded directly by ComfyUI at startup.

- **Syntax-check after edits**: `python3 -c "import ast; ast.parse(open('__init__.py').read())"`
- **Verify a LoRA/UNet pairing before wiring the node** (pure stdlib, no GPU, no ComfyUI needed — only parses safetensors headers):
  ```
  python_embeded\python.exe ComfyUI\custom_nodes\regional_character_lora\recon_krea2.py ^
      --unet "models\unet\<file>.safetensors" --lora "models\loras\<file>.safetensors"
  ```
  Answers whether the UNet uses fused `qkv` or separate `q/k/v` projections, whether the LoRA targets attn-only or attn+MLP, and whether it's a standard LoRA (`lora_up/down` or `lora_A/B` — supported) vs LoHa/LyCORIS (`hada_w1_a` — **not** supported by this node's `up @ down` injection path).
- **Actually verifying a change**: there's no test harness; changes are verified by restarting ComfyUI, running a generation, and reading the `[RegionalCharacterLora]` console output (see below). Requires a GPU and Krea2 checkpoint — cannot be done from this environment alone.

## Architecture

Everything lives in `__init__.py` (single file by design — don't split it without reason). The pieces, in the order you'd trace a request:

**1. Mask/grid math (module-level functions).** Pure-tensor helpers with no ComfyUI dependency: `_build_token_grid`/`_resolve_grid` derive the `rows x cols` token grid from the *actual* runtime latent shape (`H//2, W//2` — Krea2 VAE is f8, patch size 2), never from a typed canvas size. `_smoothstep_ramp`, `_rect_token_mask`, `_masks_from_grid`, `_masks_from_boxlist`, `_masks_from_regions` each build a soft-edged `(mask_a, mask_b)` pair over that grid from a different source (geometric split, bbox rectangle, JSON region list). `_apply_blend` interpolates between a clean split and a 50/50 merge. `_build_masks` and `_parse_bbox_str` are unused dead code left over from an earlier version — don't build on them.

**2. `_RegionalSession`** is the runtime engine both node classes delegate to. Constructed once per `apply()` call with `lora_a_stack`/`lora_b_stack` (each a list of `(matrices_dict, strength)` — the single-LoRA node just passes 1-item lists), plus the split config. Two-phase prep, run every forward call via a registered ComfyUI model wrapper (`add_wrapper_with_key(..., WRAPPER_KEY, wrapper)`):
  - `_prepare_weights` (once): moves each matched LoRA's up/down matrices to device + bf16.
  - `_prepare_grid` (every call, but skips the rebuild if the latent shape is unchanged from last call): re-derives the token grid and masks from the *current* latent. This is what makes `auto`/`vertical_auto`/`horizontal_auto` track the real resolution even across a resolution change mid-session (e.g. a hires-fix second pass reusing the same patched model) — do not reintroduce a "prepare once and cache forever" pattern here, that was a real bug.
  - `_build_layer_map` matches each stacked LoRA's target-module names to live `nn.Linear` modules by a normalized signature (`_norm` — strips `lora_unet_`/`diffusion_model.`/etc. prefixes, collapses `.`/`_`, then applies `_SIG_RENAMES`, a diffusers→native rename table mirroring `comfy.utils.krea2_to_diffusers`, so both native-keyed and diffusers-keyed Krea2 LoRAs match). This is what makes the node "self-discovering": it never hardcodes fused-vs-separate qkv or which layers a LoRA targets. It warns when some LoRA targets matched no layer. If layers don't match, check with `recon_krea2.py`, not by hardcoding names here.
  - `_full_mask` builds the full text+image sequence mask per hooked layer, assuming image tokens are the **trailing** block (`[text | image]`) and the text-token count is `seq_len - n_image_tokens`, measured fresh from the real activation shape at each hook call — not a hardcoded constant. The text prefix gets `text_strength` (default 1.0 — both sides' deltas also transform the shared text tokens, matching what a global LoRA load does to the trigger-word pathway; 0 restores the old image-tokens-only behavior, which is much weaker for likeness LoRAs). Layers whose sequence has no image tail (the `txtfusion`/`tmlp`/`txtmlp` conditioning stack, which trained Krea2 LoRAs do target) run uniformly at `text_strength`, with a console note.

**3. The forward hook (`_make_hook`/`_stack_delta`)** — registered as a `register_forward_hook` on every matched `nn.Linear` for the duration of one `executor(...)` call, then removed. For each side (A/B), it sums the `(x @ down.T) @ up.T` delta of every LoRA stacked on that side, multiplies by that side's full-sequence mask *once*, and adds the result to the layer's real output. This is the actual "regional injection" — LoRA deltas are never merged into weights, so both identities can coexist without bleeding.

**4. Mask source precedence** (`_RegionalSession._build_masks_now`): painted `mask_a`/`mask_b` MASK sockets (if wired) always win → else `manual` uses the JS visual editor's `regions` JSON, falling back to `auto` if nothing's been drawn → else `bbox` uses an external `BOUNDINGBOX` wire (e.g. from KJ nodes), falling back to `auto` if nothing's wired → else the geometric `auto`/`vertical_auto`/`horizontal_auto` split. `auto` (`_resolve_auto_split`) picks `vertical_auto` (left/right) for landscape latents and `horizontal_auto` (top/bottom) for portrait/square, matching a side-by-side composition — `vertical_auto`/`horizontal_auto` selected explicitly bypass this heuristic entirely.

**5. `web/regional_lora.js`** — a ComfyUI DOM canvas widget (`app.registerExtension`) that replaces the auto-generated `regions` string widget with a draggable two-box editor (only meaningful when `split_mode = manual`). It writes normalized `[{char:'a'|'b', x,y,w,h}, ...]` JSON that `_masks_from_regions` consumes. Must stay registered for every node name that uses `manual` mode — check the `nodeData.name` guard at the top of `beforeRegisterNodeDef` when adding new node variants.

**6. Node classes** (`Krea2RegionalCharacterLoRA`, `Krea2RegionalCharacterLoRAStack`) are thin: `INPUT_TYPES` declares widgets, `apply()` loads LoRA files (`_load_lora_matrices`/`_load_optional_lora`), clones the incoming `MODEL`, builds a `_RegionalSession`, and registers it as a wrapper on the clone. Both `NODE_CLASS_MAPPINGS` entries for the original node (`Krea2RegionalCharacterLoRA` and the legacy id `RegionalCharacterLora`) must keep pointing at the same class — that's intentional, to keep old saved workflows loading.

## Known sharp edges (by design, not yet hardened)

- LyCORIS/LoHa/`diff`-only files still aren't supported (`_load_lora_matrices` only recognizes `lora_down/up` and `lora_A/B` keys), but loading one now prints a loud "no effect" warning instead of failing silently.
- Rect masks (`manual`/`bbox` modes) are peak-normalized to 1.0 — the raw sigmoid product tops out well below 1 for boxes smaller than ~half the canvas, which used to silently weaken the LoRA inside its own region.
- `text_strength` applies BOTH sides' deltas to the shared text tokens — that's the identity-strength/bleed tradeoff knob, not a bug. It is unrelated to the `clip` strength on stock LoRA loaders (that patches the separate text-encoder model; this stays entirely inside the diffusion model).
- Only 2 zones (A/B) are supported; a 3rd zone would require generalizing the mask-source functions and the JS editor beyond binary A/B.
