# Krea2 Regional Character LoRA  —  Beta v1

Place **two trained character LoRAs into one coherent image**, each kept in its own
region, with **no identity blend** — in a single pass, no inpainting, no compositing.
Built for the **Krea2 / Flux.2-Klein** single-stream DiT (incl. the fp8 turbo model).

![two characters, one image](docs/example.png)

## Why it's different
A normal LoRA stack *merges* each LoRA's weight delta into the model globally, so both
identities bleed into every pixel. This node never merges them: it injects each LoRA's
**activation delta** (`up @ down * scale`) only into the image tokens inside that
character's region, at forward time. The base model still runs full attention across
the whole image, so you get one natural, interacting scene — just with distinct faces.

It also runs on **fp8 Krea2**, where the native ComfyUI hook-LoRA path crashes
(`'Linear' object has no attribute 'weight_scale'`), because we touch activations, not
quantized weights.

## Requirements
- A **Krea2 / Flux.2-Klein** diffusion model (fp8 turbo is fine).
- A recent **ComfyUI** (needs model wrapper support + DOM widgets).
- Two standard character LoRAs (kohya `lora_up/down` or diffusers `lora_A/B`).

## Install
1. Copy the `regional_character_lora/` folder into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.
3. Add node: **conditioning/regional → "Krea2 Regional Character LoRA"**.

(Optional) Run `recon_krea2.py` once to confirm your LoRA keys map onto the model —
see the header of that file for usage. It's pure-stdlib and needs no GPU.

## Minimal workflow
```
UNETLoader → ModelSamplingFlux → Krea2 Regional Character LoRA → KSampler
```
- `ModelSamplingFlux` is recommended (standard Krea2 sampling shift), not strictly required.
- A global `LoraLoaderStack` (style/NSFW LoRAs) can sit before this node; keep character
  LoRAs OUT of it — they go in this node.
- Enhancer / NAG are optional extras.

## Using it
- **lora_a / lora_b** — your two character LoRAs (A = first region, B = second).
- **strength_a / strength_b** — 1.0 is the sweet spot. Some LoRAs run "hot"; drop one to
  ~0.85 if it dominates.
- **split_mode** (the master switch):
  - `manual` *(default)* — drag the two boxes on the node's visual editor.
  - `auto` — picks vertical for portrait/square, horizontal for landscape.
  - `vertical_auto` / `horizontal_auto` — fixed left-right / top-bottom halves.
  - `bbox` *(experimental)* — feed an external `BOUNDINGBOX` wire (e.g. KJ nodes).
- **seam_feather** — softness of region edges (low sensitivity unless regions overlap).
- **blend_override** — 0 = clean split; raise toward ~0.5 to let overlapping bodies mix.
  Identities collapse past ~0.8.
- **mask_a / mask_b** *(experimental)* — feed hand-painted MASK tensors to override.

## The one rule that matters: place the box where the LoRA's features live
The box marks **where the LoRA is injected**, not just "where the character is."
Most character LoRAs are **face/portrait-trained**, so the box must cover where the
**head/face** will land — err generous, not tight. Body-trained LoRAs want the torso.
A box that misses the face gives weak identity. (This is why simple side-by-side
"just works" — each full-height column always contains a face.)

## Tips from testing
- The **prompt only needs scene / pose / framing** — the mask carries identity and
  placement. Trigger words are optional but good insurance for overlapping poses.
- **Faces must not occlude each other.** Bodies can overlap freely; keep the two faces
  in separate regions.
- Complex/tangled poses can produce anatomy errors — that's the **base model**, not this
  node. A pose ControlNet or a refiner pass helps.
- Results are **seed-dependent**; lock a good seed, then A/B your settings.

## Status — Beta v1
Works well for 2 characters across side-by-side, stacked, and overlapping comps.
Known rough edges: `bbox` wire + `mask_a/b` inputs are experimental/under-tested;
3+ characters not yet supported; expression can flatten on the turbo model at CFG 1.

Feedback welcome.
