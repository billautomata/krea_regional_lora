# Krea2 Regional Character LoRA

Place **two or three trained character LoRAs into one coherent image**, each kept in its
own region, with **no identity blend** — in a single pass, no inpainting, no compositing.
Built for the **Krea2 / Flux.2-Klein** single-stream DiT (incl. the fp8 turbo model).

![two characters, one image](docs/example.png)

## Why it's different
A normal LoRA stack *merges* each LoRA's weight delta into the model globally, so both
identities bleed into every pixel. This node never merges them: it injects each LoRA's
**activation delta** (`up @ down * scale`) at forward time — into the image tokens inside
that character's region, and (at `text_strength`, default 1.0) into the shared prompt
tokens, the trigger-word pathway a global LoRA load would also transform. The base model
still runs full attention across the whole image, so you get one natural, interacting
scene — just with distinct faces.

It also runs on **fp8 Krea2**, where the native ComfyUI hook-LoRA path crashes
(`'Linear' object has no attribute 'weight_scale'`), because we touch activations, not
quantized weights.

## Requirements
- A **Krea2 / Flux.2-Klein** diffusion model (fp8 turbo is fine).
- A recent **ComfyUI** (needs model wrapper support + DOM widgets).
- Standard character LoRAs (kohya `lora_up/down` or diffusers `lora_A/B`); up to 3 per
  zone on the Stack nodes. LoHa/LyCORIS are not supported.

## Install
1. Copy the `regional_character_lora/` folder into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.
3. Add node: **conditioning/regional → "Krea2 Regional Character LoRA"**.

A Python change needs a full ComfyUI **restart** (not just a browser refresh), and a node
already on the canvas keeps the sockets it was saved with — right-click → *Fix node
(recreate)* after upgrading.

Four nodes, same regional engine and widgets throughout — pick by zone count and how
many LoRAs you stack per zone:

| node | zones | LoRAs per zone | slots |
|---|---|---|---|
| Krea2 Regional Character LoRA | 2 (A/B) | 1 | `lora_a`, `lora_b` |
| Krea2 Regional Character LoRA (3 zones) | 3 (A/B/C) | 1 | `lora_a`…`lora_c` |
| Krea2 Regional Character LoRA (Stack) | 2 (A/B) | 3 | `lora_a1..a3`, `lora_b1..b3` |
| Krea2 Regional Character LoRA (3 zones, Stack) | 3 (A/B/C) | 3 | `lora_a1..a3` … `lora_c1..c3` |

Stacking = character + outfit + style in the same region. Leave unused slots as `None`.

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

Tuning more than 2 zones: **[docs/TUNING.md](docs/TUNING.md)** — how zone count,
`seam_feather` and `text_strength` interact, with a regenerable measurement table
(`python test_masks.py --report`).

## Two-pass workflows (one node, both passes)

Every node has a second optional `model_b` input and a matching output. Wire your
Krea2Raw model into `model` and Turbo into `model_b`, and both come out patched with
the **identical** zones, boxes, LoRAs and strengths — one set of controls driving both
stages, so the boxes only get drawn once.

```
UNETLoader (Krea2Raw) ─┐                          ┌─ model   → KSampler (pass 1)
                       ├→ Krea2 Regional … ───────┤
UNETLoader (Turbo) ────┘                          └─ model_b → KSampler (pass 2)
```

Leave `model_b` unwired and the node behaves exactly as a single-model node; the second
output is `None`. If you need *different* settings per pass, use two nodes — that's the
only granularity this node deliberately doesn't offer.

## Reference image under the boxes

Turn on **show_reference**. That's it — no wiring. After a run the editor draws the
image your graph just produced as its backdrop, so you drag boxes onto where the
characters actually landed instead of guessing. The canvas takes the image's aspect
ratio, so boxes and picture share one coordinate space.

**Why there's no wire:** the image is made downstream of the model this node outputs, so
an `IMAGE` link back into this node would close a loop and ComfyUI would reject the graph
with a dependency cycle. The editor reads the output in the browser, after execution,
where no graph edge exists at all. There is deliberately no image input.

The widget resizes itself to the image's aspect ratio so the picture fills it without
stretching. Your boxes are unaffected — they stay in whole-canvas 0..1 coordinates,
exactly where you put them.

With the flag off nothing is written and nothing is fetched — the editor is byte-identical
to not having the feature.

## Using it
- **lora_a / lora_b** (+ **lora_c** on the 3-zone node) — your character LoRAs. Zone order
  is reading order: A is the left/top region, then B, then C.
- **strength_a / strength_b** — 1.0 is the sweet spot. Some LoRAs run "hot"; drop one to
  ~0.85 if it dominates.
- **split_mode** (the master switch):
  - `manual` *(default)* — drag the boxes on the node's visual editor (one per zone).
  - `auto` — picks vertical for landscape, horizontal for portrait/square.
  - `vertical_auto` / `horizontal_auto` — fixed left-to-right / top-to-bottom slices,
    one equal slice per zone (halves on the 2-zone nodes, thirds on the 3-zone one).
  - `bbox` *(experimental)* — feed an external `BOUNDINGBOX` wire (e.g. KJ nodes).
- **seam_feather** — softness of region edges. Costs each zone real area: it's a
  fraction of the whole axis and N zones have N-1 seams, so interior zones are soft on
  both sides. At the default 0.08 a 2-zone split holds ~43% of the image per zone at
  full strength, a 3-zone split only ~26%/18%/25%. See [docs/TUNING.md](docs/TUNING.md).
- **blend_override** — 0 = clean split; raise toward ~0.5 to let overlapping bodies mix.
  Identities collapse past ~0.8.
- **text_strength** — how strongly each LoRA also transforms the shared prompt/
  conditioning tokens inside the diffusion model. **1.0** *(default)* behaves like a
  global LoRA load on that pathway — expect much stronger identity than older builds;
  **0** restores the old image-tokens-only behavior. Both regions share the prompt
  tokens. **More zones want a higher value, not lower**: each seam feathers away part
  of a zone's full-strength area (at the default `seam_feather` 0.08, 2 zones keep
  ~43% of the image each at full strength, 3 zones only ~26%/18%/25%), so each
  identity leans harder on the shared text pathway. Measured on real generations:
  **1.0 for 2 zones, ~2.0 for 3**. Lower it only if characters actually bleed. (Unrelated to the `clip`
  strength on stock loaders — that patches the separate text encoder; this stays in the
  model-only path.)
- **isolate_zones** *(default ON)* — hard containment. The model runs once per zone
  with that zone's LoRA applied globally (its own private copy of the text tokens
  included), and the outputs are composited with the zone masks in latent space, where
  every value has coordinates. Nothing can leak through the shared prompt — even LoRAs
  that live entirely in Krea2's text-conditioning layers stay in their box. Costs one
  extra full model pass per zone (2 zones ≈ 2× step time, plus one clean baseline pass
  when the boxes don't cover the whole canvas). Turn it OFF for the old single-pass
  activation blend, where `text_strength` is the leak-vs-likeness tradeoff.
- **mask_a / mask_b** *(experimental)* — feed hand-painted MASK tensors to override.

## The one rule that matters: place the box where the LoRA's features live
The box marks **where the LoRA is injected**, not just "where the character is."
Most character LoRAs are **face/portrait-trained**, so the box must cover where the
**head/face** will land — err generous, not tight. Body-trained LoRAs want the torso.
A box that misses the face gives weak identity. (This is why simple side-by-side
"just works" — each full-height column always contains a face.)

## Tips from testing
- With `text_strength` at its default, **include each LoRA's trigger word in the
  prompt** — the prompt-token pathway now carries identity too, and the trigger word is
  how a LoRA's text-side delta finds its character. The mask still controls placement.
- **Faces must not occlude each other.** Bodies can overlap freely; keep the faces in
  separate regions.
- Complex/tangled poses can produce anatomy errors — that's the **base model**, not this
  node. A pose ControlNet or a refiner pass helps.
- Results are **seed-dependent**; lock a good seed, then A/B your settings.

## Status
Works well for 2 and 3 characters across side-by-side, stacked, and overlapping comps.
Known rough edges: `bbox` wire + `mask_*` inputs are experimental/under-tested;
expression can flatten on the turbo model at CFG 1; a 4th zone would work mechanically
but the interior zones lose most of their full-strength area (see
[docs/TUNING.md](docs/TUNING.md)).

**In this release:** 3-zone nodes (1 or 3 LoRAs per zone); a second `model_b`
input/output so one node configures both stages of a two-pass workflow; `show_reference`
to draw the last generated image behind the region boxes; resize handles that stay
grabbable under overlapping boxes; `text_strength` ceiling raised to 4.0.

Earlier fixes (full writeup in `JOURNAL.md`): LoRA deltas reach the shared prompt tokens
and Krea2's internal text-conditioning stack (`text_strength`, previously dropped or
silently fraction-strength — the cause of "much weaker than a global load"); manual/bbox
region masks now peak at full strength inside their box; diffusers-named Krea2 LoRAs
match correctly; loud console warnings for unmatched targets and unsupported
(LoHa/LyCORIS) files.

Feedback welcome.
