# Journal

## 2026-07-05 — Why the regional path felt so much weaker than a global LoRA load

The node worked — two characters, two regions, no crash on fp8 — but the LoRAs came out
strikingly weaker than the same files loaded globally through the stock
`[load model] → [load lora] → [sampler]` path at the same strength. Weak enough to ask
whether the LoRA was really being loaded and used the same way at all. This entry records
the fundamental review of the injection path and the four real strength-loss mechanisms
it turned up. All four are now fixed.

The per-layer math was checked first and was **not** the problem: adding
`(x @ down.T) @ up.T` on a Linear's output is exactly equivalent to patching
`W + up @ down` into its weight, and the `strength × alpha/rank` scale convention matches
what ComfyUI's own loader does. The gaps were all in *where* the delta was allowed to land.

### 1. The LoRA's effect on the prompt tokens was thrown away

Krea2 is a single-stream DiT: the prompt tokens and image tokens run through the same 28
transformer blocks as one `[text | image]` sequence. A global LoRA load patches weights,
so it also changes how the *text* tokens are processed inside every block — the trigger
word's keys/values, which is a major identity pathway for a likeness LoRA. The regional
hook masked the delta to image tokens only and zeroed the text prefix, cutting that whole
pathway. This was the dominant gap.

**Change:** new `text_strength` widget on both nodes. The text prefix of every hooked
layer's mask is now `text_strength` instead of hard zero. Default 1.0 behaves like a
global load on the shared tokens; 0 restores the old behavior. Both sides' deltas land on
the same shared text tokens — that is the identity-strength vs. cross-bleed tradeoff, and
it's now a dial instead of a silent floor. (Not to be confused with the `clip` strength on
stock loaders: that patches the separate text-encoder model. Everything here stays inside
the diffusion model, i.e. the model-only path.)

### 2. The text-conditioning stack ran at a fraction of strength, silently

Parsing one of the actual character LoRAs off disk showed 256 trained layers — 32 of them
in Krea2's internal text-conditioning transformer (`txtfusion`), which the diffusion model
contains and trained LoRAs really do target. Those layers see a text-only sequence with no
image tail, so they fell into a silent fallback that scaled the delta by the *mean of the
region mask* — roughly 0.5 for a half-split, and as low as ~0.2 with small manual boxes.

**Change:** layers with no image-token tail now run uniformly at `text_strength`, and the
node prints a console note when that path triggers instead of doing it silently.

### 3. Manual/bbox masks never reached full strength inside their own box

The rectangle masks were built from a sigmoid product whose edge softness scales with the
*canvas* grid, not the box. Result: a box a quarter of the canvas wide peaked around 0.7 —
the LoRA ran at ~70% inside its own region, worse for smaller boxes. The geometric
auto-splits were unaffected (their ramp reaches exactly 1.0), so this only hit `manual`
and `bbox` modes — which compounded with #1 and #2 for exactly the workflows the visual
editor encourages.

**Change:** rect masks are peak-normalized to 1.0 before use.

### 4. Diffusers-named Krea2 LoRAs matched zero layers, silently

The character LoRAs use native key names (`blocks.N.attn.wq`) and matched 256/256. But
Krea2 LoRAs in the diffusers naming scheme (`transformer_blocks.N.attn.to_q`,
`ff.*`, `text_fusion.*` — e.g. the darkbrush style LoRA) matched **nothing**, because the
name normalizer only stripped prefixes and never translated the vocabulary. The stock
loader translates it via `comfy.utils.krea2_to_diffusers`; the node didn't. Stacking such
a file in a zone contributed zero, with no warning.

**Change:** the same rename table is now applied inside the signature normalizer
(`_SIG_RENAMES`), verified 264/264 against the real diffusers-format file. The layer
mapper also warns when some LoRA targets matched no layer, and the loader warns loudly
when a file contains no supported LoRA weights at all (LoHa/LyCORIS/`diff`-only files
previously just did nothing).

### Verification status

Syntax-checked, and the key-matching + mask changes were exercised directly against the
real safetensors files on disk (both naming schemes, full match). The generation-quality
half — does `text_strength = 1.0` recover global-load identity, and where does bleed set
in — needs real runs on the GPU box: watch the `[RegionalCharacterLora]` console lines
(matched-layer count, the new conditioning-stack note) and walk `text_strength` down from
1.0 if the characters contaminate each other.
