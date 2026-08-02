# Tuning: zone count, seam_feather, text_strength

**The short version:** adding a zone makes every character weaker, and the fix is to
turn `text_strength` **up**, not down. 1.0 is right for 2 zones; ~2.0 for 3. This is
the opposite of what "more LoRAs sharing the prompt = more bleed" suggests, which is
why it's written down here with the arithmetic behind it.

## Regenerate the numbers, don't trust this page

```
python test_masks.py --report      # the table below
python test_masks.py               # 9 correctness checks
```

Pure torch — no GPU, no ComfyUI, no LoRA files. If you're about to write a number
about masks into a tooltip, a README, or a Discord post, run that first.

## What the seam actually costs

```
full-strength share of the image per zone  (token grid 96x64)

  feather                 2 zones               3 zones               4 zones
  0.12                    39%/38%            22%/9%/21%         14%/1%/1%/12%
  0.08                    43%/42%           26%/18%/25%         18%/9%/9%/17%  <- default
  0.06                    45%/44%           28%/22%/27%       20%/14%/14%/19%
  0.04                    47%/46%           30%/26%/29%       22%/18%/18%/21%
  0.02                    49%/48%           32%/30%/31%       24%/22%/22%/23%
  0.00                    50%/50%           33%/33%/33%       25%/25%/25%/25%
```

`seam_feather` is a fraction of the **whole axis**, not of a slice. So a seam is the
same physical width no matter how many zones you have, and N zones have N-1 seams. Two
zones: one seam, each character loses one soft edge. Three zones: two seams, and the
middle character is soft on *both* sides — it holds full strength over 18% of the image
where a 2-up zone holds 42%. Four zones at the default feather leaves each interior
character 9%, and at 0.12 they're down to 1% — effectively erased.

Nothing in the code divides by zone count. Every zone's mask still peaks at exactly
1.0 inside its own region and the zones sum to exactly 1.0; the plateau just gets
narrower because the ramps eat inward from both sides.

## Why that means MORE text_strength

A character's identity arrives through two paths: the LoRA delta on its own **image
tokens**, and the delta on the **text tokens** every zone shares (the trigger-word
pathway). `text_strength` scales the second one.

Shrink the first path — fewer full-strength image tokens per character — and the
identity has to come from somewhere. It comes from the text path, so that path needs
to be louder. Measured on real generations: 3 zones at `text_strength` 1.0 produced
mush; at 2.0 all three characters resolved cleanly with no visual artifacts.

The widget max is **4.0** for this reason. If you're sitting at the cap and it still
isn't enough, that's a signal, not a plateau.

## Order to turn the knobs

1. **`text_strength` first.** 1.0 for 2 zones, start at 2.0 for 3. It's the strongest
   lever and it costs nothing structurally.
2. **`seam_feather` second, if the interior character is the weak one.** Dropping
   0.08 → 0.04 buys the middle zone 26% instead of 18%. It hardens the seams, which
   only shows where bodies overlap.
3. **`strength_a/b/c` last.** Per-LoRA trim for one that runs hot, not a fix for a
   structural shortfall — raising it doesn't give a character back its image tokens.
4. **`blend_override` only to let overlapping bodies mix.** It pulls every zone toward
   an even `1/N` floor (0.5 for 2 zones, 0.33 for 3), so the same blend value merges
   *harder* at 3 zones than at 2. At the default 0 it does nothing.

## Things that are deliberately NOT automatic

Do not "fix" any of the above by scaling `seam_feather` or `text_strength` by zone
count in code. Two reasons: it silently changes every existing 2-zone workflow, and it
hides the one relationship a user needs to understand to tune this node at all. The
knobs stay in absolute units and the tuning guidance lives here.

`auto` picks its axis from the latent's aspect ratio only — landscape gives
left/middle/right, portrait/square gives top/middle/bottom. Zone count doesn't enter
into it, on purpose.

## The mistake this page exists to prevent

The first version of the 3-zone node shipped with a tooltip claiming 3 zones bleed
harder than 2 and that you should *lower* `text_strength`. That was reasoning from a
plausible story about shared text tokens, never measured, and it was backwards. The
plateau table above takes two seconds to generate and would have caught it.

Measure before writing a number into a tooltip.
