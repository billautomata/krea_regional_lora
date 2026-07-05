r"""
recon_krea2.py  —  run this FIRST, inside ComfyUI's embedded python.

  cd <ComfyUI root>
  python_embeded\python.exe ComfyUI\custom_nodes\regional_character_lora\recon_krea2.py ^
      --unet  "models\unet\krea2TurboOfficialComfy_krea2TurboFp8.safetensors" ^
      --lora  "models\loras\your_character_lora.safetensors"

It answers the three open recon questions from the handoff WITHOUT generating:
  Q1  fused (qkv) vs separate (to_q/to_k/to_v) attention projections, + MLP targets
  Q2  whether the LoRA touches attn only or attn+mlp
  Q3  LoRA key naming convention  ->  needed to map lora_key -> model_weight_key

It does NOT need a GPU. It only parses safetensors headers (no tensor load), so it
is safe to run even with ComfyUI open. The runtime token-count check (the live
x.shape print) is a separate snippet at the bottom you paste into the node during
the PoC — that one genuinely needs a generation to fire.
"""
import argparse, json, struct, sys, re
from collections import Counter, defaultdict


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def dump_unet(path):
    print("\n" + "=" * 70)
    print("UNET / DIFFUSION MODEL  :", path)
    print("=" * 70)
    hdr = read_header(path)
    keys = list(hdr.keys())
    print("total tensors:", len(keys))

    # Find the block-0 keys so we can see the per-block layout once.
    pats = ["blocks.0.", "double_blocks.0.", "single_blocks.0.",
            "transformer_blocks.0.", "joint_blocks.0."]
    blk = None
    for p in pats:
        sub = sorted(k for k in keys if p in k)
        if sub:
            blk = (p, sub)
            break
    if not blk:
        print("!! no recognised block-0 prefix; printing 40 sample keys:")
        for k in keys[:40]:
            print("   ", k)
        return
    prefix, sub = blk
    print(f"\nblock prefix detected: '{prefix}'  ({len(sub)} tensors in block 0)")
    print("--- block 0 weight keys (shapes) ---")
    for k in sub:
        if k.endswith(".weight") or k.endswith(".bias"):
            print(f"   {k:<60} {hdr[k]['shape']}")

    # Heuristic: fused qkv vs separate.
    flat = " ".join(sub)
    fused = bool(re.search(r"\bqkv\b|to_qkv|\.qkv\.", flat))
    sep = all(t in flat for t in ["to_q", "to_k", "to_v"]) or \
          all(t in flat for t in [".q.", ".k.", ".v."])
    print("\nQ1  attention projection style:",
          "FUSED qkv" if fused else ("SEPARATE q/k/v" if sep else "UNKNOWN - inspect above"))
    has_mlp = bool(re.search(r"mlp|ffn|feed_forward|\.fc1|\.fc2", flat))
    print("    MLP/FFN present in block:", has_mlp)

    # Count distinct block indices.
    idxs = set()
    for k in keys:
        m = re.search(re.escape(prefix.rstrip("0.")) + r"(\d+)\.", k)
        if m:
            idxs.add(int(m.group(1)))
    if idxs:
        print(f"    block count: {max(idxs)+1}  (indices {min(idxs)}..{max(idxs)})")


def dump_lora(path):
    print("\n" + "=" * 70)
    print("CHARACTER LORA          :", path)
    print("=" * 70)
    hdr = read_header(path)
    keys = list(hdr.keys())
    print("total tensors:", len(keys))

    # Group up/down/alpha triplets by their base module path.
    bases = defaultdict(dict)
    for k in keys:
        base = re.sub(r"\.(lora_up|lora_down|lora_A|lora_B|hada_w1_a|hada_w1_b|"
                      r"hada_w2_a|hada_w2_b|alpha)(\.weight)?$", "", k)
        if base == k:
            base = re.sub(r"\.(up|down)(\.weight)?$", "", k)
        tag = k[len(base):].lstrip(".")
        bases[base][tag] = hdr[k]["shape"]

    # Detect adapter type.
    flat = " ".join(keys)
    if "hada_w1_a" in flat:
        atype = "LoHa (LyCORIS) — node's raw up@down path will NOT work, needs special handling"
    elif re.search(r"lora_up|lora_down|lora_A|lora_B", flat):
        atype = "standard LoRA (up + down [+ alpha]) — node's up@down path works"
    else:
        atype = "UNKNOWN — inspect keys below"
    print("Q2/adapter type:", atype)

    print(f"\ndistinct target modules: {len(bases)}")
    # What kinds of layers are targeted?
    kinds = Counter()
    for base in bases:
        if re.search(r"qkv", base): kinds["attn.qkv"] += 1
        elif re.search(r"to_q|\.q\b|_q\b", base): kinds["attn.q"] += 1
        elif re.search(r"to_k|\.k\b|_k\b", base): kinds["attn.k"] += 1
        elif re.search(r"to_v|\.v\b|_v\b", base): kinds["attn.v"] += 1
        elif re.search(r"to_out|proj_out|\.o\b|out_proj", base): kinds["attn.out"] += 1
        elif re.search(r"mlp|ffn|fc1|fc2|feed_forward", base): kinds["mlp"] += 1
        else: kinds["other"] += 1
    print("Q3/target layer kinds:", dict(kinds))

    print("\n--- first 12 target modules (base -> tags) ---")
    for base in list(bases)[:12]:
        print(f"   {base}")
        for tag, shp in bases[base].items():
            print(f"        .{tag:<14} {shp}")
    print("\n>>> Compare a LoRA 'base' string above against a UNET block-0 weight key.")
    print(">>> The node maps them by stripping 'lora_unet_'/'diffusion_model.' prefixes")
    print(">>> and normalising '_' vs '.'.  Confirm the stems line up.")


RUNTIME_SNIPPET = r'''
# ---- Q (runtime token offset): paste inside the node wrapper for the PoC ----
# def regional_lora_wrapper(executor, x, timesteps, context, *a, **k):
#     print("[recon] x", tuple(x.shape), "context",
#           tuple(context.shape) if context is not None else None)
#     return executor(x, timesteps, context, *a, **k)
# x is [batch, seq, dim]; n_text = seq - n_image_tokens.
# n_image_tokens = (H//16)*(W//16). If n_text is constant across prompts -> padded.
'''

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet")
    ap.add_argument("--lora")
    args = ap.parse_args()
    if not args.unet and not args.lora:
        print("usage: recon_krea2.py --unet <unet.sft> --lora <char_lora.sft>")
        sys.exit(1)
    if args.unet: dump_unet(args.unet)
    if args.lora: dump_lora(args.lora)
    print(RUNTIME_SNIPPET)
