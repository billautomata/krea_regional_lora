"""Mask-math self-check for the regional node. Pure torch — no GPU, no ComfyUI,
no LoRA files. Run it wherever torch is installed:

    python_embeded\\python.exe ComfyUI\\custom_nodes\\regional_character_lora\\test_masks.py

Covers the N-zone generalisation: that 2 zones still behave exactly as the old
half/half code did, and that 3 zones slice in reading order (A = top/left).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from regional_character_lora import (
    _RegionalSession, _apply_blend, _bake_stacks, _masks_from_boxlist,
    _masks_from_grid, _masks_from_regions, _norm, _resolve_auto_split,
    _smoothstep_ramp,
)


def session(patcher, stacks, split_mode="auto", feather=0.05, blend=0.0,
            mask_ins=None, text_strength=1.0):
    """Build a session the way a node does: bake the stacks first."""
    return _RegionalSession(patcher, _bake_stacks(stacks), split_mode, feather,
                            blend, None, mask_ins or {}, "", text_strength)

ROWS, COLS, F = 12, 12, 0.05


def row(m, r):      # one grid row out of the flattened token mask
    return m.view(ROWS, COLS)[r]


def col(m, c):
    return m.view(ROWS, COLS)[:, c]


def cell(m, c):
    """One token from the middle row. Rect masks (manual/bbox) feather across the
    canvas edge, so the outermost row/column sits near 0.5 and the next one in is
    still ~0.9 — sample region centres, not their borders."""
    return float(m.view(ROWS, COLS)[ROWS // 2, c])


def test_two_zones_unchanged():
    """N=2 must reproduce the old (ramp, 1-ramp) pair bit for bit."""
    a, b = _masks_from_grid("vertical_auto", ROWS, COLS, F, 0.0, 2)
    ramp = _smoothstep_ramp(COLS, COLS / 2 - F * COLS, COLS / 2 + F * COLS)
    old_a = ramp.unsqueeze(0).expand(ROWS, COLS).reshape(-1)
    assert torch.allclose(a, old_a), "2-zone A drifted from the old ramp"
    assert torch.allclose(b, 1.0 - old_a), "2-zone B drifted from the old 1-ramp"


def test_three_zones_reading_order():
    a, b, c = _masks_from_grid("vertical_auto", ROWS, COLS, F, 0.0, 3)
    assert col(a, 0).mean() > 0.99 and col(a, COLS - 1).mean() < 0.01, "A is not the left slice"
    assert col(c, 0).mean() < 0.01 and col(c, COLS - 1).mean() > 0.99, "C is not the right slice"
    assert col(b, COLS // 2).mean() > 0.99, "B is not the middle slice"
    total = a + b + c
    assert total.min() > 0.98 and total.max() < 1.02, "slices don't partition the grid"

    # horizontal = top / middle / bottom, same reading order down the rows
    a, b, c = _masks_from_grid("horizontal_auto", ROWS, COLS, F, 0.0, 3)
    assert row(a, 0).mean() > 0.99 and row(c, ROWS - 1).mean() > 0.99, "T/M/B order is wrong"


def test_auto_heuristic_is_unchanged_for_three():
    """A 3-zone portrait latent must still resolve to top/middle/bottom — the
    same landscape/portrait heuristic the 2-zone node uses, no special case."""
    assert _resolve_auto_split(rows=96, cols=64) == "horizontal_auto"
    assert _resolve_auto_split(rows=64, cols=96) == "vertical_auto"

    sess = session(None, {z: [] for z in "abc"})
    masks, used = sess._build_masks_now(ROWS, COLS)   # square -> portrait branch
    assert used == "horizontal_auto", used
    a, b, c = masks
    assert row(a, 0).mean() > 0.99, "A is not the top band on a portrait/square latent"
    assert row(c, ROWS - 1).mean() > 0.99, "C is not the bottom band"


def test_blend_pulls_to_one_over_n():
    for n in (2, 3):
        masks = _masks_from_grid("vertical_auto", ROWS, COLS, F, 1.0, n)
        for m in masks:
            assert torch.allclose(m, torch.full_like(m, 1.0 / n)), f"blend=1 wrong at n={n}"


def test_regions_three_zones():
    js = ('[{"char":"a","x":0,"y":0,"w":0.33,"h":1},'
          ' {"char":"b","x":0.34,"y":0,"w":0.32,"h":1},'
          ' {"char":"c","x":0.67,"y":0,"w":0.33,"h":1}]')
    a, b, c = _masks_from_regions(js, ROWS, COLS, F, 0.0, ("a", "b", "c"))
    assert cell(a, 2) > 0.9 and cell(c, COLS - 2) > 0.9   # region centres
    assert cell(b, COLS // 2) > 0.9

    # exactly one zone left undrawn -> it gets what the others don't cover
    two = '[{"char":"a","x":0,"y":0,"w":0.4,"h":1},{"char":"b","x":0.4,"y":0,"w":0.3,"h":1}]'
    a, b, c = _masks_from_regions(two, ROWS, COLS, F, 0.0, ("a", "b", "c"))
    assert cell(c, COLS - 2) > 0.9, "the single undrawn zone got no complement"
    assert cell(c, 2) < 0.1

    # two zones undrawn is ambiguous -> they stay empty (those LoRAs are inert)
    one = '[{"char":"a","x":0,"y":0,"w":0.4,"h":1}]'
    a, b, c = _masks_from_regions(one, ROWS, COLS, F, 0.0, ("a", "b", "c"))
    assert b.max() == 0 and c.max() == 0, "ambiguous gaps should not be filled"
    assert _masks_from_regions("[]", ROWS, COLS, F, 0.0, ("a", "b", "c")) is None


def test_boxlist_complement():
    boxes = [[0.0, 0.0, 0.5, 1.0]]          # 2 zones, 1 box -> B = not-A (old rule)
    a, b = _masks_from_boxlist(boxes, ROWS, COLS, COLS * 16, ROWS * 16, F, 0.0, 2)
    assert cell(a, 2) > 0.9 and cell(b, COLS - 2) > 0.9
    boxes = [[0.0, 0.0, 0.33, 1.0], [0.34, 0.0, 0.66, 1.0]]   # 3 zones, 2 boxes
    a, b, c = _masks_from_boxlist(boxes, ROWS, COLS, COLS * 16, ROWS * 16, F, 0.0, 3)
    assert cell(c, COLS - 2) > 0.9, "trailing zone got no complement"


def test_mask_socket_gap():
    """Painted sockets follow the same one-gap rule as everything else."""
    left, mid = torch.zeros(1, 64, 64), torch.zeros(1, 64, 64)
    left[:, :, :20] = 1.0
    mid[:, :, 20:40] = 1.0
    sess = session(None, {z: [] for z in "abc"}, mask_ins={"a": left, "b": mid})
    masks, used = sess._build_masks_now(ROWS, COLS)
    assert used == "mask-socket"
    assert col(masks[0], 0).mean() > 0.9
    assert col(masks[2], COLS - 1).mean() > 0.9, "unwired c should be the complement"

    # two unwired zones is ambiguous -> both stay empty
    sess = session(None, {z: [] for z in "abc"}, mask_ins={"a": left})
    masks, _ = sess._build_masks_now(ROWS, COLS)
    assert masks[1].max() == 0 and masks[2].max() == 0


def test_hook_lands_in_its_own_zone():
    """End-to-end through the real forward hook: zone C's LoRA delta must show up
    only on the bottom third of the image tokens, and nowhere in the text prefix
    at text_strength=0. Catches mask/zone misalignment the mask tests can't."""
    tiny, mats = _tiny_lora()                     # delta[..., 0] = x[..., 0]
    assert _norm("attn_wq") == "attnwq"

    class Patcher:                                # only .model is touched
        model = tiny

    sess = session(Patcher(), {"a": [], "b": [], "c": [(mats, 1.0)]},
                   "horizontal_auto", 0.02, text_strength=0.0)
    n_text, grid = 2, 24                          # latent 24x24 -> 12x12 token grid
    tokens = torch.ones(1, n_text + 144, 4)

    def executor(latent, seq):
        return tiny.attn_wq(seq)

    out = sess.run(executor, torch.zeros(1, 4, grid, grid), tokens)
    got = out[0, :, 0]                            # base weights are 0 -> this IS the mask
    assert got[:n_text].abs().max() < 1e-3, "text tokens moved at text_strength=0"
    img = got[n_text:].view(12, 12)               # C owns rows 8-11 (bottom third)
    assert img[11].min() > 0.99, "C's delta missed the bottom row of tokens"
    assert img[0].max() < 0.01, "C's delta leaked into A's rows"
    assert img[6].max() < 0.01, "C's delta leaked into B's rows"


def _tiny_lora():
    """A 1-layer model + a LoRA whose delta copies input channel 0 to output 0."""
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_wq = torch.nn.Linear(4, 4)
            torch.nn.init.zeros_(self.attn_wq.weight)
            torch.nn.init.zeros_(self.attn_wq.bias)

    up = torch.zeros(4, 2); up[0, 0] = 1.0
    return Tiny(), {"attnwq": {"down": torch.eye(4)[:2], "up": up, "scale": 1.0}}


def test_two_models_hook_their_own_layers():
    """model + model_b share one config and one set of LoRA matrices, but each
    needs its OWN session: a session holds direct references to one model's
    nn.Linear objects. If session B reused A's layer map it would inject into the
    wrong model with no error at all — the whole reason this test exists."""
    tiny_a, mats = _tiny_lora()
    tiny_b, _ = _tiny_lora()
    # exactly what a node does: bake once, hand the SAME maps to both sessions
    zone_maps = _bake_stacks({"a": [], "b": [], "c": [(mats, 1.0)]})

    def make(model):
        class P:
            pass
        p = P(); p.model = model
        return _RegionalSession(p, zone_maps, "horizontal_auto", 0.02, 0.0,
                                None, {}, "", 0.0)

    sess_a, sess_b = make(tiny_a), make(tiny_b)
    latent, tokens = torch.zeros(1, 4, 24, 24), torch.ones(1, 2 + 144, 4)
    seen = {}

    def exec_b(lat, seq):
        seen["a_while_b_runs"] = tiny_a.attn_wq(seq)     # A must stay untouched
        return tiny_b.attn_wq(seq)

    sess_a.run(lambda lat, seq: tiny_a.attn_wq(seq), latent, tokens)
    out_b = sess_b.run(exec_b, latent, tokens)

    assert out_b[0, -1, 0] > 0.99, "model_b got no delta — its own layer wasn't hooked"
    assert seen["a_while_b_runs"].abs().max() < 1e-6, \
        "model_b's session injected into model_a — wrong model, silently"

    # the LoRA matrices themselves ARE shared: same dict object, one copy in VRAM
    da = sess_a._layer_map["attn_wq"][1]["c"][0]
    db = sess_b._layer_map["attn_wq"][1]["c"][0]
    assert da is db, "matrices got duplicated per model instead of shared"
    assert da["down_d"].device == torch.device("cpu")


def test_isolation_contains_the_text_pathway():
    """isolate=True: B's LoRA runs GLOBALLY inside its own pass (text tokens
    included — the exact pathway that leaks everywhere in single-pass mode), yet
    the composite keeps the effect strictly inside B's latent slice. Zone A is
    empty, so coverage has a gap and a clean baseline pass must run."""
    tiny, mats = _tiny_lora()

    class P:
        model = tiny

    sess = _RegionalSession(P(), _bake_stacks({"a": [], "b": [(mats, 1.0)]}),
                            "horizontal_auto", 0.0, 0.0, None, {}, "", 1.0,
                            isolate=True)
    latent = torch.zeros(1, 4, 24, 24)
    tokens = torch.ones(1, 2 + 144, 4)
    text_deltas = []

    def executor(lat, seq):
        o = tiny.attn_wq(seq)
        text_deltas.append(float(o[0, 0, 0]))            # delta on a TEXT token
        return torch.full_like(lat, float(o[0, -1, 0]))  # uniform image effect

    out = sess.run(executor, latent, tokens)
    assert len(text_deltas) == 2, "expected B's pass + one clean baseline pass"
    assert max(text_deltas) > 0.99, "B's pass didn't apply its LoRA to the text tokens"
    assert out[0, 0, 23].min() > 0.99, "B's zone (bottom) lost its LoRA effect"
    assert out[0, 0, 0].max() < 0.01, "B leaked outside its zone despite isolation"


def test_isolation_full_coverage_skips_clean_pass():
    """Both zones filled -> the grid masks sum to 1, so exactly one pass per zone
    (no clean baseline) and each half shows only its own zone's effect."""
    tiny, mats = _tiny_lora()

    class P:
        model = tiny

    sess = _RegionalSession(P(),
                            _bake_stacks({"a": [(mats, 1.0)], "b": [(mats, 2.0)]}),
                            "horizontal_auto", 0.0, 0.0, None, {}, "", 1.0,
                            isolate=True)
    latent = torch.zeros(1, 4, 24, 24)
    tokens = torch.ones(1, 2 + 144, 4)
    calls = []

    def executor(lat, seq):
        calls.append(1)
        return torch.full_like(lat, float(tiny.attn_wq(seq)[0, -1, 0]))

    out = sess.run(executor, latent, tokens)
    assert len(calls) == 2, "full coverage should not need a clean pass"
    assert abs(float(out[0, 0, 0, 0]) - 1.0) < 0.01, "A's half isn't A's output"
    assert abs(float(out[0, 0, 23, 0]) - 2.0) < 0.01, "B's half isn't B's output"


class _FakePatcher:
    """Stands in for a ComfyUI ModelPatcher: clone + wrapper registration only."""
    def __init__(self, tag):
        self.tag = tag
        self.model = _tiny_lora()[0]
        self.wrappers = {}

    def clone(self):
        c = _FakePatcher(self.tag)
        c.model = self.model
        return c

    def add_wrapper_with_key(self, kind, key, fn):
        self.wrappers.setdefault(key, []).append(fn)


def test_node_drives_both_models_from_one_config():
    import regional_character_lora as R

    mats = _tiny_lora()[1]
    real_load, real_opt = R._load_lora_matrices, R._load_optional_lora
    R._load_lora_matrices = lambda path: mats
    R._load_optional_lora = lambda name: (None if not name or name == "None" else mats)
    try:
        for name, cls in R.NODE_CLASS_MAPPINGS.items():
            kw = {}
            for z in cls.ZONES:                       # fill whichever slot shape
                kw[f"lora_{z}"] = "x.safetensors"
                for i in range(1, 4):
                    kw[f"lora_{z}{i}"] = "x.safetensors"
            a, b = _FakePatcher("a"), _FakePatcher("b")
            out = cls().apply(a, "auto", 0.08, 0.0, model_b=b, **kw)

            assert isinstance(out, tuple) and len(out) == 2, f"{name}: not 2 outputs"
            assert out[0] is not a and out[1] is not b, f"{name}: model not cloned"
            assert out[0] is not out[1], f"{name}: both outputs are the same clone"
            for i, patched in enumerate(out):
                assert patched.wrappers, f"{name}: output {i} got no wrapper"

            # model_b unwired -> second output is None, first is untouched
            solo = cls().apply(_FakePatcher("a"), "auto", 0.08, 0.0, **kw)
            assert solo[1] is None, f"{name}: model_b absent should give None"
            assert solo[0].wrappers, f"{name}: single-model path broke"

            # show_reference off -> plain tuple, no temp file written, no ui dict
            assert not isinstance(solo, dict), f"{name}: ui returned with flag off"
    finally:
        R._load_lora_matrices, R._load_optional_lora = real_load, real_opt


def test_apply_blend_list_identity():
    m = [torch.rand(8), torch.rand(8)]
    assert all(torch.allclose(x, y) for x, y in zip(_apply_blend(m, 0.0), m))


def report():
    """Print the zone-count / seam_feather plateau table. This is the measurement
    behind docs/TUNING.md — regenerate it instead of trusting remembered numbers."""
    R, C = 96, 64                      # a 1024x1536 portrait latent's token grid
    print(f"full-strength share of the image per zone  (token grid {R}x{C})\n")
    print(f"  {'feather':<9}" + "".join(f"{n} zones".rjust(22) for n in (2, 3, 4)))
    for f in (0.12, 0.08, 0.06, 0.04, 0.02, 0.0):
        cells = []
        for n in (2, 3, 4):
            ms = _masks_from_grid("horizontal_auto", R, C, f, 0.0, n)
            pct = [100 * (m.view(R, C)[:, 0] > 0.999).sum().item() / R for m in ms]
            cells.append("/".join(f"{p:.0f}%" for p in pct).rjust(22))
        star = "  <- default" if f == 0.08 else ""
        print(f"  {f:<9.2f}" + "".join(cells) + star)
    print("\nEach seam feathers away part of two zones. N zones have N-1 seams, so"
          "\nthe interior zones are soft on BOTH sides and lose the most area.")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
        raise SystemExit(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} checks passed")
