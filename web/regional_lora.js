// Regional Character LoRA — in-node visual region editor (DOM canvas).
// Two draggable/resizable boxes (A = lora_a, B = lora_b). The canvas widget IS the
// "regions" input — it carries the normalized-coords JSON the Python node reads, so
// there is no separate text widget to leak. Active only when split_mode = "manual".
import { app } from "/scripts/app.js";

const COL_A = "#4ea1ff";
const COL_B = "#ff5d6c";
const HANDLE = 12;

const defaultRegions = () => ([
  { char: "a", x: 0.0, y: 0.0, w: 0.5, h: 1.0 },
  { char: "b", x: 0.5, y: 0.0, w: 0.5, h: 1.0 },
]);
const clamp01 = (v) => Math.max(0, Math.min(1, v));
function parseRegions(v) {
  try { const a = JSON.parse(v); if (Array.isArray(a) && a.length) return a; } catch (e) {}
  return defaultRegions();
}
function wval(node, name) {
  const w = node.widgets && node.widgets.find((w) => w.name === name);
  return w ? w.value : undefined;
}
function shortName(p) {
  if (!p || typeof p !== "string") return "";
  const s = p.split(/[\\/]/).pop().replace(/\.safetensors$/i, "");
  return s.length > 16 ? s.slice(0, 15) + "…" : s;
}

app.registerExtension({
  name: "RegionalCharacterLora.editor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Krea2RegionalCharacterLoRA" && nodeData.name !== "RegionalCharacterLora" &&
        nodeData.name !== "Krea2RegionalCharacterLoRAStack") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      const node = this;

      // remove the auto-created "regions" string widget; the canvas replaces it
      let initVal = JSON.stringify(defaultRegions());
      const autoIdx = node.widgets ? node.widgets.findIndex((w) => w.name === "regions") : -1;
      if (autoIdx >= 0) {
        if (node.widgets[autoIdx].value) initVal = node.widgets[autoIdx].value;
        node.widgets.splice(autoIdx, 1);
      }

      let regions = parseRegions(initVal);
      let store = JSON.stringify(regions);

      // editor preview aspect (cosmetic). canvas dims were removed; default portrait.
      const aspect = () => 1.5;

      const canvas = document.createElement("canvas");
      canvas.style.width = "100%";
      canvas.style.display = "block";
      canvas.style.marginTop = "8px";
      canvas.style.borderRadius = "6px";
      canvas.style.touchAction = "none";
      canvas.style.cursor = "crosshair";

      function draw() {
        const dpr = window.devicePixelRatio || 1;
        const cw = canvas.clientWidth || 200;
        const chh = canvas.clientHeight || 200;
        canvas.width = Math.round(cw * dpr);
        canvas.height = Math.round(chh * dpr);
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cw, chh);
        ctx.fillStyle = "#15151a";
        ctx.fillRect(0, 0, cw, chh);
        ctx.strokeStyle = "#3a3a42";
        ctx.strokeRect(0.5, 0.5, cw - 1, chh - 1);

        const nameA = shortName(wval(node, "lora_a")) || "A";
        const nameB = shortName(wval(node, "lora_b")) || "B";
        const active = wval(node, "split_mode") === "manual";

        for (const reg of regions) {
          const col = reg.char === "b" ? COL_B : COL_A;
          const x = reg.x * cw, y = reg.y * chh, w = reg.w * cw, h = reg.h * chh;
          ctx.globalAlpha = active ? 1 : 0.3;
          ctx.fillStyle = col + "22"; ctx.fillRect(x, y, w, h);
          ctx.lineWidth = 2; ctx.strokeStyle = col; ctx.strokeRect(x, y, w, h);
          ctx.fillStyle = col; ctx.fillRect(x + w - HANDLE, y + h - HANDLE, HANDLE, HANDLE);
          ctx.font = "11px sans-serif"; ctx.textBaseline = "top";
          ctx.fillText((reg.char === "b" ? "B " : "A ") + (reg.char === "b" ? nameB : nameA), x + 5, y + 4);
          ctx.globalAlpha = 1;
        }
        if (!active) {
          ctx.fillStyle = "#ddd"; ctx.font = "11px sans-serif";
          ctx.fillText("set split_mode = manual to use", 6, chh - 16);
        }
      }

      const widget = node.addDOMWidget("regions", "rcl_editor", canvas, {
        getValue() { return store; },
        setValue(v) { store = v; regions = parseRegions(v); draw(); },
        getMinHeight() {
          const w = node.size ? node.size[0] - 20 : 200;
          return Math.round(Math.max(140, Math.min(w * aspect(), 460)));
        },
        hideOnZoom: false,
      });
      widget.serializeValue = () => store;

      const sync = () => { store = JSON.stringify(regions); widget.value = store; };

      // interaction
      let drag = null;
      const toNorm = (e) => {
        const r = canvas.getBoundingClientRect();
        return [clamp01((e.clientX - r.left) / r.width), clamp01((e.clientY - r.top) / r.height)];
      };
      const onDown = (e) => {
        const r = canvas.getBoundingClientRect();
        const [nx, ny] = toNorm(e);
        for (let i = regions.length - 1; i >= 0; i--) {
          const reg = regions[i];
          if (Math.abs(nx - (reg.x + reg.w)) * r.width <= HANDLE &&
              Math.abs(ny - (reg.y + reg.h)) * r.height <= HANDLE) { drag = { i, mode: "resize" }; break; }
          if (nx >= reg.x && nx <= reg.x + reg.w && ny >= reg.y && ny <= reg.y + reg.h) {
            drag = { i, mode: "move", ox: nx - reg.x, oy: ny - reg.y }; break;
          }
        }
        if (drag) {
          e.preventDefault();
          window.addEventListener("pointermove", onMove);
          window.addEventListener("pointerup", onUp);
        }
      };
      const onMove = (e) => {
        if (!drag) return;
        const [nx, ny] = toNorm(e);
        const reg = regions[drag.i];
        if (drag.mode === "move") {
          reg.x = clamp01(nx - drag.ox); reg.y = clamp01(ny - drag.oy);
          if (reg.x + reg.w > 1) reg.x = 1 - reg.w;
          if (reg.y + reg.h > 1) reg.y = 1 - reg.h;
        } else {
          reg.w = Math.max(0.04, clamp01(nx - reg.x));
          reg.h = Math.max(0.04, clamp01(ny - reg.y));
        }
        sync(); draw();
      };
      const onUp = () => {
        drag = null; sync();
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      canvas.addEventListener("pointerdown", onDown);

      try { new ResizeObserver(() => draw()).observe(canvas); } catch (e) {}
      sync();
      setTimeout(draw, 50);
      const oldResize = node.onResize;
      node.onResize = function () { oldResize && oldResize.apply(this, arguments); draw(); };

      return r;
    };
  },
});
