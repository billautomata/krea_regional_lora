// Regional Character LoRA — in-node visual region editor (DOM canvas).
// One draggable/resizable box per zone (A = lora_a, B = lora_b, C = lora_c). The
// canvas widget IS the "regions" input — it carries the normalized-coords JSON the
// Python node reads, so there is no separate text widget to leak. Active only when
// split_mode = "manual".
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const COL = { a: "#4ea1ff", b: "#ff5d6c", c: "#5fd98a" };
const ZONES = ["a", "b", "c"];
const HANDLE = 12;

// Which zones this node actually has, read off its own widgets: lora_a (plain
// node) or lora_a1 (stack node). Falls back to A/B if widgets aren't up yet.
function nodeZones(node) {
  const has = (n) => node.widgets && node.widgets.some((w) => w.name === n);
  const z = ZONES.filter((c) => has("lora_" + c) || has("lora_" + c + "1"));
  return z.length ? z : ["a", "b"];
}
// Equal vertical strips, one per zone — a starting layout to drag from.
const defaultRegions = (zones) => zones.map((c, i) => (
  { char: c, x: i / zones.length, y: 0.0, w: 1 / zones.length, h: 1.0 }
));
const clamp01 = (v) => Math.max(0, Math.min(1, v));
function parseRegions(v, zones) {
  try { const a = JSON.parse(v); if (Array.isArray(a) && a.length) return a; } catch (e) {}
  return defaultRegions(zones);
}
function wval(node, name) {
  const w = node.widgets && node.widgets.find((w) => w.name === name);
  return w ? w.value : undefined;
}
// Every editor on the canvas, so a graph-wide image can be pushed to all of them.
const editors = new Set();
let lastGraphImages = null;

// The backdrop that needs no wiring: whatever image the graph last produced
// (SaveImage, PreviewImage, ...). It arrives AFTER this node ran, so it shows the
// run you just did - and because nothing is wired, there is no edge back into this
// node and no dependency cycle. There is no image input on the node at all.
api.addEventListener("executed", (e) => {
  const out = e && e.detail && e.detail.output;
  if (!out || !out.images || !out.images.length) return;
  lastGraphImages = out.images;
  for (const node of editors) {
    try {
      if (node._rclShowRef && node._rclShowRef()) node._rclBackground(lastGraphImages);
    } catch (err) { /* a dead node in the set must not break the others */ }
  }
});

function shortName(p) {
  if (!p || typeof p !== "string") return "";
  const s = p.split(/[\\/]/).pop().replace(/\.safetensors$/i, "");
  return s.length > 16 ? s.slice(0, 15) + "…" : s;
}

app.registerExtension({
  name: "RegionalCharacterLora.editor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!/^Krea2RegionalCharacterLoRA(3|Stack|Stack3)?$/.test(nodeData.name)) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      const node = this;

      const zones = nodeZones(node);

      // remove the auto-created "regions" string widget; the canvas replaces it
      let initVal = JSON.stringify(defaultRegions(zones));
      const autoIdx = node.widgets ? node.widgets.findIndex((w) => w.name === "regions") : -1;
      if (autoIdx >= 0) {
        if (node.widgets[autoIdx].value) initVal = node.widgets[autoIdx].value;
        node.widgets.splice(autoIdx, 1);
      }

      let regions = parseRegions(initVal, zones);
      let store = JSON.stringify(regions);

      // Reference backdrop. The WIDGET is resized to the image's aspect (see
      // getMinHeight) so the image fills it without stretching — the region boxes
      // keep spanning the whole canvas 0..1, exactly as they do with no backdrop.
      let bg = null;               // loaded <img>, or null
      const showRef = () => wval(node, "show_reference") === true;
      const aspect = () => (showRef() && bg ? bg.naturalHeight / bg.naturalWidth : 1.5);

      // Resizing the node is the fragile step (litegraph internals move between
      // frontend versions). It must never be able to stop the repaint, or the
      // backdrop would load and then never be drawn.
      const refit = () => {
        try {
          node.setSize([node.size[0], node.computeSize()[1]]);
          app.graph && app.graph.setDirtyCanvas(true, true);
        } catch (e) {
          console.warn("[RegionalCharacterLora] could not resize node:", e);
        }
        draw();
      };

      node._rclShowRef = showRef;
      editors.add(node);
      const onRemoved = node.onRemoved;
      node.onRemoved = function () {
        editors.delete(node);
        return onRemoved ? onRemoved.apply(this, arguments) : undefined;
      };

      node._rclBackground = (images) => {
        if (!images || !images.length) return;
        const it = images[0];
        const url = api.apiURL("/view?filename=" + encodeURIComponent(it.filename) +
                               "&type=" + encodeURIComponent(it.type || "output") +
                               "&subfolder=" + encodeURIComponent(it.subfolder || ""));
        const img = new Image();
        img.onload = () => {
          console.debug("[RegionalCharacterLora] backdrop", img.naturalWidth, "x",
                        img.naturalHeight);
          bg = img; refit();
        };
        img.onerror = () => {   // fail soft: plain backdrop, but say why
          console.warn("[RegionalCharacterLora] backdrop fetch failed:", url);
          bg = null; refit();
        };
        img.src = url;
      };

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
        if (showRef() && bg) {
          // the widget was resized to the image's aspect, so this fills it exactly
          try { ctx.drawImage(bg, 0, 0, cw, chh); } catch (e) {}
          ctx.fillStyle = "rgba(0,0,0,0.25)";        // knock it back under the boxes
          ctx.fillRect(0, 0, cw, chh);
        }
        ctx.strokeStyle = "#3a3a42";
        ctx.strokeRect(0.5, 0.5, cw - 1, chh - 1);

        const active = wval(node, "split_mode") === "manual";

        const box = (reg) => [reg.x * cw, reg.y * chh, reg.w * cw, reg.h * chh];
        ctx.globalAlpha = active ? 1 : 0.3;
        for (const reg of regions) {
          const ch = COL[reg.char] ? reg.char : "a";
          const col = COL[ch];
          const label = ch.toUpperCase() + " " +
            (shortName(wval(node, "lora_" + ch) || wval(node, "lora_" + ch + "1")) || "");
          const [x, y, w, h] = box(reg);
          ctx.fillStyle = col + "22"; ctx.fillRect(x, y, w, h);
          ctx.lineWidth = 2; ctx.strokeStyle = col; ctx.strokeRect(x, y, w, h);
          ctx.font = "11px sans-serif"; ctx.textBaseline = "top";
          ctx.fillStyle = col; ctx.fillText(label, x + 5, y + 4);
        }
        // Handles are drawn in their own pass, ON TOP of every box body, because
        // that is the order they are hit-tested in — a resize handle must stay
        // reachable when another region overlaps it, or overlapping boxes deadlock.
        for (const reg of regions) {
          const [x, y, w, h] = box(reg);
          const col = COL[COL[reg.char] ? reg.char : "a"];
          ctx.fillStyle = "#0009";
          ctx.fillRect(x + w - HANDLE - 1, y + h - HANDLE - 1, HANDLE + 2, HANDLE + 2);
          ctx.fillStyle = col;
          ctx.fillRect(x + w - HANDLE, y + h - HANDLE, HANDLE, HANDLE);
        }
        ctx.globalAlpha = 1;
        if (!active) {
          ctx.fillStyle = "#ddd"; ctx.font = "11px sans-serif";
          ctx.fillText("set split_mode = manual to use", 6, chh - 16);
        }
        if (showRef() && !bg) {
          ctx.fillStyle = "#888"; ctx.font = "11px sans-serif";
          ctx.fillText("reference: run once to capture the output", 6, 6);
        }
      }

      const widget = node.addDOMWidget("regions", "rcl_editor", canvas, {
        getValue() { return store; },
        setValue(v) { store = v; regions = parseRegions(v, zones); draw(); },
        getMinHeight() {
          const w = node.size ? node.size[0] - 20 : 200;
          // With a backdrop the widget takes the image's aspect EXACTLY, so the
          // image fills it without stretching and the boxes keep spanning the whole
          // canvas 0..1 as they always have. Clamping the height here is what broke
          // the aspect before - a capped canvas forced drawImage to distort.
          if (showRef() && bg) return Math.max(60, Math.round(w * aspect()));
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
        // Pass 1: EVERY resize handle, topmost first. Handles beat bodies globally,
        // so a box sitting on top of another can't swallow its corner and leave it
        // impossible to grab. Without this you have to shrink all the regions first
        // and place them one at a time to avoid a deadlock.
        for (let i = regions.length - 1; i >= 0 && !drag; i--) {
          const reg = regions[i];
          if (Math.abs(nx - (reg.x + reg.w)) * r.width <= HANDLE &&
              Math.abs(ny - (reg.y + reg.h)) * r.height <= HANDLE) {
            drag = { i, mode: "resize" };
          }
        }
        // Pass 2: bodies, topmost first.
        for (let i = regions.length - 1; i >= 0 && !drag; i--) {
          const reg = regions[i];
          if (nx >= reg.x && nx <= reg.x + reg.w && ny >= reg.y && ny <= reg.y + reg.h) {
            drag = { i, mode: "move", ox: nx - reg.x, oy: ny - reg.y };
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

      const refWidget = node.widgets && node.widgets.find((w) => w.name === "show_reference");
      if (refWidget) {
        const cb = refWidget.callback;
        refWidget.callback = function () {
          const r = cb ? cb.apply(this, arguments) : undefined;
          refit();
          return r;
        };
      }

      try { new ResizeObserver(() => draw()).observe(canvas); } catch (e) {}
      sync();
      setTimeout(() => {
        // a reloaded graph still has last run's outputs cached in the frontend
        if (lastGraphImages && showRef()) node._rclBackground(lastGraphImages);
        draw();
      }, 50);
      const oldResize = node.onResize;
      node.onResize = function () { oldResize && oldResize.apply(this, arguments); draw(); };

      return r;
    };
  },
});
