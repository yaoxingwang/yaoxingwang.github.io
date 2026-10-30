// The introduction: the mark right after the title shows it in place of the panel's content, as tall as its text
// allows in the window (style.css #panel.intro); a second click brings the panel back as it was.  The markup is in
// index.html (#intro-box).
import { $ } from "./dom.js";
import { state } from "./state.js";
import { viewer } from "./scene.js";

/**
 * The GPU the page uses, by name: the 3D view's WebGL renderer string (Chrome and Edge give the model, e.g.
 * "ANGLE (NVIDIA, NVIDIA GeForce GTX 1660 Ti (0x00002191) Direct3D11 vs_5_0 ps_5_0, D3D11)"); where the browser
 * hides the model, the WebGPU adapter's vendor and architecture ("NVIDIA Turing"); null when neither is known.
 */
async function gpuInUse() {
  try {
    const angle = /^ANGLE \((.*)\)$/.exec(viewer.rendererName());
    if (angle) {
      const name = angle[1].split(", ")[1]
        ?.replace(/^ANGLE Metal Renderer: /, "")
        .replace(/^Vulkan [0-9.]+ \((.*)\)$/, "$1")
        .replace(/\s*\(0x[0-9a-f]+\).*$/i, "")
        .replace(/\s+(Direct3D|OpenGL|Vulkan|vs_|D3D).*$/, "")
        .replace(/\/PCIe.*$/, "")
        .replace(/\((R|TM)\)/g, "")
        .replace(/^(\S+) \1 /, "$1 ")
        .trim();
      if (name) return name;
    }
  } catch {
    // fall through to WebGPU
  }
  try {
    // the device's adapter; before the device exists (the introduction opened while loading), one asked for now
    const adapter = state.gpuInfo ? null : await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    const info = state.gpuInfo || adapter?.info || (await adapter?.requestAdapterInfo?.());
    const word = (w) => (["nvidia", "amd", "arm"].includes(w) ? w.toUpperCase() : w[0].toUpperCase() + w.slice(1));
    const name = [info?.vendor, info?.architecture].filter(Boolean).map(word).join(" ");
    if (name) return name;
  } catch {
    // unknown
  }
  return null;
}

/** Whether the introduction is shown (in place of the panel's content). */
export const introOpen = () => $("panel").classList.contains("intro");

let afterClose = () => {};            // set by setupIntroduction

/** Show the introduction in place of the panel's content, or close it (the panel is then fitted again: afterClose). */
export function showIntroduction(on) {
  const panel = $("panel"), box = $("intro-box");
  if (on === introOpen()) return;
  if (on) panel.scrollTop = 0;          // a short window's scrolled panel: the introduction from its title
  panel.classList.toggle("intro", on);
  box.hidden = !on;
  $("intro-mark").setAttribute("aria-pressed", String(on));
  if (on) box.scrollTop = 0;
  if (on && !$("intro-gpu").textContent) {
    gpuInUse().then((name) => {
      if (!name) return;
      $("intro-gpu").textContent = name;
      box.querySelector(".intro-device").hidden = false;
    });
  }
  if (!on) afterClose();
}

/** Wire the introduction mark; onClose() runs after the introduction closes (main.js: the panel's fit). */
export function setupIntroduction(onClose) {
  afterClose = onClose;
  const mark = $("intro-mark");
  mark.hidden = false;                // shown once it works (the page's scripts have loaded)
  mark.addEventListener("click", () => showIntroduction(!introOpen()));
}
