(function () {
  "use strict";
  const D = window.DARA_DATA;
  const C = {
    GRPO: "#8c8c8c", GDPO: "#2e9e44", DVAO: "#f28e2b", "GD²PO-Hard": "#9467bd",
    "DARA-Asym": "#d1453b", "DARA-Sym": "#2563eb",
  };
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const fmt = (v, d) => (v == null || isNaN(v) ? "–" : Number(v).toFixed(d));

  /* ------------------------------------------------------------------
   * Minimal SVG line chart: series {name,color,dash,x[],y[],lo[],hi[]}
   * ------------------------------------------------------------------ */
  function lineChart(host, opt) {
    const W = opt.width || 520, H = opt.height || 300;
    const m = Object.assign({ t: 30, r: 14, b: 42, l: 50 }, opt.margin || {});
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const wrap = document.createElement("div");
    wrap.className = "chart";
    host.appendChild(wrap);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opt.title || "chart" }, wrap);
    const [x0, x1] = opt.x, [y0, y1] = opt.y;
    const sx = (v) => m.l + ((v - x0) / (x1 - x0)) * iw;
    const sy = (v) => m.t + ih - ((v - y0) / (y1 - y0)) * ih;

    if (opt.title) el("text", { x: m.l, y: 16, class: "ttl" }, svg).textContent = opt.title;
    const ax = el("g", { class: "axis" }, svg);
    (opt.yTicks || []).forEach((t) => {
      el("line", { x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t), class: "gridl" }, ax);
      const tx = el("text", { x: m.l - 7, y: sy(t) + 4, "text-anchor": "end" }, ax);
      tx.textContent = opt.yFmt ? opt.yFmt(t) : t;
    });
    (opt.xTicks || []).forEach((t) => {
      const tx = el("text", { x: sx(t), y: m.t + ih + 16, "text-anchor": "middle" }, ax);
      tx.textContent = t;
    });
    el("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }, ax);
    el("line", { x1: m.l, x2: m.l, y1: m.t, y2: m.t + ih }, ax);
    if (opt.xLabel) el("text", { x: m.l + iw / 2, y: H - 6, "text-anchor": "middle", class: "albl" }, svg).textContent = opt.xLabel;
    if (opt.yLabel) {
      const t = el("text", { x: 0, y: 0, "text-anchor": "middle", class: "albl", transform: `translate(13 ${m.t + ih / 2}) rotate(-90)` }, svg);
      t.textContent = opt.yLabel;
    }
    (opt.hlines || []).forEach((h) => {
      el("line", { x1: m.l, x2: m.l + iw, y1: sy(h.y), y2: sy(h.y), stroke: h.color || "#999", "stroke-dasharray": "5 4", "stroke-width": 1.2 }, svg);
      if (h.label) el("text", { x: m.l + iw - 4, y: sy(h.y) - 5, "text-anchor": "end", class: "albl", fill: h.color || "#666" }, svg).textContent = h.label;
    });
    (opt.vspans || []).forEach((v) => {
      el("rect", { x: sx(v.from), y: m.t, width: sx(v.to) - sx(v.from), height: ih, fill: v.color || "rgba(0,0,0,.04)" }, svg);
      if (v.label) el("text", { x: sx(v.from) + 5, y: m.t + 13, class: "albl", fill: "#8a8f99" }, svg).textContent = v.label;
    });

    const layers = {};
    const gS = el("g", {}, svg);
    opt.series.forEach((s) => {
      const g = el("g", {}, gS);
      if (s.lo && s.hi) {
        let d = "";
        s.x.forEach((xv, i) => (d += (i ? "L" : "M") + sx(xv) + "," + sy(s.hi[i])));
        for (let i = s.x.length - 1; i >= 0; i--) d += "L" + sx(s.x[i]) + "," + sy(s.lo[i]);
        el("path", { d: d + "Z", fill: s.color, "fill-opacity": 0.13, stroke: "none" }, g);
      }
      let d = "";
      s.x.forEach((xv, i) => (d += (i ? "L" : "M") + sx(xv) + "," + sy(s.y[i])));
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2.2, "stroke-dasharray": s.dash ? "6 4" : "none", "stroke-linejoin": "round" }, g);
      if (s.markers) s.x.forEach((xv, i) => el("circle", { cx: sx(xv), cy: sy(s.y[i]), r: 2.8, fill: s.color }, g));
      layers[s.name] = g;
    });

    const hidden = new Set();
    return {
      toggle(name, on) {
        if (!layers[name]) return;
        layers[name].style.display = on ? "" : "none";
        on ? hidden.delete(name) : hidden.add(name);
      },
    };
  }

  function legend(host, items, charts) {
    const box = document.createElement("div");
    box.className = "legend";
    items.forEach((it) => {
      const b = document.createElement("button");
      b.innerHTML = `<span class="ln${it.dash ? " dash" : ""}" style="border-color:${it.color}"></span>${it.name}`;
      b.addEventListener("click", () => {
        b.classList.toggle("off");
        charts.forEach((c) => c.toggle(it.name, !b.classList.contains("off")));
      });
      box.appendChild(b);
    });
    host.prepend(box);
  }

  /* ---------- Figure 1(b): format reward & active-group density ---------- */
  function fig1b() {
    const host = document.getElementById("chart-fig1b");
    if (!host) return;
    const order = ["GRPO", "GDPO", "DARA-Asym", "DARA-Sym"];
    const row = document.createElement("div");
    row.className = "grid2";
    host.appendChild(row);
    const a = document.createElement("div"), b = document.createElement("div");
    row.appendChild(a); row.appendChild(b);
    const mk = (panel) => order.map((n) => {
      const s = D.fig1b[panel][n];
      return { name: n, color: C[n], dash: n === "DARA-Asym", x: s.step, y: s.med, lo: s.q25, hi: s.q75 };
    });
    const c1 = lineChart(a, {
      title: "Format reward (median, IQR band)", series: mk("format_reward"),
      x: [0, 100], y: [0, 1], xTicks: [0, 20, 40, 60, 80, 100], yTicks: [0, .2, .4, .6, .8, 1],
      xLabel: "Training step", hlines: [{ y: 0.8, color: "#b0b4bc", label: "0.8" }],
    });
    const c2 = lineChart(b, {
      title: "Format active-group density (median, IQR band)", series: mk("active_group_density"),
      x: [0, 100], y: [0, 0.6], xTicks: [0, 20, 40, 60, 80, 100], yTicks: [0, .2, .4, .6],
      xLabel: "Training step",
    });
    legend(host, order.map((n) => ({ name: n, color: C[n], dash: n === "DARA-Asym" })), [c1, c2]);
  }

  /* ---------- BFCL-v4 checkpoint dynamics ---------- */
  function bfcl() {
    const host = document.getElementById("chart-bfcl");
    if (!host) return;
    const order = ["GRPO", "GDPO", "DVAO", "GD²PO-Hard", "DARA-Asym", "DARA-Sym"];
    const row = document.createElement("div");
    row.className = "grid2";
    host.appendChild(row);
    const a = document.createElement("div"), b = document.createElement("div");
    row.appendChild(a); row.appendChild(b);
    const mk = (key) => order.map((n) => ({ name: n, color: C[n], x: D.bfcl[n].step, y: D.bfcl[n][key], markers: true, width: n.startsWith("DARA") ? 2.6 : 1.8 }));
    const pct = (v) => fmt(v, 2) + "%";
    const c1 = lineChart(a, {
      title: "BFCL-v4 Average Accuracy (%)", series: mk("acc"), x: [0, 100], y: [0, 55],
      xTicks: [0, 20, 40, 60, 80, 100], yTicks: [0, 10, 20, 30, 40, 50], xLabel: "Training step", tipFmt: pct,
    });
    const c2 = lineChart(b, {
      title: "BFCL-v4 Average Format (%)", series: mk("fmt"), x: [0, 100], y: [0, 100],
      xTicks: [0, 20, 40, 60, 80, 100], yTicks: [0, 20, 40, 60, 80, 100], xLabel: "Training step", tipFmt: pct,
    });
    legend(host, order.map((n) => ({ name: n, color: C[n] })), [c1, c2]);
  }

  /* ---------- energy ratio on logged training batches ---------- */
  function energy() {
    const host = document.getElementById("chart-energy");
    if (!host) return;
    const names = { "1.5b": "DeepSeek-R1-1.5B", "4b": "Qwen3-4B-Instruct", "7b": "DeepSeek-R1-7B" };
    const tabs = document.getElementById("tabs-energy");
    const box = document.createElement("div");
    host.appendChild(box);
    function draw(scale) {
      box.innerHTML = "";
      const e = D.energy[scale];
      const capStep = e.step[e.cap.indexOf(1)];
      const ch = lineChart(box, {
        title: names[scale] + " — length / correctness energy ratio",
        series: [
          { name: "GDPO normalization", color: "#8c8c8c", x: e.step, y: e.gdpo, width: 2 },
          { name: "DARA weighting", color: "#d1453b", x: e.step, y: e.dara, width: 2.6 },
        ],
        x: [0, 100], y: [0, 1.4], xTicks: [0, 20, 40, 60, 80, 100], yTicks: [0, .5, 1],
        xLabel: "Training step", yLabel: "E_length / E_correct",
        width: 900, height: 300, margin: { l: 56 },
        hlines: [{ y: 1, color: "#555", label: "equal energy" }],
        vspans: [{ from: capStep, to: 100, color: "rgba(209,69,59,.05)", label: `weight cap w_max = 5 reached (step ${capStep})` }],
      });
      legend(box, [{ name: "GDPO normalization", color: "#8c8c8c" }, { name: "DARA weighting", color: "#d1453b" }], [ch]);
    }
    Object.keys(names).forEach((k, i) => {
      const b = document.createElement("button");
      b.textContent = names[k];
      if (i === 0) b.classList.add("on");
      b.addEventListener("click", () => {
        tabs.querySelectorAll("button").forEach((x) => x.classList.remove("on"));
        b.classList.add("on");
        draw(k);
      });
      tabs.appendChild(b);
    });
    draw("1.5b");
  }

  /* ---------- density / energy playground ---------- */
  function playground() {
    const root = document.getElementById("playground");
    if (!root) return;
    const WMAX = 5;
    const st = { G: 4, pA: 0.5, pB: 0.05 };
    const q = (p, G) => 1 - Math.pow(p, G) - Math.pow(1 - p, G);
    const $ = (id) => document.getElementById(id);
    const segG = $("pg-G");
    [4, 8, 16, 32].forEach((g) => {
      const b = document.createElement("button");
      b.textContent = "G = " + g;
      b.dataset.g = g;
      b.addEventListener("click", () => { st.G = g; render(); });
      segG.appendChild(b);
    });
    $("pg-pA").addEventListener("input", (e) => { st.pA = +e.target.value; render(); });
    $("pg-pB").addEventListener("input", (e) => { st.pB = +e.target.value; render(); });
    root.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
      const [G, pA, pB] = b.dataset.preset.split(",").map(Number);
      Object.assign(st, { G, pA, pB });
      $("pg-pA").value = pA; $("pg-pB").value = pB;
      render();
    }));

    const W = 640, H = 300;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, $("pg-plot"));

    function render() {
      segG.querySelectorAll("button").forEach((b) => b.classList.toggle("on", +b.dataset.g === st.G));
      $("pg-pA-v").textContent = st.pA.toFixed(2);
      $("pg-pB-v").textContent = st.pB.toFixed(2);
      const piA = q(st.pA, st.G), piB = q(st.pB, st.G);
      const ref = Math.max(piA, piB);
      const w = (pi) => (pi > 0 ? Math.min(WMAX, Math.sqrt(ref / pi)) : 1);
      const wA = w(piA), wB = w(piB);
      const gA = piA, gB = piB, dA = wA * wA * piA, dB = wB * wB * piB;

      svg.innerHTML = "";
      // left: q_G(p) curve
      const L = { x: 46, y: 26, w: 250, h: 220 };
      el("text", { x: L.x, y: 14, class: "ttl", style: "font-size:13px;font-weight:700" }, svg).textContent = `Active probability 1 − pᴳ − (1 − p)ᴳ,  G = ${st.G}`;
      const sx = (p) => L.x + p * L.w, sy = (v) => L.y + L.h - v * L.h;
      [0, .25, .5, .75, 1].forEach((t) => {
        el("line", { x1: L.x, x2: L.x + L.w, y1: sy(t), y2: sy(t), stroke: "#eef0f4" }, svg);
        el("text", { x: L.x - 6, y: sy(t) + 4, "text-anchor": "end", style: "font-size:11px;fill:#6b7280" }, svg).textContent = t;
        el("text", { x: sx(t), y: L.y + L.h + 15, "text-anchor": "middle", style: "font-size:11px;fill:#6b7280" }, svg).textContent = t;
      });
      el("text", { x: L.x + L.w / 2, y: L.y + L.h + 32, "text-anchor": "middle", style: "font-size:11.5px;fill:#3d4350" }, svg).textContent = "per-rollout success rate p";
      [4, 8, 16, 32].forEach((g) => {
        let d = "";
        for (let i = 0; i <= 100; i++) { const p = i / 100; d += (i ? "L" : "M") + sx(p) + "," + sy(q(p, g)); }
        el("path", { d, fill: "none", stroke: g === st.G ? "#16181d" : "#d5d8df", "stroke-width": g === st.G ? 2.4 : 1.2 }, svg);
      });
      [[st.pA, piA, "#2e9e44", "A"], [st.pB, piB, "#d1453b", "B"]].forEach(([p, v, c, n]) => {
        el("line", { x1: sx(p), x2: sx(p), y1: sy(0), y2: sy(v), stroke: c, "stroke-dasharray": "3 3" }, svg);
        el("circle", { cx: sx(p), cy: sy(v), r: 6, fill: c, stroke: "#fff", "stroke-width": 2 }, svg);
        el("text", { x: sx(p) + 8, y: sy(v) - 8, style: `font-size:12px;font-weight:700;fill:${c}` }, svg).textContent = n;
      });

      // right: energy bars
      const R = { x: 360, y: 26, w: 270, h: 220 };
      el("text", { x: R.x, y: 14, style: "font-size:13px;font-weight:700;fill:#16181d" }, svg).textContent = "Advantage energy  E / B(G−1)";
      const vmax = Math.max(1, dA, dB, gA, gB);
      const by = (v) => R.y + R.h - (v / vmax) * R.h;
      [0, .5, 1].forEach((t) => {
        el("line", { x1: R.x, x2: R.x + R.w, y1: by(t), y2: by(t), stroke: "#eef0f4" }, svg);
        el("text", { x: R.x - 6, y: by(t) + 4, "text-anchor": "end", style: "font-size:11px;fill:#6b7280" }, svg).textContent = t;
      });
      const bw = 44;
      const bars = [
        [R.x + 20, gA, "#2e9e44", "A"], [R.x + 20 + bw + 6, gB, "#d1453b", "B"],
        [R.x + 160, dA, "#2e9e44", "A"], [R.x + 160 + bw + 6, dB, "#d1453b", "B"],
      ];
      bars.forEach(([x, v, c, n], i) => {
        el("rect", { x, y: by(v), width: bw, height: Math.max(0, R.y + R.h - by(v)), fill: c, opacity: i < 2 ? 0.55 : 0.95, rx: 4 }, svg);
        el("text", { x: x + bw / 2, y: by(v) - 5, "text-anchor": "middle", style: "font-size:11.5px;font-weight:600;fill:#3d4350" }, svg).textContent = v.toFixed(2);
      });
      el("text", { x: R.x + 20 + bw + 3, y: R.y + R.h + 16, "text-anchor": "middle", style: "font-size:12px;font-weight:700;fill:#3d4350" }, svg).textContent = "GDPO";
      el("text", { x: R.x + 160 + bw + 3, y: R.y + R.h + 16, "text-anchor": "middle", style: "font-size:12px;font-weight:700;fill:#3d4350" }, svg).textContent = "DARA-Sym";
      el("text", { x: R.x + 20 + bw + 3, y: R.y + R.h + 32, "text-anchor": "middle", style: "font-size:11.5px;fill:#6b7280" }, svg).textContent = "= π";
      el("text", { x: R.x + 160 + bw + 3, y: R.y + R.h + 32, "text-anchor": "middle", style: "font-size:11.5px;fill:#6b7280" }, svg).textContent = "= w² π";

      $("pg-gdpo").textContent = piA > 0 ? (piB / piA).toFixed(2) : "–";
      $("pg-dara").textContent = dA > 0 ? (dB / dA).toFixed(2) : "–";
      $("pg-w").textContent = `w_A = ${wA.toFixed(2)},  w_B = ${wB.toFixed(2)}`;
      const capped = Math.sqrt(ref / Math.max(1e-12, Math.min(piA, piB))) > WMAX && Math.min(piA, piB) > 0;
      $("pg-cap").style.display = capped ? "" : "none";
    }
    render();
  }

  /* ---------- small tables with tabs ---------- */
  function tabbed() {
    document.querySelectorAll("[data-tabgroup]").forEach((g) => {
      const name = g.dataset.tabgroup;
      const panes = document.querySelectorAll(`[data-pane="${name}"]`);
      g.querySelectorAll("button").forEach((b, i) => {
        b.addEventListener("click", () => {
          g.querySelectorAll("button").forEach((x) => x.classList.remove("on"));
          b.classList.add("on");
          panes.forEach((p, j) => (p.style.display = j === i ? "" : "none"));
        });
      });
    });
  }

  function copyBib() {
    const b = document.getElementById("copy-bib");
    if (!b) return;
    b.addEventListener("click", () => {
      navigator.clipboard.writeText(document.getElementById("bibtex").innerText).then(() => {
        b.textContent = "Copied!";
        setTimeout(() => (b.textContent = "Copy"), 1500);
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    fig1b(); bfcl(); energy(); playground(); tabbed(); copyBib();
    if (window.renderMathInElement) {
      renderMathInElement(document.body, {
        delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
        throwOnError: false,
      });
    }
  });
})();
