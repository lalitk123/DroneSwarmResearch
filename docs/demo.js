/* In-browser drone-swarm SAR demo — a faithful lightweight port of the Python
   simulator: grid world, sensing drones, three coordination strategies, and a
   pluggable comms-failure model with multi-hop connectivity. Pure client-side. */
(function () {
  "use strict";
  const W = 48, H = 48, CELL = 11;            // grid + pixel size
  const cvs = document.getElementById("simCanvas");
  if (!cvs) return;
  cvs.width = W * CELL; cvs.height = H * CELL;
  const ctx = cvs.getContext("2d");

  const $ = (id) => document.getElementById(id);
  const ui = {
    strategy: $("d_strategy"), failure: $("d_failure"), sev: $("d_sev"),
    sevVal: $("d_sevVal"), drones: $("d_drones"), dronesVal: $("d_dronesVal"),
    run: $("d_run"), reset: $("d_reset"),
    step: $("m_step"), cov: $("m_cov"), vic: $("m_vic"), conn: $("m_conn"),
    sevLabel: $("d_sevLabel"),
  };

  let grid, covered, victims, drones, base, pher, running = false, step = 0;
  let rafId = null, acc = 0, lastT = 0;
  const STEP_MS = 90;                          // sim tick period
  let rng = mulberry32(12345);

  function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
  const idx = (x, y) => y * W + x;
  const inB = (x, y) => x >= 0 && y >= 0 && x < W && y < H;

  function build() {
    step = 0; acc = 0;
    grid = new Uint8Array(W * H);
    covered = new Uint8Array(W * H);
    pher = new Float32Array(W * H);
    // obstacle clusters via short random walks
    const clusters = 9;
    for (let c = 0; c < clusters; c++) {
      let cx = 4 + Math.floor(rng() * (W - 8)), cy = 4 + Math.floor(rng() * (H - 8));
      for (let k = 0; k < 26; k++) {
        const x = Math.min(W - 1, Math.max(0, cx + ((rng() * 5) | 0) - 2));
        const y = Math.min(H - 1, Math.max(0, cy + ((rng() * 5) | 0) - 2));
        if (x < 3 && y < 3) continue;          // keep base corner clear
        grid[idx(x, y)] = 1;
      }
    }
    base = { id: -1, x: 1, y: 1, alive: true };
    grid[idx(1, 1)] = 0;
    // victims on free cells
    victims = [];
    while (victims.length < 12) {
      const x = (rng() * W) | 0, y = (rng() * H) | 0;
      if (!grid[idx(x, y)]) victims.push({ x, y, found: false });
    }
    // drones near base
    const n = +ui.drones.value;
    drones = [];
    for (let i = 0; i < n; i++) {
      let x = 1 + ((rng() * 4) | 0), y = 1 + ((rng() * 4) | 0), tries = 0;
      while (grid[idx(x, y)] && tries++ < 50) { x = (rng() * 6) | 0; y = (rng() * 6) | 0; }
      drones.push({ id: i, x, y, tx: null, ty: null, alive: true, map: new Uint8Array(W * H) });
    }
    draw();
    updateMetrics();
  }

  // ---- comms model -------------------------------------------------------
  function severity() { return +ui.sev.value / 100; }
  function deadZone() {                          // central rectangle
    const s = severity(); const side = Math.round(Math.sqrt(s * W * H));
    const h = (side / 2) | 0; const cx = W >> 1, cy = H >> 1;
    return [cx - h, cy - h, cx + h, cy + h];
  }
  function linkOK(a, b) {
    const f = ui.failure.value;
    const dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy);
    if (f === "perfect") return true;
    if (f === "packet") return rng() > severity();            // per-tick drop
    if (f === "range") { const R = Math.max(2, 30 * (1 - severity())); return d <= R; }
    if (f === "deadzone") {
      const [x0, y0, x1, y1] = deadZone();
      const inz = (p) => p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1;
      return !(inz(a) || inz(b));
    }
    return true;
  }
  // connected components over base+alive drones; returns comp id array & adjacency
  function components() {
    const nodes = [base, ...drones.filter(d => d.alive)];
    const N = nodes.length, comp = new Int32Array(N).fill(-1), adj = Array.from({length:N},()=>[]);
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++)
      if (linkOK(nodes[i], nodes[j])) { adj[i].push(j); adj[j].push(i); }
    let c = 0;
    for (let s = 0; s < N; s++) if (comp[s] < 0) {
      const st = [s]; comp[s] = c;
      while (st.length) { const u = st.pop(); for (const v of adj[u]) if (comp[v] < 0) { comp[v] = c; st.push(v); } }
      c++;
    }
    return { nodes, comp, adj };
  }

  // ---- sensing -----------------------------------------------------------
  const R_SENSE = 3;
  function sense(d) {
    for (let yy = Math.max(0, d.y - R_SENSE); yy <= Math.min(H - 1, d.y + R_SENSE); yy++)
      for (let xx = Math.max(0, d.x - R_SENSE); xx <= Math.min(W - 1, d.x + R_SENSE); xx++) {
        if ((xx - d.x) ** 2 + (yy - d.y) ** 2 > R_SENSE * R_SENSE) continue;
        if (grid[idx(xx, yy)]) continue;
        covered[idx(xx, yy)] = 1; d.map[idx(xx, yy)] = 1;
      }
    for (const v of victims) if (!v.found && (v.x - d.x) ** 2 + (v.y - d.y) ** 2 <= R_SENSE * R_SENSE) v.found = true;
  }

  // nearest frontier in a given coverage map (free, uncovered, adjacent to covered;
  // fallback: nearest uncovered free cell)
  function nearestFrontier(map, x, y) {
    let best = null, bestD = 1e9, fb = null, fbD = 1e9;
    for (let yy = 0; yy < H; yy++) for (let xx = 0; xx < W; xx++) {
      const i = idx(xx, yy);
      if (grid[i] || map[i]) continue;
      const dd = Math.abs(xx - x) + Math.abs(yy - y);
      if (dd < fbD) { fbD = dd; fb = { x: xx, y: yy }; }
      let adjc = false;
      if (xx > 0 && map[i - 1]) adjc = true;
      else if (xx < W - 1 && map[i + 1]) adjc = true;
      else if (yy > 0 && map[i - W]) adjc = true;
      else if (yy < H - 1 && map[i + W]) adjc = true;
      if (adjc && dd < bestD) { bestD = dd; best = { x: xx, y: yy }; }
    }
    return best || fb;
  }

  function moveToward(d) {
    if (d.tx == null) return;
    const sx = Math.sign(d.tx - d.x), sy = Math.sign(d.ty - d.y);
    const cand = [[d.x + sx, d.y + sy], [d.x + sx, d.y], [d.x, d.y + sy]];
    for (const [nx, ny] of cand) if (inB(nx, ny) && !grid[idx(nx, ny)]) { d.x = nx; d.y = ny; return; }
  }

  // ---- one simulation step ----------------------------------------------
  function tick() {
    step++;
    drones.forEach(sense);
    const strat = ui.strategy.value;
    const { nodes, comp } = components();
    const compOf = new Map(); nodes.forEach((nd, i) => compOf.set(nd.id, comp[i]));

    if (strat === "centralized") {
      const baseComp = compOf.get(-1);
      // base aggregates maps of connected drones into one master map
      const master = new Uint8Array(W * H);
      const conn = drones.filter(d => d.alive && compOf.get(d.id) === baseComp);
      for (const d of conn) { const m = d.map; for (let i = 0; i < master.length; i++) if (m[i]) master[i] = 1; }
      const claimed = [];
      for (const d of conn) {
        let t = nearestFrontier(master, d.x, d.y);
        // soft de-dup: nudge if a near-identical target already claimed
        if (t && claimed.some(c => Math.abs(c.x - t.x) + Math.abs(c.y - t.y) < 2)) {
          const t2 = nearestFrontier(master, d.x + ((rng()*6)|0)-3, d.y + ((rng()*6)|0)-3);
          if (t2) t = t2;
        }
        if (t) { d.tx = t.x; d.ty = t.y; claimed.push(t); }
      }
      // disconnected drones keep their last target (the failure mode)
    } else if (strat === "decentralized") {
      // pairwise gossip merge among linked neighbors (snapshot, one hop/step)
      const snap = drones.map(d => d.alive ? d.map.slice() : null);
      for (let i = 0; i < drones.length; i++) for (let j = 0; j < drones.length; j++) {
        if (i === j) continue; const a = drones[i], b = drones[j];
        if (!a.alive || !b.alive) continue;
        if (compOf.get(a.id) === compOf.get(b.id) && linkOK(a, b)) {
          const sj = snap[j], ma = a.map; for (let k = 0; k < ma.length; k++) if (sj[k]) ma[k] = 1;
        }
      }
      for (const d of drones) if (d.alive) { const t = nearestFrontier(d.map, d.x, d.y); if (t) { d.tx = t.x; d.ty = t.y; } }
    } else { // stigmergy
      for (let i = 0; i < pher.length; i++) pher[i] *= 0.97;          // evaporate
      for (const d of drones) if (d.alive) pher[idx(d.x, d.y)] += 5;  // deposit
      for (const d of drones) if (d.alive) {
        let best = null, bv = 1e18;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue; const nx = d.x + dx, ny = d.y + dy;
          if (!inB(nx, ny) || grid[idx(nx, ny)]) continue;
          const v = pher[idx(nx, ny)] + rng() * 0.5;
          if (v < bv) { bv = v; best = { x: nx, y: ny }; }
        }
        if (best) { d.tx = best.x; d.ty = best.y; }
      }
    }
    drones.forEach(moveToward);
    updateMetrics();
    if (victims.every(v => v.found)) { running = false; ui.run.textContent = "▶ Run"; flashComplete(); }
  }

  // ---- metrics + drawing -------------------------------------------------
  let nFree = 0;
  function updateMetrics() {
    if (!nFree) for (let i = 0; i < grid.length; i++) if (!grid[i]) nFree++;
    let cov = 0; for (let i = 0; i < covered.length; i++) if (covered[i]) cov++;
    const found = victims.filter(v => v.found).length;
    const { comp } = components();
    const counts = {}; let big = 0;
    comp.forEach(c => { counts[c] = (counts[c] || 0) + 1; big = Math.max(big, counts[c]); });
    ui.step.textContent = step;
    ui.cov.textContent = (100 * cov / nFree).toFixed(0) + "%";
    ui.vic.textContent = found + "/" + victims.length;
    ui.conn.textContent = (100 * big / comp.length).toFixed(0) + "%";
    draw();
  }

  function draw() {
    ctx.fillStyle = "#0c1326"; ctx.fillRect(0, 0, cvs.width, cvs.height);
    const showPher = ui.strategy.value === "stigmergy";
    let pmax = 1; if (showPher) for (let i = 0; i < pher.length; i++) if (pher[i] > pmax) pmax = pher[i];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = idx(x, y);
      if (grid[i]) { ctx.fillStyle = "#2a3358"; ctx.fillRect(x*CELL, y*CELL, CELL, CELL); continue; }
      if (showPher && pher[i] > 0.05) {
        const t = Math.min(1, pher[i] / pmax);
        ctx.fillStyle = `rgba(255,${(180-120*t)|0},${(90-60*t)|0},${0.15+0.5*t})`;
        ctx.fillRect(x*CELL, y*CELL, CELL, CELL);
      } else if (covered[i]) { ctx.fillStyle = "rgba(92,200,255,0.13)"; ctx.fillRect(x*CELL, y*CELL, CELL, CELL); }
    }
    // dead zone overlay
    if (ui.failure.value === "deadzone") {
      const [x0, y0, x1, y1] = deadZone();
      ctx.fillStyle = "rgba(255,80,80,0.12)";
      ctx.fillRect(x0*CELL, y0*CELL, (x1-x0+1)*CELL, (y1-y0+1)*CELL);
      ctx.strokeStyle = "rgba(255,80,80,0.5)"; ctx.setLineDash([5,4]);
      ctx.strokeRect(x0*CELL, y0*CELL, (x1-x0+1)*CELL, (y1-y0+1)*CELL); ctx.setLineDash([]);
    }
    // comm links
    const { nodes, adj } = components();
    ctx.strokeStyle = "rgba(124,140,255,0.22)"; ctx.lineWidth = 1;
    for (let i = 0; i < nodes.length; i++) for (const j of adj[i]) if (j > i) {
      const a = nodes[i], b = nodes[j];
      ctx.beginPath(); ctx.moveTo(a.x*CELL+CELL/2, a.y*CELL+CELL/2); ctx.lineTo(b.x*CELL+CELL/2, b.y*CELL+CELL/2); ctx.stroke();
    }
    // victims
    for (const v of victims) {
      ctx.fillStyle = v.found ? "#46d39a" : "#ffce5c";
      ctx.beginPath(); ctx.arc(v.x*CELL+CELL/2, v.y*CELL+CELL/2, CELL*0.34, 0, 7); ctx.fill();
    }
    // base
    ctx.fillStyle = "#ffffff"; ctx.fillRect(base.x*CELL-1, base.y*CELL-1, CELL+2, CELL+2);
    // drones
    for (const d of drones) { if (!d.alive) continue;
      ctx.fillStyle = "#5cc8ff";
      ctx.beginPath(); ctx.arc(d.x*CELL+CELL/2, d.y*CELL+CELL/2, CELL*0.4, 0, 7); ctx.fill();
    }
  }

  let flash = 0;
  function flashComplete() { flash = 1; const t0 = performance.now();
    (function f(t){ const a = Math.max(0, 1 - (t - t0)/1400); ctx.fillStyle=`rgba(70,211,154,${a*0.25})`;
      ctx.fillRect(0,0,cvs.width,cvs.height); if (a>0) requestAnimationFrame(f); else draw(); })(t0);
  }

  function loop(t) {
    if (!running) { rafId = null; return; }
    if (!lastT) lastT = t; acc += t - lastT; lastT = t;
    while (acc >= STEP_MS) { tick(); acc -= STEP_MS; if (!running) break; }
    rafId = requestAnimationFrame(loop);
  }

  // ---- controls ----------------------------------------------------------
  function syncLabels() {
    ui.dronesVal.textContent = ui.drones.value;
    const f = ui.failure.value;
    ui.sevLabel.textContent = f === "perfect" ? "Severity (n/a)" :
      f === "packet" ? "Packet-loss rate" : f === "range" ? "Range loss" : "Dead-zone size";
    ui.sev.disabled = (f === "perfect");
    ui.sevVal.textContent = f === "perfect" ? "—" : ui.sev.value + "%";
  }
  ui.run.addEventListener("click", () => {
    running = !running; ui.run.textContent = running ? "❚❚ Pause" : "▶ Run";
    if (running) { lastT = 0; rafId = requestAnimationFrame(loop); }
  });
  ui.reset.addEventListener("click", () => { running = false; ui.run.textContent = "▶ Run";
    rng = mulberry32((Math.random()*1e9)|0); build(); });
  ui.drones.addEventListener("input", () => { syncLabels(); build(); });
  ui.sev.addEventListener("input", syncLabels);
  [ui.strategy, ui.failure].forEach(el => el.addEventListener("change", () => { syncLabels(); draw(); updateMetrics(); }));
  ui.failure.addEventListener("change", () => { drones && drones.forEach(d => {}); });

  syncLabels(); build();
})();
