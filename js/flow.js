/* flow.js — data-flow diagrams for the arithmetic panel.
 *
 * The tables in the arithmetic panel list every number, but not where it comes
 * from. Each function here draws the cell of the selected node as a block
 * diagram — gate boxes, multiply/add nodes, wires with arrows — and writes the
 * value that flows along every wire at the current step. Hovering a block shows
 * its formula with the live numbers filled in.
 *
 * Everything is a plain SVG string so the panel can keep rebuilding its
 * innerHTML a few times a second.
 */

const Flow = (() => {
  const BLUE = '#2b6cb0', INK = '#33465a', MUTED = '#7b8794';
  const POS = '#c2760f', NEG = '#0877bd';
  const OPFILL = '#dbe7fb';

  /* ------------------------------------------------------------ text */
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const num = (v, d = 3) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
  const signColor = (v) => (Math.abs(v) < 5e-4 ? MUTED : v > 0 ? POS : NEG);

  /** "h_{t−1}" → h with a subscript; plain text otherwise. */
  function rich(str) {
    const parts = String(str).split(/(_\{[^}]*\}|_[A-Za-z0-9])/);
    let out = '', low = false;
    for (const p of parts) {
      if (!p) continue;
      if (p[0] === '_' && p.length > 1) {
        const sub = p[1] === '{' ? p.slice(2, -1) : p.slice(1);
        out += '<tspan dy="3.5" font-size="0.72em">' + esc(sub) + '</tspan>';
        low = true;
      } else {
        // the text after a subscript has to climb back to the baseline itself;
        // an empty tspan carrying the dy is ignored by the browser
        out += low ? '<tspan dy="-3.5">' + esc(p) + '</tspan>' : esc(p);
        low = false;
      }
    }
    return out;
  }

  function txt(x, y, s, o = {}) {
    return '<text x="' + x + '" y="' + y + '" text-anchor="' + (o.anchor || 'middle') + '"' +
      ' font-size="' + (o.size || 12) + '" fill="' + (o.color || INK) + '"' +
      (o.weight ? ' font-weight="' + o.weight + '"' : '') +
      (o.mono ? ' font-family="ui-monospace,Consolas,monospace"' : '') +
      (o.italic ? ' font-style="italic"' : '') + '>' + rich(s) + '</text>';
  }

  const tip = (t) => (t ? '<title>' + esc(t) + '</title>' : '');

  /* ----------------------------------------------------------- wires */
  /** A polyline with rounded corners and an arrow at the end. */
  function wire(pts, o = {}) {
    const r = 8;
    let d = 'M' + pts[0][0] + ' ' + pts[0][1];
    for (let i = 1; i < pts.length; i++) {
      const [x, y] = pts[i];
      if (i < pts.length - 1) {
        const [px, py] = pts[i - 1], [nx, ny] = pts[i + 1];
        const l1 = Math.hypot(x - px, y - py) || 1, l2 = Math.hypot(nx - x, ny - y) || 1;
        const k1 = Math.min(r, l1 / 2), k2 = Math.min(r, l2 / 2);
        d += ' L' + (x - (x - px) / l1 * k1) + ' ' + (y - (y - py) / l1 * k1) +
             ' Q' + x + ' ' + y + ' ' + (x + (nx - x) / l2 * k2) + ' ' + (y + (ny - y) / l2 * k2);
      } else {
        d += ' L' + x + ' ' + y;
      }
    }
    const dash = o.dash ? ' stroke-dasharray="4 3"' : '';
    const arrow = o.arrow === false ? '' : ' marker-end="url(#flowArrow)"';
    // a white halo underneath lets a wire visibly pass over another one
    const halo = o.over
      ? '<path d="' + d + '" fill="none" stroke="#fff" stroke-width="7"/>' : '';
    return halo + '<path d="' + d + '" fill="none" stroke="' + (o.color || INK) + '"' +
      ' stroke-width="' + (o.width || 1.3) + '"' + dash + arrow + '/>';
  }

  /** A value written on a wire: a white pill, coloured by sign. */
  function pill(x, y, label, v, o = {}) {
    const s = (label ? label + ' = ' : '') + (typeof v === 'number' ? num(v, o.d || 3) : v);
    const plain = s.replace(/_\{([^}]*)\}/g, '$1').replace(/_([A-Za-z0-9])/g, '$1');
    const w = plain.length * 6.3 + 12;
    const col = typeof v === 'number' ? signColor(v) : MUTED;
    return '<g class="blk">' + tip(o.tip) +
      '<rect x="' + (x - w / 2) + '" y="' + (y - 9) + '" width="' + w + '" height="18" rx="9"' +
      ' fill="#fff" stroke="#c9d6e6"/>' +
      txt(x, y + 4, s, { size: 11, mono: true, color: col, weight: 600 }) + '</g>';
  }

  /* ---------------------------------------------------------- blocks */
  /**
   * A gate box (σ, tanh, ReLU …) with its output value inside and a thin bar
   * showing how far open it is: 0…1 for σ, −1…1 around the centre for tanh.
   */
  function gate(x, y, sym, v, o = {}) {
    const w = o.w || 70, h = 44;
    let bar = '';
    if (typeof v === 'number' && o.range) {
      const bw = w - 10, bx = x - bw / 2, by = y + h / 2 - 7;
      if (o.range === 'unit') {
        bar = '<rect x="' + bx + '" y="' + by + '" width="' + (bw * Math.max(0, Math.min(1, v))) +
          '" height="3" rx="1.5" fill="#cfe0f7"/>';
      } else {
        const half = bw / 2, len = half * Math.max(-1, Math.min(1, v));
        bar = '<rect x="' + (len >= 0 ? x : x + len) + '" y="' + by + '" width="' + Math.abs(len) +
          '" height="3" rx="1.5" fill="#cfe0f7"/>' +
          '<rect x="' + (x - 0.5) + '" y="' + (by - 1) + '" width="1" height="5" fill="#9fbbe0"/>';
      }
      bar = '<rect x="' + bx + '" y="' + by + '" width="' + bw + '" height="3" rx="1.5" fill="#1f4f86"/>' + bar;
    }
    return '<g class="blk">' + tip(o.tip) +
      '<rect x="' + (x - w / 2) + '" y="' + (y - h / 2) + '" width="' + w + '" height="' + h + '" rx="4"' +
      ' fill="' + (o.fill || BLUE) + '"/>' +
      txt(x, y - 4, sym, { size: 13, color: '#fff', weight: 600 }) +
      (typeof v === 'number' || typeof v === 'string'
        ? txt(x, y + 11, typeof v === 'number' ? num(v) : v, { size: 10.5, color: '#e6efff', mono: true })
        : '') +
      bar + '</g>';
  }

  /** A light box for a fixed weight matrix or a buffer (W, Ā, D, U_n …). */
  function wbox(x, y, s, o = {}) {
    const w = o.w || 56, h = o.h || 28;
    return '<g class="blk">' + tip(o.tip) +
      '<rect x="' + (x - w / 2) + '" y="' + (y - h / 2) + '" width="' + w + '" height="' + h + '" rx="5"' +
      ' fill="#fff" stroke="' + BLUE + '" stroke-width="1.2"/>' +
      txt(x, y + 4.5, s, { size: 12, color: BLUE, weight: 600 }) +
      (o.sub ? txt(x, y + h / 2 + 12, o.sub, { size: 10, color: MUTED, mono: true }) : '') + '</g>';
  }

  /** A pointwise operation node: ⊙ (multiply), + (add), 1− … */
  function op(x, y, sym, o = {}) {
    const r = 14;
    let inner;
    if (sym === 'mul') {
      inner = '<circle cx="' + x + '" cy="' + y + '" r="7" fill="none" stroke="#1d2b38" stroke-width="1.3"/>' +
              '<circle cx="' + x + '" cy="' + y + '" r="3" fill="#1d2b38"/>';
    } else {
      inner = txt(x, y + 5, sym, { size: sym.length > 1 ? 11 : 16, color: '#1d2b38', weight: 600 });
    }
    return '<g class="blk">' + tip(o.tip) +
      '<circle cx="' + x + '" cy="' + y + '" r="' + r + '" fill="' + OPFILL + '" stroke="#1d2b38" stroke-width="1"/>' +
      inner + '</g>';
  }

  /** The tanh squashing an LSTM applies to its cell state — an ellipse, as usual. */
  function ell(x, y, s, o = {}) {
    return '<g class="blk">' + tip(o.tip) +
      '<ellipse cx="' + x + '" cy="' + y + '" rx="32" ry="15" fill="' + OPFILL + '" stroke="#1d2b38" stroke-width="1"/>' +
      txt(x, y + 4.5, s, { size: 12, color: '#1d2b38', weight: 600 }) + '</g>';
  }

  /** Grey caption lines, e.g. "forget" / "gate", anchored at (x, y). */
  function cap(x, y, lines, anchor) {
    return lines.map((l, i) => txt(x, y + i * 13, l, { size: 11, color: MUTED, anchor: anchor || 'end' })).join('');
  }

  function cellFrame(x0, y0, x1, y1) {
    return '<rect x="' + x0 + '" y="' + y0 + '" width="' + (x1 - x0) + '" height="' + (y1 - y0) + '"' +
      ' rx="10" fill="#fff" stroke="#1d2b38" stroke-width="1"/>';
  }

  function svg(w, h, body, caption) {
    // one viewBox unit ≈ one pixel on a wide screen, so every diagram has the same scale
    return '<div class="flow"><svg viewBox="0 0 ' + w + ' ' + h + '" role="img" ' +
      'style="max-width:' + Math.round(w * 1.05) + 'px" ' +
      'font-family="system-ui,Segoe UI,Roboto,Arial,sans-serif">' +
      '<defs><marker id="flowArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" ' +
      'markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="' + INK + '"/></marker></defs>' +
      body + '</svg>' +
      '<div class="flowcap">' + caption + ' Hover any block for its formula with the current numbers.</div></div>';
  }

  /* ------------------------------------------------------ recurrent */
  /** Wx·x, Wh·h and the bias behind gate g, exactly as the table computes them. */
  function gateSums(d, g) {
    let ix = 0, ih = 0;
    for (let i = 0; i < d.D; i++) ix += d.wx[g][i] * d.xv[i];
    for (let v = 0; v < d.H; v++) ih += d.wh[g][v] * d.hprev[v];
    const rec = d.kind === 'gru' && g === 2 ? d.gates[1] * d.q : ih;
    return { ix, rec, b: d.bias[g], z: ix + rec + d.bias[g] };
  }

  function gateTip(d, g, name, fn, extra) {
    const s = gateSums(d, g);
    return name + ' = ' + fn + '( Wx·x + ' + (extra || 'Wh·h') + ' + b )\n' +
      '  = ' + fn + '( ' + num(s.ix, 4) + ' + ' + num(s.rec, 4) + ' + ' + num(s.b) + ' )\n' +
      '  = ' + fn + '( ' + num(s.z, 4) + ' ) = ' + num(d.gates[g], 4);
  }

  /** Labels that depend on the direction a bidirectional unit reads the window. */
  function stepNames(d) {
    const prev = d.back ? 't+1' : 't−1';
    const xin = d.D === 1 ? num(d.xv[0]) : d.D + ' inputs';
    const xTip = 'x_t — what this layer reads at step ' + d.t + ':\n' +
      Array.from(d.xv).map((v, i) => '  x[' + i + '] = ' + num(v, 4)).join('\n');
    const hTip = 'h_' + prev + ' — the whole previous state (' + d.H + ' units) feeds every gate:\n' +
      Array.from(d.hprev).map((v, i) => '  h[' + i + '] = ' + num(v, 4) + (i === d.u ? '   ← this unit' : '')).join('\n');
    return { prev, xin, xTip, hTip };
  }

  function lstm(d) {
    const { prev, xin, xTip, hTip } = stepNames(d);
    const [iv, fv, ov, gv] = d.gates;
    const fc = fv * d.cprev, ig = iv * gv, tc = Math.tanh(d.c);
    const Y = 70, B = 262;                        // memory line, input line
    let s = cellFrame(150, 34, 790, 290);

    // memory line: c_{t−1} → ⊙f → + → c_t
    s += wire([[40, Y], [246, Y]]);
    s += wire([[274, Y], [526, Y]]);
    s += wire([[554, Y], [870, Y]]);
    // input line: h_{t−1} and x_t feed every gate
    s += wire([[40, B], [680, B]], { arrow: false });
    s += wire([[200, 318], [200, B]], { arrow: false });
    for (const gx of [260, 400, 540, 680]) s += wire([[gx, B], [gx, 228]]);

    // gates
    s += cap(221, 201, ['forget', 'gate']);
    s += cap(361, 201, ['input', 'gate']);
    s += cap(501, 201, ['candidate', 'C̃']);
    s += cap(641, 201, ['output', 'gate']);
    s += gate(260, 206, 'σ  f', fv, { range: 'unit', tip: gateTip(d, 1, 'f', 'σ') + '\n\nHow much of the old cell state survives.' });
    s += gate(400, 206, 'σ  i', iv, { range: 'unit', tip: gateTip(d, 0, 'i', 'σ') + '\n\nHow much of the candidate gets written.' });
    s += gate(540, 206, 'tanh  g', gv, { range: 'sym', tip: gateTip(d, 3, 'g', 'tanh') + '\n\nThe candidate value to write (C̃).' });
    s += gate(680, 206, 'σ  o', ov, { range: 'unit', tip: gateTip(d, 2, 'o', 'σ') + '\n\nHow much of the squashed cell becomes the output.' });

    // f ⊙ c_{t−1}
    s += wire([[260, 184], [260, Y + 14]]);
    s += op(260, Y, 'mul', { tip: 'f · c_' + prev + ' = ' + num(fv) + ' · ' + num(d.cprev) + ' = ' + num(fc, 4) });
    // i ⊙ g
    s += wire([[400, 184], [400, 150], [526, 150]]);
    s += wire([[540, 184], [540, 164]]);
    s += op(540, 150, 'mul', { tip: 'i · g = ' + num(iv) + ' · ' + num(gv) + ' = ' + num(ig, 4) });
    s += wire([[540, 136], [540, Y + 14]]);
    s += op(540, Y, '+', { tip: 'c_t = f·c_' + prev + ' + i·g = ' + num(fc, 4) + ' + ' + num(ig, 4) + ' = ' + num(d.c, 4) });
    // tanh(c_t) ⊙ o → h_t
    s += wire([[680, Y], [680, 97]], { arrow: true });
    s += ell(680, 112, 'tanh', { tip: 'tanh(c_t) = tanh(' + num(d.c, 4) + ') = ' + num(tc, 4) +
      (Math.abs(d.c) > 2.5 ? '\nThe cell is saturating here — tanh′ ≈ 0.' : '') });
    s += wire([[680, 127], [680, 146]]);
    s += wire([[680, 184], [680, 174]]);
    s += op(680, 160, 'mul', { tip: 'h_t = o · tanh(c_t) = ' + num(ov) + ' · ' + num(tc) + ' = ' + num(d.h, 4) });
    s += wire([[694, 160], [740, 160], [740, B], [870, B]]);

    // values on the wires
    s += pill(95, Y, 'c_{' + prev + '}', d.cprev, { tip: 'Cell state from the previous step.' });
    s += pill(95, B, 'h_{' + prev + '}', d.hprev[d.u], { tip: hTip });
    s += pill(200, 309, 'x_t', xin, { tip: xTip });
    s += pill(400, Y, 'f·c', fc);
    s += pill(470, 150, '', ig, { tip: 'i · g' });
    s += pill(610, Y, '', d.c, { tip: 'the new cell state c_t' });
    s += pill(740, 112, '', tc, { tip: 'tanh(c_t)' });
    s += pill(800, B, '', d.h, { tip: 'the new hidden state h_t — this unit\'s map at t = ' + d.t });
    s += txt(880, Y + 5, 'c_t', { size: 16, weight: 600, anchor: 'start' });
    s += txt(880, B + 5, 'h_t', { size: 16, weight: 600, anchor: 'start' });
    s += txt(40, Y - 16, 'memory', { size: 11, color: MUTED, anchor: 'start' });
    s += txt(40, B - 16, 'hidden state', { size: 11, color: MUTED, anchor: 'start' });
    s += txt(200, 334, 'input', { size: 11, color: MUTED });
    return svg(940, 344, s, 'LSTM cell of unit ' + (d.u + 1) + ' at step t = ' + d.t + '.');
  }

  function gru(d) {
    const { prev, xin, xTip, hTip } = stepNames(d);
    const [zv, rv, nv] = d.gates;
    const keep = zv * d.hprev[d.u], fresh = (1 - zv) * nv, rq = rv * d.q;
    const Y = 70, B = 262;
    let s = cellFrame(150, 34, 820, 290);

    // state line: h_{t−1} → ⊙(z) → + → h_t
    s += wire([[40, Y], [606, Y]]);
    s += wire([[634, Y], [746, Y]]);
    s += wire([[774, Y], [870, Y]]);
    // the gates read h_{t−1} and x_t
    s += wire([[175, Y], [175, B], [620, B]], { arrow: false });
    s += wire([[200, 318], [200, B]], { arrow: false });
    s += wire([[220, B], [220, 228]]);
    s += wire([[460, B], [460, 228]]);
    s += wire([[620, B], [620, 228]]);

    // reset gate scales the recurrent part of the candidate
    s += cap(220, 172, ['reset gate'], 'middle');
    s += gate(220, 206, 'σ  r', rv, { range: 'unit', tip: gateTip(d, 1, 'r', 'σ') + '\n\nHow much of the old state the candidate may look at.' });
    s += wire([[330, Y], [330, 112]]);
    s += wbox(330, 126, 'U_n', { tip: 'q = Uₙ · h_' + prev + ' = ' + num(d.q, 4) + '\nThe recurrent part of the candidate, before the reset gate.' });
    s += wire([[330, 140], [330, 191]]);
    s += wire([[255, 206], [316, 206]]);
    s += op(330, 206, 'mul', { tip: 'r · q = ' + num(rv) + ' · ' + num(d.q, 4) + ' = ' + num(rq, 4) });
    s += wire([[344, 206], [425, 206]]);
    // candidate
    s += cap(500, 202, ['candidate', 'n'], 'start');
    s += gate(460, 206, 'tanh  n', nv, { range: 'sym', tip: gateTip(d, 2, 'n', 'tanh', 'r·(Uₙ·h)') + '\n\nThe new content the state could take.' });
    // update gate
    s += cap(581, 176, ['update gate']);
    s += gate(620, 206, 'σ  z', zv, { range: 'unit', tip: gateTip(d, 0, 'z', 'σ') + '\n\nHow much of the old state is kept.' });
    s += wire([[620, 184], [620, Y + 14]]);
    s += op(620, Y, 'mul', { tip: 'z · h_' + prev + ' = ' + num(zv) + ' · ' + num(d.hprev[d.u]) + ' = ' + num(keep, 4) });
    s += wire([[655, 206], [686, 206]]);
    s += op(700, 206, '1−', { tip: '1 − z = ' + num(1 - zv) });
    s += wire([[714, 206], [760, 206], [760, 164]]);
    // n → (1−z)·n, passing over the z wire
    s += wire([[460, 184], [460, 150], [746, 150]], { over: true });
    s += op(760, 150, 'mul', { tip: '(1−z) · n = ' + num(1 - zv) + ' · ' + num(nv) + ' = ' + num(fresh, 4) });
    s += wire([[760, 136], [760, Y + 14]]);
    s += op(760, Y, '+', { tip: 'h_t = z·h_' + prev + ' + (1−z)·n = ' + num(keep, 4) + ' + ' + num(fresh, 4) + ' = ' + num(d.h, 4) });

    s += pill(95, Y, 'h_{' + prev + '}', d.hprev[d.u], { tip: hTip });
    s += pill(200, 309, 'x_t', xin, { tip: xTip });
    s += pill(330, 166, 'q', d.q, { tip: 'Uₙ · h_' + prev });
    s += pill(385, 206, '', rq, { tip: 'r · q' });
    s += pill(690, Y, '', keep, { tip: 'z · h_' + prev + ' — the part of the old state that is kept' });
    s += pill(540, 150, '', nv, { tip: 'n — the candidate' });
    s += pill(760, 112, '', fresh, { tip: '(1−z) · n — the part that is new' });
    s += pill(822, Y, '', d.h, { tip: 'the new hidden state h_t — this unit\'s map at t = ' + d.t });
    s += txt(880, Y + 5, 'h_t', { size: 16, weight: 600, anchor: 'start' });
    s += txt(40, Y - 16, 'hidden state', { size: 11, color: MUTED, anchor: 'start' });
    s += txt(200, 334, 'input', { size: 11, color: MUTED });
    return svg(940, 344, s, 'GRU cell of unit ' + (d.u + 1) + ' at step t = ' + d.t + '.');
  }

  function rnn(d) {
    const { prev, xin, xTip, hTip } = stepNames(d);
    const g = gateSums(d, 0);
    const Y = 70, R = 200;                         // state line, sum row
    let s = cellFrame(150, 34, 760, 304);

    // h_{t−1} → W_h → +
    s += wire([[40, Y], [200, Y], [200, R], [248, R]]);
    s += wbox(274, R, 'W_h', { w: 48, tip: 'Wh · h_' + prev + ' = ' + num(g.rec, 4) + '\nThe whole previous state, one weight per unit.' });
    s += wire([[298, R], [386, R]]);
    // x_t → W_x → +
    s += wire([[400, 344], [400, 294]]);
    s += wbox(400, 280, 'W_x', { w: 48, tip: 'Wx · x_t = ' + num(g.ix, 4) });
    s += wire([[400, 266], [400, R + 14]]);
    // bias
    s += wbox(400, 110, 'b', { w: 40, tip: 'bias b = ' + num(g.b, 4) });
    s += wire([[400, 124], [400, R - 14]]);
    s += op(400, R, '+', { tip: 'pre-activation = Wh·h + Wx·x + b\n  = ' + num(g.rec, 4) + ' + ' + num(g.ix, 4) + ' + ' + num(g.b) + ' = ' + num(g.z, 4) });
    s += wire([[414, R], [523, R]]);
    s += gate(560, R, 'tanh', d.h, { range: 'sym', tip: 'h_t = tanh(' + num(g.z, 4) + ') = ' + num(d.h, 4) });
    s += wire([[595, R], [680, R], [680, Y], [870, Y]]);

    s += pill(95, Y, 'h_{' + prev + '}', d.hprev[d.u], { tip: hTip });
    s += pill(342, R, '', g.rec, { tip: 'Wh · h_' + prev });
    s += pill(400, 240, '', g.ix, { tip: 'Wx · x_t' });
    s += pill(400, 157, 'b', g.b);
    s += pill(400, 324, 'x_t', xin, { tip: xTip });
    s += pill(470, R, '', g.z, { tip: 'pre-activation' });
    s += pill(790, Y, '', d.h, { tip: 'the new hidden state h_t' });
    s += txt(880, Y + 5, 'h_t', { size: 16, weight: 600, anchor: 'start' });
    s += txt(40, Y - 16, 'hidden state', { size: 11, color: MUTED, anchor: 'start' });
    s += txt(440, 346, 'input', { size: 11, color: MUTED, anchor: 'start' });
    return svg(940, 356, s, 'Simple tanh RNN cell of unit ' + (d.u + 1) + ' at step t = ' + d.t + '.');
  }

  /* ------------------------------------------------------ convolution */
  /**
   * @param c   convAt() of the selected filter and position
   * @param o   { t, li, ch, act, actName, a, pool: {even, v0, v1, out, tp} | null }
   */
  function conv(c, o) {
    const K = c.K, rows = Math.min(c.cin, 6), more = c.cin - rows;
    const cw = 44, cg = 4, x0 = 40, top = 100, rh = 50;
    const gridR = x0 + K * (cw + cg);
    const H = Math.max(240, top + rows * rh + (more ? 26 : 0) + 40);
    const midY = top + (rows * rh) / 2 - 8;
    const SX = gridR + 120;                       // big Σ
    let s = '';

    // tap header: which sample each tap reads
    for (let j = 0; j < K; j++) {
      const idx = o.t + c.conv.tapOffset(j);
      s += txt(x0 + j * (cw + cg) + cw / 2, top - 10, 'x[' + idx + ']', { size: 10, color: MUTED, mono: true });
    }
    c.terms.slice(0, rows).forEach((tr, r) => {
      const y = top + r * rh;
      tr.row.forEach((cell, j) => {
        const cx = x0 + j * (cw + cg);
        s += '<g class="blk">' + tip('x[' + cell.idx + '] = ' + (cell.outside ? '0 (padding)' : num(cell.xv, 4)) +
          '\nw[' + j + '] = ' + num(cell.w, 4) + '\nproduct = ' + num(cell.p, 4)) +
          '<rect x="' + cx + '" y="' + y + '" width="' + cw + '" height="36" rx="4" fill="' +
          (cell.outside ? '#f3f5f7' : '#fff') + '" stroke="#c9d6e6"/>' +
          txt(cx + cw / 2, y + 14, cell.outside ? 'pad' : num(cell.xv, 2), { size: 10, mono: true, color: cell.outside ? '#b6c0ca' : '#5b6873' }) +
          txt(cx + cw / 2, y + 29, '×' + num(cell.w, 2), { size: 10, mono: true, color: signColor(cell.w), weight: 600 }) +
          '</g>';
      });
      const name = o.li === 0 ? 'signal' : 'L' + o.li + ' f' + (tr.ci + 1);
      s += txt(x0 - 6, y + 22, name, { size: 10, color: MUTED, anchor: 'end' });
      s += wire([[gridR + 2, y + 18], [SX - 14, midY]], { arrow: true, color: '#8796a6', width: 1 });
      if (c.cin > 1) s += pill(gridR + 44, y + 18, 'Σ', tr.sub, { tip: 'the ' + K + ' products of this input channel, summed' });
    });
    if (more) {
      s += txt(x0, top + rows * rh + 10, '+ ' + more + ' more input channel' + (more > 1 ? 's' : '') +
        ' (in the table below)', { size: 10.5, color: MUTED, anchor: 'start' });
    }

    let x = SX;
    s += op(x, midY, 'Σ', { tip: 'Σ over all ' + (c.cin * K) + ' products = ' + num(c.sum, 4) });
    s += wire([[x + 14, midY], [x + 66, midY]]);
    x += 80;
    s += wbox(x, midY - 62, 'b', { w: 40, tip: 'bias of filter ' + (o.ch + 1) + ' = ' + num(c.bias, 4) });
    s += wire([[x, midY - 48], [x, midY - 14]]);
    s += op(x, midY, '+', { tip: 'z = Σ + b = ' + num(c.sum, 4) + ' + ' + num(c.bias, 4) + ' = ' + num(c.z, 4) });
    s += pill(x - 40, midY - 22, '', c.sum, { tip: 'Σ of all products' });
    s += wire([[x + 14, midY], [x + 66, midY]]);
    s += pill(x + 40, midY + 22, 'z', c.z, { tip: 'pre-activation' });
    x += 106;
    s += gate(x, midY, o.actName, o.a, { w: 78, tip: o.actExpr + '  →  a = ' + num(o.a, 4) +
      (o.a === 0 && o.act === 'relu' ? '\nThe filter is silent here.' : '') });
    if (o.pool) {
      const p = o.pool;
      s += wire([[x + 40, midY], [x + 92, midY]]);
      x += 132;
      s += gate(x, midY, 'max ×2', p.out, { w: 78, fill: '#3d7cc0', tip: 'out[' + p.tp + '] = max( a[' + p.even + '] = ' + num(p.v0) +
        ' , a[' + (p.even + 1) + '] = ' + num(p.v1) + ' ) = ' + num(p.out, 4) });
      s += wire([[x, midY + 92], [x, midY + 23]]);
      s += pill(x, midY + 72, 'a[' + (o.t === p.even ? p.even + 1 : p.even) + ']',
        o.t === p.even ? p.v1 : p.v0, { tip: 'the neighbouring position the pool compares with' });
      s += pill(x - 66, midY - 22, 'a', o.a);
    }
    s += wire([[x + 40, midY], [x + 92, midY]]);
    s += pill(x + 66, midY - 22, '', o.pool ? o.pool.out : o.a, { tip: 'the value drawn in this filter\'s map' });
    s += txt(x + 98, midY + 5, 'out[' + (o.pool ? o.pool.tp : o.t) + ']', { size: 14, weight: 600, anchor: 'start' });
    const W = x + 160;
    s += txt(x0, 62, 'kernel window over the input — each cell is value × weight', { size: 11, color: MUTED, anchor: 'start' });
    return svg(W, H, s, 'Filter ' + (o.ch + 1) + ' of layer ' + (o.li + 1) + ' at position t = ' + o.t + '.');
  }

  /* ------------------------------------------------------ state space */
  /**
   * @param d  stepDetail() of the selected channel
   * @param o  { ch, ySum, y, out, gate }  gate is the raw Mamba gate input (null for S4)
   */
  function ssm(d, o) {
    const isS4 = d.mode === 's4';
    const M = 170, T = 84, Bt = 296;            // main row, feedback row, skip row
    let s = '';
    const modes = d.N + (isS4 ? ' complex modes' : ' modes');

    s += wire([[40, M], [168, M]]);
    s += wbox(200, M, isS4 ? 'B̄' : 'B̄(t)', { w: 60, tip: isS4
      ? 'B̄ = (Ā − 1)/A — one value per mode, the same at every step'
      : 'B̄(t) = Δ(t)·B(t) — B(t) is computed from the input at this step' });
    s += wire([[230, M], [286, M]]);
    s += op(300, M, '+', { tip: 'x_t = Ā ⊙ x_t−1 + B̄ · u_t   (for each of the ' + d.N + ' modes)' });
    s += wire([[314, M], [350, M]]);
    s += '<g class="blk">' + tip('the state x_t: ' + modes + ', listed in the table below') +
      '<rect x="350" y="' + (M - 22) + '" width="100" height="44" rx="22" fill="#fff" stroke="#1d2b38"/>' +
      txt(400, M - 2, 'state x_t', { size: 12, weight: 600 }) +
      txt(400, M + 13, modes, { size: 10, color: MUTED }) + '</g>';
    // feedback: x_t → delay → Ā → +
    s += wire([[400, M - 22], [400, T], [332, T]]);
    s += wbox(300, T, isS4 ? 'Ā' : 'Ā(t)', { w: 60, tip: isS4
      ? 'Ā = exp(Δ·A) — how much of each mode survives one sample'
      : 'Ā(t) = exp(Δ(t)·A) — Δ(t) decides how fast the state forgets' });
    s += wire([[300, T + 14], [300, M - 14]]);
    s += txt(408, T - 8, 'x_{t−1} (one step later)', { size: 10.5, color: MUTED, anchor: 'start' });
    // readout C
    s += wire([[450, M], [496, M]]);
    s += wbox(526, M, isS4 ? 'C' : 'C(t)', { w: 56, tip: isS4
      ? 'C — fixed readout weights, one complex value per mode'
      : 'C(t) — readout weights computed from the input at this step' });
    s += wire([[554, M], [596, M]]);
    s += op(610, M, 'Σ', { tip: (isS4 ? '2·Re( Σₙ Cₙ xₙ )' : 'Σₙ Cₙ(t) xₙ') + ' = ' + num(o.ySum, 4) });
    s += wire([[624, M], [686, M]]);
    s += op(700, M, '+', { tip: 'y_t = Σ + D·u_t = ' + num(o.ySum, 4) + ' + ' + num(d.Dskip * d.u, 4) + ' = ' + num(o.y, 4) });
    // skip path D·u
    s += wire([[90, M], [90, Bt], [474, Bt]], { arrow: true });
    s += wbox(500, Bt, 'D', { w: 44, tip: 'skip weight D = ' + num(d.Dskip, 4) + '\nD·u_t = ' + num(d.Dskip * d.u, 4) });
    s += wire([[522, Bt], [700, Bt], [700, M + 14]]);
    s += pill(610, Bt, 'D·u', d.Dskip * d.u);

    if (!isS4) {
      // Δ(t) from the input steers Ā and B̄
      s += wire([[130, M], [130, T], [162, T]], { arrow: true });
      s += gate(200, T, 'softplus Δ', d.dt, { w: 76, fill: '#3d7cc0', tip: 'Δ(t) = softplus( ' + num(Math.log(d.dtBase)) + ' + ' + num(d.dtW) +
        '·u_t ) = ' + num(d.dt, 4) + '\nLarge Δ writes the input into the state, Δ → 0 ignores it.' });
      s += wire([[238, T], [268, T]], { dash: true, color: BLUE });
      s += wire([[200, T + 22], [200, M - 14]], { dash: true, color: BLUE });
    }

    s += wire([[714, M], [766, M]]);
    if (isS4) {
      s += gate(800, M, 'SiLU', o.out, { w: 64, tip: 'out = y·σ(y) = ' + num(o.y, 4) + ' · ' + num(1 / (1 + Math.exp(-o.y)), 4) + ' = ' + num(o.out, 4) +
        '\nThe recurrence is linear; this makes the stack non-linear.' });
      s += wire([[832, M], [880, M]]);
    } else {
      const sg = o.gate / (1 + Math.exp(-o.gate));
      s += op(800, M, 'mul', { tip: 'out = y · SiLU(g) = ' + num(o.y, 4) + ' · ' + num(sg, 4) + ' = ' + num(o.out, 4) });
      s += gate(800, 246, 'SiLU', sg, { w: 60, tip: 'SiLU(g) = g·σ(g), g = Wg·x + b = ' + num(o.gate, 4) });
      s += wire([[800, 224], [800, M + 14]]);
      s += wire([[800, 318], [800, 268]]);
      s += pill(860, 300, 'gate g', o.gate, { tip: 'the gate branch, computed from the layer input' });
      s += wire([[814, M], [880, M]]);
    }

    s += pill(60, M - 22, 'u_t', d.u, { tip: 'the input to the recurrence at step ' + d.t });
    s += pill(655, M - 22, '', o.ySum, { tip: 'contribution of all modes' });
    s += pill(744, M - 22, 'y', o.y);
    s += pill(850, M - 22, '', o.out, { tip: 'this channel\'s map at t = ' + d.t });
    s += txt(888, M + 5, 'out', { size: 15, weight: 600, anchor: 'start' });
    s += txt(40, M + 30, 'input', { size: 11, color: MUTED, anchor: 'start' });
    return svg(940, 336, s, (isS4 ? 'S4D' : 'Mamba (selective)') + ' channel ' + (o.ch + 1) +
      ' at step t = ' + d.t + (isS4 ? '.' : '. Dashed lines: Δ(t) steering Ā and B̄.'));
  }

  /* ------------------------------------------------------ graph network */
  /**
   * @param d  stepDetail() of the selected node
   * @param o  { li, ch, fname, sSelf, sNeigh, src, L }
   */
  function gnn(d, o) {
    const shown = d.neigh.slice(0, 6), more = d.neigh.length - shown.length;
    const AX = 230, AY = 222, P = 120;            // aggregator, + node row
    let s = '';
    const feats = (j) => Array.from({ length: d.D }, (_, i) => o.fname(i) + ' = ' + num(o.src[i * o.L + j], 4)).join('\n');

    // the node itself
    s += '<g class="blk">' + tip('node ' + d.node + ' — its own features:\n' + feats(d.node)) +
      '<circle cx="70" cy="' + P + '" r="22" fill="#fde9d9" stroke="#c2760f" stroke-width="1.5"/>' +
      txt(70, P + 4, String(d.node), { size: 12, weight: 600 }) + '</g>';
    s += txt(70, P - 30, 'this node', { size: 11, color: MUTED });
    s += wire([[92, P], [384, P]]);
    s += wbox(416, P, 'W_{self}', { w: 64, tip: 'W_self · h_i = ' + num(o.sSelf, 4) });
    s += wire([[448, P], [586, P]]);

    // neighbours → aggregate
    shown.forEach((j, k) => {
      const y = 190 + k * 30;
      s += '<g class="blk">' + tip('neighbour node ' + j + ':\n' + feats(j)) +
        '<circle cx="70" cy="' + y + '" r="12" fill="#e6effa" stroke="' + BLUE + '"/>' +
        txt(70, y + 4, String(j), { size: 10, weight: 600, color: BLUE }) + '</g>';
      s += wire([[82, y], [AX - 34, AY]], { color: '#8796a6', width: 1 });
    });
    if (!shown.length) s += txt(70, 230, 'no neighbours', { size: 11, color: MUTED });
    if (more) s += txt(70, 190 + shown.length * 30, '+' + more + ' more', { size: 10.5, color: MUTED });
    s += txt(70, 178 - 8, 'neighbours (' + d.degree + ')', { size: 11, color: MUTED });
    s += gate(AX, AY, d.agg_kind, '', { w: 64, tip: d.agg_kind + ' of the ' + d.degree + ' neighbours, per feature:\n' +
      Array.from(d.agg).map((v, i) => '  ' + o.fname(i) + ' = ' + num(v, 4)).join('\n') });
    s += wire([[AX + 32, AY], [379, AY]]);
    s += wbox(416, AY, 'W_{neigh}', { w: 70, tip: 'W_neigh · agg = ' + num(o.sNeigh, 4) });
    s += wire([[451, AY], [600, AY], [600, P + 14]]);

    s += wbox(600, 44, 'b', { w: 40, tip: 'bias = ' + num(d.bias, 4) });
    s += wire([[600, 58], [600, P - 14]]);
    s += op(600, P, '+', { tip: 'pre = self + neighbours + b = ' + num(o.sSelf, 4) + ' + ' + num(o.sNeigh, 4) + ' + ' + num(d.bias, 4) + ' = ' + num(d.pre, 4) });
    s += wire([[614, P], [714, P]]);
    s += gate(750, P, 'ReLU', d.out, { tip: 'h′ = max(0, ' + num(d.pre, 4) + ') = ' + num(d.out, 4) +
      (d.out === 0 ? '\nThe channel is silent at this node.' : '') });
    s += wire([[785, P], [870, P]]);

    s += pill(510, P, 'self', o.sSelf);
    s += pill(320, AY, '', d.agg_kind + ' of ' + d.D + ' features');
    s += pill(520, AY, 'neigh', o.sNeigh);
    s += pill(664, P - 22, 'pre', d.pre);
    s += pill(830, P - 22, '', d.out, { tip: 'this channel\'s value at node ' + d.node });
    s += txt(878, P + 5, 'h′', { size: 16, weight: 600, anchor: 'start' });
    const H = Math.max(300, 210 + shown.length * 30 + (more ? 20 : 0));
    return svg(940, H, s, 'Message passing into node ' + d.node + ', channel ' + (o.ch + 1) +
      ' of layer ' + (o.li + 1) + '.');
  }

  /* ------------------------------------------------------ output head */
  /**
   * @param o { C, L, head, headLabel, emb, z, p, cls, trueIdx }
   */
  function output(o) {
    const K = o.z.length, rowsC = Math.min(o.C, 8);
    const rows = Math.max(rowsC, K);
    const top = 50, rh = 30;
    const H = top + rows * rh + 40;
    const mid = top + (rows * rh) / 2 - rh / 2;
    let s = '';

    // final feature maps
    s += txt(60, 28, 'last layer · ' + o.C + ' maps × ' + o.L, { size: 11, color: MUTED, anchor: 'start' });
    for (let c = 0; c < rowsC; c++) {
      const y = top + c * rh + (rows - rowsC) * rh / 2;
      s += '<rect x="60" y="' + (y - 8) + '" width="90" height="16" rx="3" fill="#e6effa" stroke="#c9d6e6"/>';
      s += txt(105, y + 4, 'map ' + (c + 1), { size: 10, color: '#5b6873' });
      s += wire([[150, y], [206, mid]], { color: '#8796a6', width: 1 });
    }
    if (o.C > rowsC) s += txt(105, top + rowsC * rh + 6, '+' + (o.C - rowsC) + ' more', { size: 10, color: MUTED });
    s += gate(250, mid, o.headLabel, '', { w: 84, tip: o.headTip });

    // embedding
    const ex = 360;
    if (o.emb) {
      const n = Math.min(o.emb.length, 8);
      for (let i = 0; i < n; i++) {
        const y = top + i * rh + (rows - n) * rh / 2;
        s += wire([[292, mid], [ex - 40, y]], { color: '#8796a6', width: 1 });
        s += pill(ex, y, 'h' + i, o.emb[i]);
        s += wire([[ex + 40, y], [466, mid]], { color: '#8796a6', width: 1, arrow: false });
      }
    } else {
      s += wire([[292, mid], [462, mid]]);
      s += pill(377, mid, '', o.C + '×' + o.L + ' = ' + (o.C * o.L) + ' numbers');
    }
    s += wbox(500, mid, 'W·h + b', { w: 72, h: 32, tip: 'logit_j = Σ_c W[j][c]·h[c] + b[j] — one weight per (class, input)' });

    // logits → softmax → probabilities
    for (let j = 0; j < K; j++) {
      const y = top + j * rh + (rows - K) * rh / 2;
      s += wire([[536, mid], [598, y]], { color: '#8796a6', width: 1 });
      s += pill(640, y, 'z' + j, o.z[j], { tip: o.cls[j].name + ' — logit' });
      s += wire([[684, y], [736, mid]], { color: '#8796a6', width: 1, arrow: false });
    }
    s += gate(760, mid, 'softmax', '', { w: 72, tip: 'p_j = exp(z_j − z_max) / Σ exp(z − z_max)' });
    for (let j = 0; j < K; j++) {
      const y = top + j * rh + (rows - K) * rh / 2;
      s += wire([[796, mid], [826, y]], { color: '#8796a6', width: 1 });
      // colour chip, name and share, with a thin bar underneath for the size
      const name = o.cls[j].name.replace(/\s*\(.*\)$/, '').split(' / ')[0];
      s += '<g class="blk">' + tip(o.cls[j].name + ': p = ' + (o.p[j] * 100).toFixed(2) + '%') +
        '<rect x="832" y="' + (y - 9) + '" width="8" height="8" rx="2" fill="' + o.cls[j].color + '"/>' +
        txt(845, y - 1, name + '  ' + (o.p[j] * 100).toFixed(1) + '%',
          { size: 10.5, anchor: 'start', weight: j === o.trueIdx ? 700 : 400 }) +
        '<rect x="845" y="' + (y + 3) + '" width="100" height="3" rx="1.5" fill="#dfe6ee"/>' +
        '<rect x="845" y="' + (y + 3) + '" width="' + (100 * o.p[j]) + '" height="3" rx="1.5" fill="' + o.cls[j].color + '"/>' +
        '</g>';
    }
    s += txt(500, 28, 'linear layer', { size: 11, color: MUTED });
    s += txt(640, 28, 'logits', { size: 11, color: MUTED });
    s += txt(845, 28, 'probabilities', { size: 11, color: MUTED, anchor: 'start' });
    return svg(960, Math.max(H, 180), s, 'From the last layer to the class probabilities.' +
      (o.trueIdx >= 0 ? ' The true class is in bold.' : ''));
  }

  return { lstm, gru, rnn, conv, ssm, gnn, output };
})();
