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
    let H = Math.max(240, top + rows * rh + (more ? 26 : 0) + 40);
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
    let carry = o.a;                 // what leaves the block: a, or a + skip in a residual block
    if (o.skip) {
      const sk = o.skip;
      const rowsBottom = top + (rows - 1) * rh + 36 + (more ? 26 : 0);
      const yS = rowsBottom + 26;
      s += wire([[x + 40, midY], [x + 78, midY]]);
      x += 92;
      carry = o.a + sk.s;
      s += op(x, midY, '+', { tip: 'residual: y = a + skip = ' + num(o.a, 4) + ' + ' + num(sk.s, 4) + ' = ' + num(carry, 4) });
      // the skip leaves the block input and rejoins after the activation
      s += wire([[x0 + 10, rowsBottom + 2], [x0 + 10, yS], [x, yS], [x, midY + 14]], { color: BLUE, dash: true });
      const mx = (x0 + x) / 2;
      if (sk.proj) {
        s += wbox(mx - 50, yS, '1×1 conv', { w: 74, tip: 'skip = Σ_c P[' + o.ch + '][c] · x[c][' + o.t + '] + b_P = ' + num(sk.s, 4) +
          '\nThe channel count changes here, so the skip needs a learned 1×1 convolution.' });
        s += pill(mx + 50, yS, 'skip', sk.s, { tip: 'the projected block input at t = ' + o.t });
      } else {
        s += pill(mx, yS, 'skip', sk.s, { tip: 'x[' + o.ch + '][' + o.t + '] — the same channel of the block input, passed on untouched' });
      }
      s += txt(x0 + 16, yS - 8, 'skip connection', { size: 10.5, color: BLUE, anchor: 'start' });
      H = Math.max(H, yS + 30);
      // the + node is narrower than a block, so close the gap to where the next wire starts
      s += wire([[x + 14, midY], [x + 40, midY]], { arrow: false });
    }
    const nm = o.skip ? 'y' : 'a';
    if (o.pool) {
      const p = o.pool;
      s += wire([[x + 40, midY], [x + 92, midY]]);
      x += 132;
      s += gate(x, midY, 'max ×2', p.out, { w: 78, fill: '#3d7cc0', tip: 'out[' + p.tp + '] = max( ' + nm + '[' + p.even + '] = ' + num(p.v0) +
        ' , ' + nm + '[' + (p.even + 1) + '] = ' + num(p.v1) + ' ) = ' + num(p.out, 4) });
      s += wire([[x, midY + 92], [x, midY + 23]]);
      s += pill(x, midY + 72, nm + '[' + (o.t === p.even ? p.even + 1 : p.even) + ']',
        o.t === p.even ? p.v1 : p.v0, { tip: 'the neighbouring position the pool compares with' });
      s += pill(x - 66, midY - 22, nm, carry);
    }
    s += wire([[x + 40, midY], [x + 92, midY]]);
    s += pill(x + 66, midY - 22, '', o.pool ? o.pool.out : carry, { tip: 'the value drawn in this filter\'s map' });
    s += txt(x + 98, midY + 5, 'out[' + (o.pool ? o.pool.tp : o.t) + ']', { size: 14, weight: 600, anchor: 'start' });
    const W = x + 160;
    s += txt(x0, 62, 'kernel window over the input — each cell is value × weight', { size: 11, color: MUTED, anchor: 'start' });
    return svg(W, H, s, 'Filter ' + (o.ch + 1) + ' of layer ' + (o.li + 1) + ' at position t = ' + o.t + '.');
  }

  /* ------------------------------------------------------ dense unit (MLP) */
  /** Inputs to show as rows: the strongest contributions, and position t in the first layer. */
  function pickRows(n, score, t, k) {
    if (n <= k) return Array.from({ length: n }, (_, i) => i);
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => score(b) - score(a)).slice(0, k);
    if (t != null && !idx.includes(t)) { idx.pop(); idx.push(t); }
    return idx.sort((a, b) => a - b);
  }

  /** The window with a second curve over it: the weights of a unit, or its edge outputs. */
  function strip(y, h, vals, sig, t, cap) {
    const x0 = 40, w = 860, n = vals.length;
    let m = 1e-9, ms = 1e-9;
    for (let i = 0; i < n; i++) { m = Math.max(m, Math.abs(vals[i])); ms = Math.max(ms, Math.abs(sig[i])); }
    const X = (i) => x0 + i * w / (n - 1);
    let s = '<rect x="' + x0 + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="6" fill="#fff" stroke="#c9d6e6"/>';
    s += '<line x1="' + x0 + '" x2="' + (x0 + w) + '" y1="' + (y + h / 2) + '" y2="' + (y + h / 2) + '" stroke="#dfe4ea"/>';
    let ps = '', pv = '';
    for (let i = 0; i < n; i++) {
      ps += (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + (y + h / 2 - sig[i] / ms * (h / 2 - 4)).toFixed(1);
    }
    s += '<path d="' + ps + '" fill="none" stroke="#b6c0ca" stroke-width="1.2"/>';
    for (let i = 0; i < n; i++) {
      const Y = y + h / 2 - vals[i] / m * (h / 2 - 4);
      pv += '<line x1="' + X(i).toFixed(1) + '" x2="' + X(i).toFixed(1) + '" y1="' + (y + h / 2) + '" y2="' + Y.toFixed(1) +
        '" stroke="' + (vals[i] >= 0 ? POS : NEG) + '" stroke-width="2.4"/>';
    }
    s += pv;
    if (t != null) {
      s += '<rect x="' + (X(t) - 3) + '" y="' + (y + 1) + '" width="6" height="' + (h - 2) + '" fill="rgba(29,78,216,0.22)"/>';
      s += txt(X(t), y + h + 13, 't = ' + t, { size: 10, color: BLUE, mono: true });
    }
    s += txt(x0, y - 8, cap, { size: 11, color: MUTED, anchor: 'start' });
    return s;
  }

  /**
   * @param d  { x, w, b, z, a, nin }   one MLP unit
   * @param o  { li, ch, t, act, actName, actExpr, name(i) }
   */
  function dense(d, o) {
    let s = '';
    const first = o.li === 0;
    const prod = (i) => d.w[i] * d.x[i];
    const rowsIdx = pickRows(d.nin, (i) => Math.abs(prod(i)), first ? o.t : null, 7);
    let shownSum = 0;
    rowsIdx.forEach((i) => { shownSum += prod(i); });
    const rest = d.nin - rowsIdx.length;
    let total = 0;
    for (let i = 0; i < d.nin; i++) total += prod(i);

    let top = 40;
    if (first) {
      s += strip(36, 70, d.w, d.x, o.t, 'this unit\'s ' + d.nin + ' weights over the window (colour) — the input in grey');
      top = 150;
    }
    const rh = 38, nRows = rowsIdx.length + (rest ? 1 : 0);
    // a unit with only one or two inputs still needs room above for the bias
    if (top + (nRows - 1) * rh / 2 < 84) top = 84 - (nRows - 1) * rh / 2;
    const midY = top + (nRows - 1) * rh / 2;
    const SX = 470;
    rowsIdx.forEach((i, r) => {
      const y = top + r * rh;
      const hot = first && i === o.t;
      s += '<g class="blk">' + tip(o.name(i) + ' = ' + num(d.x[i], 4) + '\nweight w[' + i + '] = ' + num(d.w[i], 4) +
        '\nproduct = ' + num(prod(i), 4)) +
        '<rect x="40" y="' + (y - 14) + '" width="250" height="28" rx="5" fill="' + (hot ? '#e8f0fc' : '#fff') +
        '" stroke="' + (hot ? BLUE : '#c9d6e6') + '"/>' +
        txt(50, y + 4, o.name(i) + ' = ' + num(d.x[i]), { size: 11, mono: true, anchor: 'start', color: '#5b6873' }) +
        txt(280, y + 4, '× ' + num(d.w[i]), { size: 11, mono: true, anchor: 'end', color: signColor(d.w[i]), weight: 600 }) + '</g>';
      s += wire([[290, y], [330, y]], { arrow: false });
      s += pill(370, y, '', prod(i));
      s += wire([[408, y], [SX - 14, midY]], { color: '#8796a6', width: 1 });
    });
    if (rest) {
      const y = top + rowsIdx.length * rh;
      s += '<g class="blk">' + tip('the other ' + rest + ' products, summed') +
        '<rect x="40" y="' + (y - 14) + '" width="250" height="28" rx="5" fill="#f7f9fb" stroke="#dfe4ea" stroke-dasharray="3 3"/>' +
        txt(50, y + 4, '+ ' + rest + ' more inputs', { size: 11, anchor: 'start', color: MUTED }) + '</g>';
      s += wire([[290, y], [330, y]], { arrow: false });
      s += pill(370, y, '', total - shownSum);
      s += wire([[408, y], [SX - 14, midY]], { color: '#8796a6', width: 1 });
    }
    let x = SX;
    s += op(x, midY, 'Σ', { tip: 'Σ over all ' + d.nin + ' products = ' + num(total, 4) });
    s += wire([[x + 14, midY], [x + 66, midY]]);
    x += 80;
    s += wbox(x, midY - 56, 'b', { w: 40, tip: 'bias = ' + num(d.b, 4) });
    s += wire([[x, midY - 42], [x, midY - 14]]);
    s += op(x, midY, '+', { tip: 'z = Σ + b = ' + num(total, 4) + ' + ' + num(d.b, 4) + ' = ' + num(d.z, 4) });
    s += pill(x - 40, midY - 22, '', total);
    s += wire([[x + 14, midY], [x + 66, midY]]);
    s += pill(x + 40, midY + 22, 'z', d.z);
    x += 106;
    s += gate(x, midY, o.actName, d.a, { w: 78, tip: o.actExpr + '  →  a = ' + num(d.a, 4) });
    s += wire([[x + 40, midY], [x + 110, midY]]);
    s += pill(x + 75, midY - 22, '', d.a, { tip: 'the unit\'s output — the bar in its box' });
    s += txt(x + 116, midY + 5, 'a', { size: 15, weight: 600, anchor: 'start' });
    const H = Math.max(top + nRows * rh + 10, midY + 70);
    return svg(940, H, s, 'Dense unit ' + (o.ch + 1) + ' of layer ' + (o.li + 1) +
      (first ? ': every one of the ' + d.nin + ' samples has its own weight.' : '.'));
  }

  /* ------------------------------------------------------ KAN node */
  /** A small plot of one edge function, with this example's input marked on it. */
  function edgePlot(x, y, w, h, e) {
    const lo = e.curve[0][0], hi = e.curve[e.curve.length - 1][0];
    let m = 1e-6;
    e.curve.forEach((p) => { m = Math.max(m, Math.abs(p[1])); });
    m = Math.max(m, Math.abs(e.phi));
    const X = (v) => x + (v - lo) / (hi - lo) * w;
    const Y = (v) => y + h / 2 - v / m * (h / 2 - 3);
    let s = '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="4" fill="#fff" stroke="#c9d6e6"/>';
    // the spline grid, where the curve can bend
    s += '<rect x="' + X(e.curve.gridLo) + '" y="' + (y + 1) + '" width="' + (X(e.curve.gridHi) - X(e.curve.gridLo)) +
      '" height="' + (h - 2) + '" fill="#f1f6fd"/>';
    s += '<line x1="' + x + '" x2="' + (x + w) + '" y1="' + Y(0) + '" y2="' + Y(0) + '" stroke="#dfe4ea"/>';
    s += '<path d="' + e.curve.map((p, q) => (q ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join(' ') +
      '" fill="none" stroke="' + BLUE + '" stroke-width="1.6"/>';
    const xi = Math.max(lo, Math.min(hi, e.x));
    s += '<line x1="' + X(xi) + '" x2="' + X(xi) + '" y1="' + (y + 2) + '" y2="' + (y + h - 2) + '" stroke="#e0342b" stroke-dasharray="2 2"/>';
    s += '<circle cx="' + X(xi) + '" cy="' + Y(e.phi) + '" r="3" fill="#e0342b"/>';
    return s;
  }

  /**
   * @param d  { edges:[{i, x, phi, base, spline, wb, active, curve}], sum, nin, rest, restSum, phiAll }
   * @param o  { li, ch, t, name(i), input }
   */
  function kan(d, o) {
    let s = '';
    const first = o.li === 0;
    let top = 50;
    if (first) {
      s += strip(36, 70, d.phiAll, o.input, o.t, 'φ(x_i) — what each of the ' + d.nin +
        ' edges delivers for this example (colour) — the input in grey');
      top = 160;
    }
    const rh = 56, nRows = d.edges.length + (d.rest ? 1 : 0);
    const midY = top + (nRows - 1) * rh / 2;
    const SX = 560;
    d.edges.forEach((e, r) => {
      const y = top + r * rh;
      const hot = first && e.i === o.t;
      s += txt(40, y + 4, o.name(e.i) + ' = ' + num(e.x), { size: 11, mono: true, anchor: 'start', color: hot ? BLUE : '#5b6873', weight: hot ? 700 : 400 });
      s += wire([[150, y], [196, y]]);
      s += '<g class="blk">' + tip('φ(x) = w_b·silu(x) + Σ c_m·B_m(x)\n' +
        'at x = ' + num(e.x, 4) + (first ? '' : ' (standardised u = ' + num(e.u, 3) + ')') +
        ':\n  base   w_b·silu(u) = ' + num(e.wb, 3) + ' · ' + num(siluK(e.u), 4) + ' = ' + num(e.base, 4) +
        '\n  spline Σ c_m·B_m(u) = ' + num(e.spline, 4) +
        (e.active.length ? '  (' + e.active.map((a) => 'c' + a.m + '·B' + a.m).join(' + ') + ')' : '  (x is outside the grid)') +
        '\n  φ = ' + num(e.phi, 4)) +
        edgePlot(200, y - 22, 170, 44, e) + '</g>';
      s += wire([[372, y], [414, y]], { arrow: false });
      s += pill(452, y, 'φ', e.phi);
      s += wire([[494, y], [SX - 14, midY]], { color: '#8796a6', width: 1 });
    });
    if (d.rest) {
      const y = top + d.edges.length * rh;
      s += '<g class="blk">' + tip('the other ' + d.rest + ' edges, summed') +
        '<rect x="40" y="' + (y - 14) + '" width="330" height="28" rx="5" fill="#f7f9fb" stroke="#dfe4ea" stroke-dasharray="3 3"/>' +
        txt(50, y + 4, '+ ' + d.rest + ' more edges, each with its own φ', { size: 11, anchor: 'start', color: MUTED }) + '</g>';
      s += wire([[372, y], [414, y]], { arrow: false });
      s += pill(452, y, '', d.restSum);
      s += wire([[494, y], [SX - 14, midY]], { color: '#8796a6', width: 1 });
    }
    s += op(SX, midY, 'Σ', { tip: 'h = Σ over all ' + d.nin + ' edges = ' + num(d.sum, 4) +
      '\nThe node only adds up — every nonlinearity sits on the edges.' });
    s += wire([[SX + 14, midY], [SX + 150, midY]]);
    s += pill(SX + 82, midY - 22, '', d.sum, { tip: 'the node value — the bar in its box' });
    s += txt(SX + 158, midY + 5, 'h', { size: 15, weight: 600, anchor: 'start' });
    s += txt(SX + 40, midY + 40, 'no bias, no activation on the node', { size: 10.5, color: MUTED, anchor: 'start' });
    s += txt(200, top - 34, 'edge functions φ(x) — shaded: the spline grid' + (first ? ' ' + KAN_LO + ' … ' + KAN_HI : ' (μ ± 2σ of what arrives)') +
      ', red: where this example lands', { size: 11, color: MUTED, anchor: 'start' });
    const H = Math.max(top + nRows * rh, midY + 70);
    return svg(940, H, s, 'KAN node ' + (o.ch + 1) + ' of layer ' + (o.li + 1) + ': φ(x) = w_b·silu(x) + Σ c_m·B_m(x) on every edge.');
  }

  /* ------------------------------------------------------ ResNet-1D block */
  /** A conv → LN → ReLU stage drawn as one box, with the numbers of channel ch at t inside. */
  function stageBox(x, y, w, r, last, tipText) {
    const h = last ? 58 : 74;
    let s = '<g class="blk">' + tip(tipText) +
      '<rect x="' + x + '" y="' + (y - h / 2) + '" width="' + w + '" height="' + h + '" rx="7" fill="#fff" stroke="' + BLUE + '" stroke-width="1.3"/>' +
      '<rect x="' + x + '" y="' + (y - h / 2) + '" width="' + w + '" height="20" rx="7" fill="' + BLUE + '"/>' +
      '<rect x="' + x + '" y="' + (y - h / 2 + 12) + '" width="' + w + '" height="8" fill="' + BLUE + '"/>' +
      txt(x + w / 2, y - h / 2 + 14, 'conv K=' + r.k + (r.bn ? ' → LN' : '') + (last ? '' : ' → ReLU'), { size: 11, color: '#fff', weight: 600 });
    let ly = y - h / 2 + 36;
    const line = (a, v) => {
      s += txt(x + 10, ly, a, { size: 10.5, anchor: 'start', color: MUTED }) +
        txt(x + w - 10, ly, num(v), { size: 10.5, anchor: 'end', mono: true, color: signColor(v), weight: 600 });
      ly += 16;
    };
    line('z (conv + bias)', r.z);
    if (r.bn) line('after LN', r.n);
    if (!last) line('after ReLU', r.h);
    return s + '</g>';
  }

  /**
   * @param d { rows:[{k, z, bn, n, h}], skip:{s, kind, x}|null, sum, out, xin, bn }
   * @param o { li, ch, t }
   */
  function resblock(d, o) {
    const M = 110, n = d.rows.length;
    const w = n === 3 ? 150 : 170, gap = 34, x0 = 130;
    let s = '';
    s += pill(62, M, 'x', d.xin, { tip: 'the block input at t = ' + o.t + (d.skip && d.skip.kind === 'identity'
      ? ', channel ' + (o.ch + 1) : ' (channel 1 shown; the first convolution reads all of them)') });
    s += txt(62, M - 22, 'block input', { size: 11, color: MUTED });
    let x = x0;
    s += wire([[100, M], [x - 2, M]]);
    d.rows.forEach((r, i) => {
      const last = i === n - 1;
      const bnTxt = r.bn ? '\nLN: (z − μ)/σ · γ + β = (' + num(r.z, 4) + ' − ' + num(r.bn.mu, 3) + ') / ' + num(r.bn.sd, 3) +
        ' · ' + num(r.bn.gamma, 3) + ' + ' + num(r.bn.beta, 3) + ' = ' + num(r.n, 4) : '';
      s += stageBox(x, M, w, r, last, 'conv ' + (i + 1) + ' (K=' + r.k + ') of the block, output channel ' + (o.ch + 1) +
        '\nz = Σ over all input channels and taps + bias = ' + num(r.z, 4) + bnTxt +
        (last ? '' : '\nReLU → ' + num(r.h, 4)) +
        (i > 0 ? '\nEvery convolution mixes all ' + d.F + ' channels of the one before; channel ' + (o.ch + 1) + ' is followed here.' : ''));
      x += w;
      if (!last) { s += wire([[x + 2, M], [x + gap - 2, M]]); x += gap; }
    });
    // the end of the block: + skip, then ReLU
    let after = x + 2;
    const yS = M + 108;
    if (d.skip) {
      s += wire([[after, M], [after + 32, M]]);
      const px = after + 46;
      s += op(px, M, '+', { tip: 'y = ' + (d.bn ? 'LN(z)' : 'z') + ' + skip = ' + num(d.rows[n - 1].n, 4) + ' + ' + num(d.skip.s, 4) + ' = ' + num(d.sum, 4) });
      s += wire([[62, M + 10], [62, yS], [px, yS], [px, M + 14]], { color: BLUE, dash: true });
      if (d.skip.kind === 'proj') {
        s += wbox((62 + px) / 2 - 60, yS, '1×1 conv' + (d.bn ? ' + LN' : ''), { w: 104,
          tip: 'the channel count changes, so the skip is a learned 1×1 convolution' + (d.bn ? ' with its own LN' : '') + ' = ' + num(d.skip.s, 4) });
        s += pill((62 + px) / 2 + 62, yS, 'skip', d.skip.s);
      } else {
        s += pill((62 + px) / 2, yS, 'skip', d.skip.s, { tip: 'x[' + o.ch + '][' + o.t + '] — the block input, added back untouched' });
      }
      s += txt(80, yS - 10, 'skip connection', { size: 10.5, color: BLUE, anchor: 'start' });
      s += pill(px + 2, M - 26, 'y', d.sum);
      after = px + 14;
    } else {
      s += txt(x0, yS - 20, 'skip connections are off — this block is a plain stack of ' + n + ' convolutions',
        { size: 11, color: MUTED, anchor: 'start' });
    }
    s += wire([[after, M], [after + 24, M]]);
    const rx = after + 56;
    s += gate(rx, M, 'ReLU', d.out, { w: 64, tip: 'out = max(0, ' + num(d.sum, 4) + ') = ' + num(d.out, 4) });
    s += wire([[rx + 33, M], [rx + 70, M]]);
    s += txt(rx + 76, M + 5, 'out', { size: 14, weight: 600, anchor: 'start' });
    const W = rx + 120;
    return svg(W, yS + 34, s, 'Residual block ' + (o.li + 1) + ', channel ' + (o.ch + 1) + ' at t = ' + o.t + '.');
  }

  /* ------------------------------------------------------ Inception module */
  const BRANCH_COLORS = ['#2b6cb0', '#2e9e5b', '#8e44ad', '#7b8794'];

  /**
   * @param d { branches:[{name, val, act, sel}], bott:{B, vals}|null, pre, bn, relu, shortcut:{s, sum, out}|null,
   *            xin, f, nb }
   * @param o { li, ch, t }
   */
  function inception(d, o) {
    let s = '';
    const ys = [62, 128, 194, 268];
    const BX = 400, CX = 540;
    s += pill(56, 165, 'x', d.xin, { tip: 'the module input at t = ' + o.t });
    s += txt(56, 143, 'module input', { size: 11, color: MUTED });
    let from = 96;
    if (d.bott) {
      s += wire([[96, 165], [170, 165]]);
      s += wbox(212, 165, '1×1 ×' + d.bott.B, { w: 80, tip: 'bottleneck: ' + d.bott.B + ' channels, each a weighted sum of the input channels\nat t: ' +
        d.bott.vals.map((v, i) => 'b' + i + ' = ' + num(v, 4)).join('  ') });
      s += txt(212, 196, 'bottleneck', { size: 10.5, color: MUTED });
      from = 252;
    }
    let mx = 1e-6;
    d.branches.forEach((b) => { mx = Math.max(mx, b.act); });
    d.branches.forEach((b, bi) => {
      const y = ys[bi], isPool = bi === d.branches.length - 1;
      const src = isPool ? 96 : from;
      const srcY = 165;
      s += wire([[src, srcY], [src + 30, srcY], [src + 30, y], [BX - 58, y]], { color: b.sel ? INK : '#9aa5b1', width: b.sel ? 1.6 : 1, over: true });
      const label = isPool ? 'max 3 → 1×1' : 'conv ' + b.name;
      s += gate(BX, y, label, b.val, { w: 112, fill: b.sel ? BRANCH_COLORS[bi] : '#b8c4d2',
        tip: (isPool ? 'pool branch: max over t−1 … t+1 of each input channel, then a 1×1 convolution'
          : 'branch ' + b.name + ': a convolution of length ' + b.name.slice(2) + (d.bott ? ' over the bottleneck' : ' over the input')) +
          '\nchannel ' + (b.slot + 1) + ' of this branch at t: ' + num(b.val, 4) +
          '\nmean activity of the branch over the window: ' + num(b.act, 4) + (b.sel ? '\n← the selected channel is in this branch' : '') });
      // activity bar: how strongly the branch responds to this example
      const bw = 100 * b.act / mx;
      s += '<rect x="' + (BX - 50) + '" y="' + (y + 25) + '" width="100" height="4" rx="2" fill="#e3e9f1"/>' +
        '<rect x="' + (BX - 50) + '" y="' + (y + 25) + '" width="' + bw + '" height="4" rx="2" fill="' + BRANCH_COLORS[bi] + '"/>';
      s += wire([[BX + 58, y], [CX - 18, y]], { color: b.sel ? INK : '#9aa5b1', width: b.sel ? 1.6 : 1 });
    });
    s += txt(BX, 22, 'four branches side by side — bar: mean activity on this example', { size: 10.5, color: MUTED });
    s += '<g class="blk">' + tip('the four branch outputs stacked into one map of ' + (d.f * d.nb) + ' channels') +
      '<rect x="' + (CX - 16) + '" y="40" width="32" height="250" rx="6" fill="#edf3fc" stroke="' + BLUE + '"/>' +
      '<text x="' + CX + '" y="165" text-anchor="middle" font-size="11" fill="' + BLUE + '" font-weight="600" transform="rotate(-90 ' + CX + ' 165)">concat</text></g>';
    const M = ys[d.branches.findIndex((b) => b.sel)];
    let x = CX + 16;
    if (d.bn) {
      s += wire([[x, M], [x + 40, M]]);
      x += 74;
      s += gate(x, M, 'LN', d.bn.out, { w: 64, tip: 'LN: (' + num(d.pre, 4) + ' − ' + num(d.bn.mu, 3) + ') / ' + num(d.bn.sd, 3) +
        ' · ' + num(d.bn.gamma, 3) + ' + ' + num(d.bn.beta, 3) + ' = ' + num(d.bn.out, 4) });
      x += 32;
    }
    s += wire([[x, M], [x + 36, M]]);
    x += 70;
    s += gate(x, M, 'ReLU', d.relu, { w: 64, tip: 'max(0, ' + num(d.bn ? d.bn.out : d.pre, 4) + ') = ' + num(d.relu, 4) });
    x += 32;
    let outV = d.relu;
    if (d.shortcut) {
      s += wire([[x, M], [x + 22, M]]);
      x += 36;
      s += op(x, M, '+', { tip: 'shortcut around the three modules: ReLU(' + num(d.relu, 4) + ' + ' + num(d.shortcut.s, 4) + ') = ' + num(d.shortcut.out, 4) });
      s += wire([[56, 178], [56, 312], [x, 312], [x, M + 14]], { color: BLUE, dash: true });
      s += pill((56 + x) / 2, 312, 'shortcut (1×1)', d.shortcut.s, { tip: 'from the input of the first module, through a 1×1 convolution' });
      outV = d.shortcut.out;
      x += 14;
    }
    s += wire([[x, M], [x + 50, M]]);
    s += pill(x + 25, M - 22, '', outV, { tip: 'the value drawn in this channel\'s map' });
    s += txt(x + 56, M + 5, 'out', { size: 14, weight: 600, anchor: 'start' });
    const W = Math.max(940, x + 100);
    return svg(W, d.shortcut ? 330 : 300, s, 'Inception module ' + (o.li + 1) + ', channel ' + (o.ch + 1) + ' (' +
      d.branches.find((b) => b.sel).name + ' branch) at t = ' + o.t + '.');
  }

  /* ------------------------------------------------------ continuous kernel */
  /** A function of Δ drawn over the span: several curves, optional dots at the taps. */
  function deltaPlot(x, y, w, h, curves, R, label) {
    let m = 1e-9;
    curves.forEach((c) => c.v.forEach((v) => { m = Math.max(m, Math.abs(c.scale ? v * c.scale : v)); }));
    const n = curves[0].v.length;
    const X = (q) => x + q / (n - 1) * w, Y = (v) => y + h / 2 - v / m * (h / 2 - 4);
    let s = '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="5" fill="#fff" stroke="#c9d6e6"/>' +
      '<line x1="' + x + '" x2="' + (x + w) + '" y1="' + (y + h / 2) + '" y2="' + (y + h / 2) + '" stroke="#dfe4ea"/>' +
      '<line x1="' + (x + w / 2) + '" x2="' + (x + w / 2) + '" y1="' + y + '" y2="' + (y + h) + '" stroke="#e0342b" stroke-dasharray="2 3"/>';
    curves.forEach((c) => {
      s += '<path d="' + c.v.map((v, q) => (q ? 'L' : 'M') + X(q).toFixed(1) + ' ' + Y(c.scale ? v * c.scale : v).toFixed(1)).join(' ') +
        '" fill="none" stroke="' + c.color + '" stroke-width="' + (c.width || 1.4) + '"' + (c.dash ? ' stroke-dasharray="4 3"' : '') + '/>';
      if (c.dots) c.v.forEach((v, q) => { s += '<circle cx="' + X(q).toFixed(1) + '" cy="' + Y(v).toFixed(1) + '" r="1.4" fill="' + c.color + '"/>'; });
    });
    s += txt(x, y + h + 12, '−' + R, { size: 9, color: MUTED, mono: true, anchor: 'start' }) +
      txt(x + w / 2, y + h + 12, 'Δ = 0 (t)', { size: 9, color: '#e0342b', mono: true }) +
      txt(x + w, y + h + 12, '+' + R, { size: 9, color: MUTED, mono: true, anchor: 'end' });
    if (label) s += txt(x, y - 7, label, { size: 10.5, color: MUTED, anchor: 'start' });
    return s;
  }

  /**
   * One continuous-kernel convolution output.
   * @param d { net[K], mask[K], w[K], xw[K], R, sigma, flex, eff, ci, cin, sum, bias, z, a, actName, actExpr, chSums[] }
   * @param o { li, ch, t }
   */
  function cconv(d, o) {
    let s = '';
    const PX = 330, PW = 560;
    // kernel network → mask → kernel
    s += pill(56, 86, 'Δt', 'ms', { tip: 'the time offset from t — a number, not a tap index' });
    s += wire([[84, 86], [118, 86]]);
    s += wbox(166, 86, 'KernelNet', { w: 92, h: 32, tip: 'two sine layers of ' + CC_HIDDEN + ' and a linear output: w(Δ) for every channel pair at any Δ' });
    s += txt(166, 116, 'sin → sin → W₃', { size: 9.5, color: MUTED });
    s += wire([[212, 86], [246, 86]]);
    s += op(260, 86, 'mul', { tip: d.flex ? 'times the Gaussian mask exp(−½ (u/σ)²), σ = ' + num(d.sigma, 3) + ' of the span — FlexConv learns how far the kernel reaches'
      : 'FlexConv is off: the kernel fills the whole span' });
    s += txt(260, 116, d.flex ? 'mask σ=' + num(d.sigma, 2) : 'no mask', { size: 9.5, color: MUTED });
    s += wire([[274, 86], [PX - 4, 86]]);
    s += '<g class="blk">' + tip('kernel from input channel ' + (d.ci + 1) + ' to output channel ' + (o.ch + 1) +
      '\nthin: KernelNet, dashed: mask, bold: the weight w(Δ) — ' + d.eff + ' of ' + (2 * d.R + 1) + ' taps above 10 % of the mask') +
      deltaPlot(PX, 40, PW, 92, [
        { v: d.net, color: '#9fbbe0', width: 1 },
        { v: d.mask, color: '#98a2ad', dash: true, scale: Math.max(...d.net.map(Math.abs)) || 1 },
        { v: d.w.map((v) => v * (d.wscale || 1)), color: BLUE, width: 2, dots: true },
      ], d.R, 'w(Δ): output ' + (o.ch + 1) + ' ← input ' + (d.ci + 1) + ' — thin: KernelNet, dashed: mask, bold + dots: the taps used') + '</g>';
    // the input around t, on the same Δ axis
    s += txt(56, 196, d.cin > 1 ? 'input ch ' + (d.ci + 1) : 'signal', { size: 11, color: MUTED });
    s += wire([[90, 196], [PX - 4, 196]]);
    s += '<g class="blk">' + tip('the input around t; the products w(Δ)·x(t+Δ) over all taps and all ' + d.cin + ' input channels add up to ' + num(d.sum, 4)) +
      deltaPlot(PX, 160, PW, 72, [{ v: d.xw, color: '#5b6873', width: 1.3 }], d.R, 'x(t+Δ) — the same span of the input') + '</g>';
    // Σ → + b → act → out
    const Y = 286;
    s += wire([[PX + PW / 2, 246], [PX + PW / 2, Y - 14]]);
    s += op(PX + PW / 2, Y, 'Σ', { tip: 'Σ over Δ and over the ' + d.cin + ' input channels = ' + num(d.sum, 4) +
      (d.cin > 1 ? '\nper channel: ' + d.chSums.map((v, i) => 'ch' + (i + 1) + ' ' + num(v, 3)).join(', ') : '') });
    let x = PX + PW / 2 + 14;
    s += wire([[x, Y], [x + 50, Y]]);
    x += 64;
    s += wbox(x, Y - 50, 'b', { w: 40, tip: 'bias = ' + num(d.bias, 4) });
    s += wire([[x, Y - 36], [x, Y - 14]]);
    s += op(x, Y, '+', { tip: 'z = ' + num(d.sum, 4) + ' + ' + num(d.bias, 4) + ' = ' + num(d.z, 4) });
    s += pill(x - 40, Y - 22, '', d.sum);
    s += wire([[x + 14, Y], [x + 54, Y]]);
    x += 92;
    s += gate(x, Y, d.actName, d.a, { w: 72, tip: d.actExpr + '  →  ' + num(d.a, 4) });
    s += wire([[x + 37, Y], [x + 80, Y]]);
    s += pill(x + 58, Y - 22, '', d.a, { tip: 'the value drawn in this channel\'s map' });
    s += txt(x + 86, Y + 5, 'out[' + o.t + ']', { size: 13, weight: 600, anchor: 'start' });
    return svg(Math.max(940, x + 170), Y + 30, s, 'Continuous kernel convolution, layer ' + (o.li + 1) + ', channel ' + (o.ch + 1) + ' at t = ' + o.t + '.');
  }

  /* ------------------------------------------------------ Transformer */
  /** A token's vector of d numbers as a small bar chart; dimension hi is outlined. */
  function vecBars(x, y, w, h, vals, hi, label, tipText) {
    const n = vals.length;
    let m = 1e-6;
    for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(vals[i]));
    const bw = w / n;
    let s = '<g class="blk">' + tip((tipText ? tipText + '\n' : '') +
      Array.from(vals).map((v, i) => 'dim ' + (i + 1) + ' = ' + num(v, 4) + (i === hi ? '   ← selected' : '')).join('\n')) +
      '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="4" fill="#fff" stroke="#c9d6e6"/>' +
      '<line x1="' + x + '" x2="' + (x + w) + '" y1="' + (y + h / 2) + '" y2="' + (y + h / 2) + '" stroke="#dfe4ea"/>';
    for (let i = 0; i < n; i++) {
      const v = vals[i], bh = Math.abs(v) / m * (h / 2 - 3);
      const bx = x + i * bw + bw * 0.18, bwid = bw * 0.64;
      const by = v >= 0 ? y + h / 2 - bh : y + h / 2;
      s += '<rect x="' + bx.toFixed(1) + '" y="' + by.toFixed(1) + '" width="' + bwid.toFixed(1) + '" height="' + Math.max(0.6, bh).toFixed(1) +
        '" fill="' + (v >= 0 ? POS : NEG) + '"' + (i === hi ? ' stroke="#e0342b" stroke-width="1.6"' : '') + '/>';
    }
    if (label) s += txt(x + w / 2, y - 6, label, { size: 10.5, color: MUTED });
    return s + '</g>';
  }

  /**
   * One patch becomes one token.
   * @param d { xs[8], ws[8], b, pos, sum, z, out, start }
   * @param o { t, ch }
   */
  function tfEmbed(d, o) {
    let s = '';
    const cw = 56, x0 = 40, y = 92;
    d.xs.forEach((xv, j) => {
      const cx = x0 + j * (cw + 4), w = d.ws[j];
      s += txt(cx + cw / 2, y - 26, 'x[' + (d.start + j) + ']', { size: 10, color: MUTED, mono: true });
      s += '<g class="blk">' + tip('x[' + (d.start + j) + '] = ' + num(xv, 4) + '\nw[' + j + '] = ' + num(w, 4) + '\nproduct = ' + num(xv * w, 4)) +
        '<rect x="' + cx + '" y="' + (y - 18) + '" width="' + cw + '" height="36" rx="4" fill="#fff" stroke="#c9d6e6"/>' +
        txt(cx + cw / 2, y - 4, num(xv, 2), { size: 10, mono: true, color: '#5b6873' }) +
        txt(cx + cw / 2, y + 11, '×' + num(w, 2), { size: 10, mono: true, color: signColor(w), weight: 600 }) + '</g>';
    });
    s += txt(x0, 36, 'patch ' + o.t + ': samples ' + d.start + '–' + (d.start + 7) + ' — the same 8 weights turn every patch into dimension ' + (o.ch + 1),
      { size: 11, color: MUTED, anchor: 'start' });
    const gridR = x0 + 8 * (cw + 4);
    let x = gridR + 40;
    s += wire([[gridR, y], [x - 14, y]]);
    s += op(x, y, 'Σ', { tip: 'Σ of the 8 products = ' + num(d.sum, 4) });
    s += wire([[x + 14, y], [x + 66, y]]);
    x += 80;
    s += wbox(x, y - 56, 'b', { w: 40, tip: 'bias = ' + num(d.b, 4) });
    s += wire([[x, y - 42], [x, y - 14]]);
    s += op(x, y, '+', { tip: 'Σ + b = ' + num(d.z, 4) });
    s += wire([[x + 14, y], [x + 66, y]]);
    x += 80;
    s += wbox(x, y - 56, 'pos[' + o.t + ']', { w: 60, tip: 'the learned position vector of token ' + o.t + ', dimension ' + (o.ch + 1) + ' = ' + num(d.pos, 4) +
      '\nWithout it every token would look the same wherever it sits in the window.' });
    s += wire([[x, y - 42], [x, y - 14]]);
    s += op(x, y, '+', { tip: 'token value = ' + num(d.z, 4) + ' + ' + num(d.pos, 4) + ' = ' + num(d.out, 4) });
    s += wire([[x + 14, y], [x + 80, y]]);
    s += pill(x + 47, y - 22, '', d.out, { tip: 'dimension ' + (o.ch + 1) + ' of token ' + o.t });
    s += txt(x + 86, y + 5, 'token[' + o.t + ']', { size: 13, weight: 600, anchor: 'start' });
    return svg(x + 170, 150, s, 'Patch embedding of token ' + o.t + ', dimension ' + (o.ch + 1) + '.');
  }

  /**
   * The convolutional tokenizer: the filter's response at each of the 8 samples of
   * the patch, ReLU, and the largest one is kept.
   * @param d { as[8] (after ReLU), zs[8] (before), win, start, max, pos, out, k }
   * @param o { t, ch }
   */
  function tfConvEmbed(d, o) {
    let s = '';
    const cw = 56, x0 = 40, y = 92;
    d.as.forEach((a, j) => {
      const cx = x0 + j * (cw + 4), hot = j === d.win;
      s += txt(cx + cw / 2, y - 26, 's=' + (d.start + j), { size: 10, color: MUTED, mono: true });
      s += '<g class="blk">' + tip('sample ' + (d.start + j) + ': filter ' + (o.ch + 1) + ' gives ' + num(d.zs[j], 4) +
        ', after ReLU ' + num(a, 4) + (hot ? '\n← the largest in this patch' : '')) +
        '<rect x="' + cx + '" y="' + (y - 18) + '" width="' + cw + '" height="36" rx="4" fill="' + (hot ? '#e8f0fc' : '#fff') +
        '" stroke="' + (hot ? BLUE : '#c9d6e6') + '"' + (hot ? ' stroke-width="1.6"' : '') + '/>' +
        txt(cx + cw / 2, y - 4, num(d.zs[j], 2), { size: 10, mono: true, color: '#9aa5b1' }) +
        txt(cx + cw / 2, y + 11, num(a, 2), { size: 10.5, mono: true, color: signColor(a), weight: 600 }) + '</g>';
    });
    s += txt(x0, 36, 'filter ' + (o.ch + 1) + ' (K=' + d.k + ') at each sample of patch ' + o.t + ' — grey: conv + bias, coloured: after ReLU',
      { size: 11, color: MUTED, anchor: 'start' });
    const gridR = x0 + 8 * (cw + 4);
    let x = gridR + 46;
    s += wire([[gridR, y], [x - 26, y]]);
    s += gate(x, y, 'max', d.max, { w: 52, tip: 'the patch keeps its largest response: ' + num(d.max, 4) + ' at sample ' + (d.start + d.win) });
    s += wire([[x + 27, y], [x + 70, y]]);
    x += 84;
    s += wbox(x, y - 56, 'pos[' + o.t + ']', { w: 60, tip: 'the learned position vector of token ' + o.t + ', dimension ' + (o.ch + 1) + ' = ' + num(d.pos, 4) +
      '\nThe max forgets where in the patch the response was; the position says which patch it is.' });
    s += wire([[x, y - 42], [x, y - 14]]);
    s += op(x, y, '+', { tip: 'token value = ' + num(d.max, 4) + ' + ' + num(d.pos, 4) + ' = ' + num(d.out, 4) });
    s += wire([[x + 14, y], [x + 80, y]]);
    s += pill(x + 47, y - 22, '', d.out, { tip: 'dimension ' + (o.ch + 1) + ' of token ' + o.t });
    s += txt(x + 86, y + 5, 'token[' + o.t + ']', { size: 13, weight: 600, anchor: 'start' });
    return svg(x + 170, 150, s, 'Convolutional tokenizer: token ' + o.t + ', dimension ' + (o.ch + 1) + '.');
  }

  /** How much token t attends to each of the T tokens, as bars. */
  function attnBars(x, y, w, h, a, scores, t, causal, head) {
    const T = a.length, bw = w / T;
    let s = '<g class="blk">' + tip('head ' + (head + 1) + ', token ' + t + ' looks at:\n' + Array.from(a).map((v, j) =>
      'token ' + j + ': score ' + (causal && j > t ? '— (future, masked)' : num(scores[j], 3)) + ' → weight ' + num(v, 3)).join('\n')) +
      '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="5" fill="#fff" stroke="#c9d6e6"/>';
    let mx = 1e-6;
    for (let j = 0; j < T; j++) mx = Math.max(mx, a[j]);
    for (let j = 0; j < T; j++) {
      const bx = x + j * bw + bw * 0.15, bwid = bw * 0.7;
      if (causal && j > t) {
        s += '<rect x="' + bx + '" y="' + (y + 4) + '" width="' + bwid + '" height="' + (h - 8) + '" fill="#f3f5f7"/>';
        continue;
      }
      const bh = a[j] / mx * (h - 10);
      s += '<rect x="' + bx.toFixed(1) + '" y="' + (y + h - 4 - bh).toFixed(1) + '" width="' + bwid.toFixed(1) + '" height="' + Math.max(0.8, bh).toFixed(1) +
        '" fill="' + BLUE + '"' + (j === t ? ' stroke="#e0342b" stroke-width="1.6"' : '') + '/>';
    }
    for (const j of [0, 4, 8, 12, T - 1]) s += txt(x + j * bw + bw / 2, y + h + 12, String(j), { size: 9, color: MUTED, mono: true });
    s += txt(x + w / 2, y - 7, 'head ' + (head + 1) + ': how much token ' + t + ' attends to each token' + (causal ? ' (future masked)' : ''),
      { size: 10.5, color: MUTED });
    return s + '</g>';
  }

  /**
   * One encoder layer at token t, following dimension ch.
   * @param d { x, n1, a, scores, head, o, attn, h, n2, z2, out, causal }
   * @param o { li, ch, t }
   */
  function attention(d, o) {
    let s = '';
    const ya = 110, yb = 296;
    // row A: x → LN₁ → attention → Σ a·v → W_o → + x
    s += vecBars(40, ya - 25, 76, 50, d.x, o.ch, 'x_t (token ' + o.t + ' in)', 'token ' + o.t + ' entering the layer');
    s += wire([[118, ya], [146, ya]]);
    s += gate(172, ya, 'LN₁', '', { w: 46, tip: 'layer norm over the ' + d.x.length + ' numbers of this token' });
    s += wire([[196, ya], [226, ya]]);
    s += attnBars(230, ya - 40, 250, 80, d.a, d.scores, o.t, d.causal, d.head);
    s += wire([[482, ya], [516, ya]]);
    s += op(530, ya, 'Σ', { tip: 'o_t = Σ_j a_tj · v_j — the attention-weighted average of the value vectors' });
    s += wire([[544, ya], [566, ya]]);
    s += vecBars(568, ya - 25, 70, 50, d.o, o.ch, 'o_t (heads joined)', 'Σ a·v of every head, side by side');
    s += wire([[640, ya], [664, ya]]);
    s += wbox(684, ya, 'W_o', { w: 38, tip: 'output projection: mixes the heads back into the residual stream' });
    s += wire([[703, ya], [742, ya]]);
    s += op(756, ya, '+', { tip: 'h = x + attention — the residual stream carries x on unchanged' });
    s += wire([[78, ya - 25], [78, 34], [756, 34], [756, ya - 14]], { color: BLUE, dash: true });
    s += txt(400, 28, 'residual: x is carried around the attention', { size: 10.5, color: BLUE });
    s += wire([[770, ya], [800, ya]]);
    s += vecBars(802, ya - 25, 70, 50, d.h, o.ch, 'h_t', 'after attention + residual');
    // row B: h → LN₂ → FFN → + h
    s += wire([[837, ya + 25], [837, ya + 70], [78, ya + 70], [78, yb - 27]]);
    s += vecBars(40, yb - 25, 76, 50, d.h, o.ch, 'h_t', 'after attention + residual');
    s += wire([[118, yb], [146, yb]]);
    s += gate(172, yb, 'LN₂', '', { w: 46, tip: 'layer norm again, before the feed-forward net' });
    s += wire([[196, yb], [236, yb]]);
    s += gate(310, yb, 'W₁ → ReLU → W₂', '', { w: 146, tip: 'feed-forward net applied to every token on its own: ' + d.x.length + ' → ' +
      (2 * d.x.length) + ' → ' + d.x.length + ' numbers' });
    s += wire([[384, yb], [420, yb]]);
    s += vecBars(422, yb - 25, 70, 50, d.z2, o.ch, 'FFN out', 'what the feed-forward net adds');
    s += wire([[494, yb], [520, yb]]);
    s += op(534, yb, '+', { tip: 'y = h + FFN — the residual stream again' });
    s += wire([[78, yb + 25], [78, yb + 48], [534, yb + 48], [534, yb + 14]], { color: BLUE, dash: true });
    s += wire([[548, yb], [578, yb]]);
    s += vecBars(580, yb - 25, 76, 50, d.out, o.ch, 'y_t (token ' + o.t + ' out)', 'leaving the layer');
    s += wire([[658, yb], [720, yb]]);
    s += pill(689, yb - 22, 'dim ' + (o.ch + 1), d.out[o.ch], { tip: 'the value drawn in this dimension\'s box at token ' + o.t });
    s += txt(728, yb + 5, 'out', { size: 14, weight: 600, anchor: 'start' });
    return svg(890, yb + 64, s, 'Encoder layer ' + o.li + ', token ' + o.t + ', following dimension ' + (o.ch + 1) +
      ' (outlined in red in every vector).');
  }

  /* ------------------------------------------------------ autoencoder */
  /** A window as a small line chart: several series over the same axis, an optional marker and shading. */
  function spark(x, y, w, h, series, o) {
    o = o || {};
    let m = 1e-6;
    series.forEach((sr) => { for (let i = 0; i < sr.vals.length; i++) m = Math.max(m, Math.abs(sr.vals[i])); });
    const n = series[0].vals.length;
    const X = (i) => x + i * w / (n - 1), Y = (v) => y + h / 2 - v / m * (h / 2 - 3);
    let s = '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="5" fill="#fff" stroke="#c9d6e6"/>';
    if (o.mask) {
      for (let i = 0; i < n; i++) {
        if (o.mask[i]) s += '<rect x="' + (X(i) - w / n / 2).toFixed(1) + '" y="' + (y + 1) + '" width="' + (w / n + 0.4).toFixed(1) +
          '" height="' + (h - 2) + '" fill="rgba(224,52,43,0.10)"/>';
      }
    }
    s += '<line x1="' + x + '" x2="' + (x + w) + '" y1="' + (y + h / 2) + '" y2="' + (y + h / 2) + '" stroke="#e3e9f1"/>';
    series.forEach((sr) => {
      let d = '';
      for (let i = 0; i < n; i++) d += (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(sr.vals[i]).toFixed(1);
      s += '<path d="' + d + '" fill="none" stroke="' + sr.color + '" stroke-width="' + (sr.width || 1.3) + '"' +
        (sr.dash ? ' stroke-dasharray="3 2"' : '') + '/>';
    });
    if (o.t != null) {
      s += '<line x1="' + X(o.t) + '" x2="' + X(o.t) + '" y1="' + (y + 2) + '" y2="' + (y + h - 2) + '" stroke="#e0342b" stroke-dasharray="2 2"/>';
    }
    if (o.label) s += txt(x, y - 6, o.label, { size: 10.5, color: MUTED, anchor: 'start' });
    return s;
  }

  /**
   * The whole autoencoder for the current window: input → encoder → code → decoder →
   * reconstruction, with the error over time and the numbers at position t.
   * @param d { x, y, code, encLabel, decLabel, encTip, decTip, t, mse, mask, vae }
   */
  function aePipeline(d, o) {
    let s = '';
    const n = d.x.length, err = new Float32Array(n);
    for (let i = 0; i < n; i++) err[i] = (d.x[i] - d.y[i]) * (d.x[i] - d.y[i]);
    s += '<g class="blk">' + tip('the window going in: ' + n + ' samples') +
      spark(24, 70, 170, 76, [{ vals: d.x, color: '#31404e' }], { t: d.t, mask: d.mask, label: 'input x (' + n + ' samples)' }) + '</g>';
    s += wire([[196, 108], [238, 108]]);
    s += gate(300, 108, 'encoder', '', { w: 120, tip: d.encTip });
    s += txt(300, 146, d.encLabel, { size: 10, color: MUTED });
    s += wire([[361, 108], [392, 108]]);
    s += vecBars(396, 83, Math.max(40, Math.min(120, d.code.length * 14)), 50, d.code, -1,
      d.codeLabel || 'code z · k = ' + d.code.length + (d.vae ? ' (μ)' : ''), d.codeTip || 'the bottleneck: everything the decoder gets about this window');
    const cx = 396 + Math.max(40, Math.min(120, d.code.length * 14));
    s += wire([[cx + 2, 108], [cx + 34, 108]]);
    s += gate(cx + 96, 108, 'decoder', '', { w: 120, tip: d.decTip });
    s += txt(cx + 96, 146, d.decLabel, { size: 10, color: MUTED });
    const ox = cx + 166;
    s += wire([[cx + 157, 108], [ox - 4, 108]]);
    s += '<g class="blk">' + tip('grey: the input, blue: what the decoder rebuilt from the ' + d.code.length + ' numbers') +
      spark(ox, 70, 220, 76, [{ vals: d.x, color: '#b6c0ca', width: 1.6 }, { vals: d.y, color: BLUE, width: 1.4 }],
        { t: d.t, mask: d.mask, label: 'reconstruction x̂ (blue) over x (grey)' }) + '</g>';
    // squared error over time, as bars
    let m = 1e-9;
    for (let i = 0; i < n; i++) m = Math.max(m, err[i]);
    const ey = 168, eh = 40;
    s += '<g class="blk">' + tip('(x − x̂)² at every sample — where the decoder could not follow') +
      '<rect x="' + ox + '" y="' + ey + '" width="220" height="' + eh + '" rx="5" fill="#fff" stroke="#c9d6e6"/>';
    for (let i = 0; i < n; i++) {
      const bh = err[i] / m * (eh - 4);
      s += '<rect x="' + (ox + i * 220 / n).toFixed(1) + '" y="' + (ey + eh - 2 - bh).toFixed(1) + '" width="' + (220 / n + 0.2).toFixed(1) +
        '" height="' + bh.toFixed(1) + '" fill="#e0342b" opacity="0.75"/>';
    }
    s += '</g>' + txt(ox, ey + eh + 12, 'squared error (x − x̂)² over time', { size: 10, color: MUTED, anchor: 'start' });
    const T = d.t;
    s += pill(ox + 110, 40, 't = ' + T + ':  x', d.x[T], { tip: 'input at the marked position' });
    s += pill(ox + 110, 230, 'x̂', d.y[T], { tip: 'reconstruction at the marked position' });
    s += pill(ox - 70, 190, 'MSE', d.mse, { d: 5, tip: 'mean of the squared error over the window — the anomaly score' });
    return svg(ox + 240, 250, s, 'The autoencoder on this window: the score is how badly it rebuilds it.');
  }

  /**
   * A hidden patch of the masked autoencoder: the learned [MASK] vector plus the position.
   * @param d { mtok[], m, pos, out, start }
   * @param o { t, ch }
   */
  function maeMask(d, o) {
    let s = '';
    s += txt(40, 30, 'patch ' + o.t + ' is hidden: the encoder gets the learned [MASK] vector instead of samples ' + d.start + '–' + (d.start + 7),
      { size: 11, color: MUTED, anchor: 'start' });
    s += vecBars(40, 62, 160, 56, d.mtok, o.ch, '[MASK] token (learned)', 'one vector, the same for every hidden patch');
    s += wire([[202, 90], [266, 90]]);
    s += op(280, 90, '+', { tip: '[MASK][' + (o.ch + 1) + '] + pos[' + o.t + '][' + (o.ch + 1) + '] = ' + num(d.m, 4) + ' + ' + num(d.pos, 4) + ' = ' + num(d.out, 4) });
    s += pill(280, 152, 'pos', d.pos, { tip: 'the learned position vector of token ' + o.t + ' — all that tells the hidden patches apart' });
    s += wire([[280, 138], [280, 104]]);
    s += wire([[294, 90], [356, 90]]);
    s += pill(400, 90, 'token', d.out, { tip: 'dimension ' + (o.ch + 1) + ' of token ' + o.t });
    return svg(480, 180, s, 'A hidden patch: the [MASK] vector plus the position of the patch.');
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

    // final feature maps (or units, for a network without a time axis)
    s += txt(60, 28, o.head === 'none' ? 'last hidden layer · ' + o.C + ' units'
      : 'last layer · ' + o.C + ' maps × ' + o.L, { size: 11, color: MUTED, anchor: 'start' });
    for (let c = 0; c < rowsC; c++) {
      const y = top + c * rh + (rows - rowsC) * rh / 2;
      s += '<rect x="60" y="' + (y - 8) + '" width="90" height="16" rx="3" fill="#e6effa" stroke="#c9d6e6"/>';
      s += txt(105, y + 4, (o.head === 'none' ? 'unit ' : 'map ') + (c + 1), { size: 10, color: '#5b6873' });
      s += wire([[150, y], [206, mid]], { color: '#8796a6', width: 1 });
    }
    if (o.C > rowsC) s += txt(105, top + rowsC * rh + 6, '+' + (o.C - rowsC) + ' more', { size: 10, color: MUTED });
    s += gate(250, mid, o.headLabel, '', { w: 84, tip: o.headTip, fill: o.head === 'none' ? '#8aa3bf' : null });

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

  return { lstm, gru, rnn, conv, ssm, gnn, output, dense, kan, resblock, inception, tfEmbed, tfConvEmbed, attention, aePipeline, maeMask, spark, cconv };
})();
