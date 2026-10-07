/* ui.js — layout helpers shared by both playgrounds (classifier and autoencoder).
 * Each page defines renderNet() and renderMetrics(); the resizable columns
 * call them to redraw after a drag.
 */

/* ------------------------------------------------- resizable columns */
const COLS = { left: 240, right: 280 };                 // default widths, px
const COL_LIMITS = { left: [190, 400], right: [230, 440] };
const CENTER_MIN = 520;

/** Applies the column widths, keeping the Architecture frame at least CENTER_MIN wide. */
function applyCols(cols) {
  const grid = document.querySelector('main.grid');
  const room = grid.clientWidth - 40 - 28 - CENTER_MIN;  // side padding, two gutters
  const fit = (side, v) => Math.round(Math.max(COL_LIMITS[side][0], Math.min(COL_LIMITS[side][1], v)));
  let l = fit('left', cols.left), r = fit('right', cols.right);
  if (l + r > room) {                                    // a narrow window: the sides give way
    const over = l + r - room;
    const take = Math.min(over, r - COL_LIMITS.right[0]);
    r -= take;
    l = Math.max(COL_LIMITS.left[0], l - (over - take));
  }
  grid.style.setProperty('--colL', l + 'px');
  grid.style.setProperty('--colR', r + 'px');
  return { left: l, right: r };
}

/**
 * Drag a gutter to trade width between Data, Architecture and Results; arrow
 * keys nudge it and a double click puts the default back. The widths are
 * remembered in this browser only.
 */
function bindGutters() {
  // `want` is what the reader chose; `cols` is what fits the window right now
  let want = { ...COLS };
  try {
    const saved = JSON.parse(localStorage.getItem('pgCols') || 'null');
    if (saved && saved.left && saved.right) want = saved;
  } catch (_) { /* storage blocked: keep the defaults */ }
  let cols = applyCols(want);

  let pending = false;
  const redraw = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; renderNet(); renderMetrics(); });
  };
  const save = () => {
    try { localStorage.setItem('pgCols', JSON.stringify(want)); } catch (_) { /* not remembered */ }
  };
  const set = (side, v) => {
    cols = applyCols({ ...cols, [side]: v });
    want = { ...want, [side]: cols[side] };
    redraw();
  };

  document.querySelectorAll('.gutter').forEach((g) => {
    const side = g.dataset.side;
    g.addEventListener('pointerdown', (ev) => {
      ev.preventDefault();
      g.setPointerCapture(ev.pointerId);
      g.classList.add('drag');
      document.body.classList.add('resizing');
      const x0 = ev.clientX, w0 = cols[side];
      const move = (e) => set(side, side === 'left' ? w0 + (e.clientX - x0) : w0 - (e.clientX - x0));
      const up = () => {
        g.removeEventListener('pointermove', move);
        g.removeEventListener('pointerup', up);
        g.removeEventListener('pointercancel', up);
        g.classList.remove('drag');
        document.body.classList.remove('resizing');
        save();
      };
      g.addEventListener('pointermove', move);
      g.addEventListener('pointerup', up);
      g.addEventListener('pointercancel', up);
    });
    g.addEventListener('dblclick', () => { set(side, COLS[side]); save(); });
    g.addEventListener('keydown', (ev) => {
      const step = ev.shiftKey ? 40 : 10;
      const grow = side === 'left' ? 'ArrowRight' : 'ArrowLeft';
      const shrink = side === 'left' ? 'ArrowLeft' : 'ArrowRight';
      if (ev.key === grow) set(side, cols[side] + step);
      else if (ev.key === shrink) set(side, cols[side] - step);
      else return;
      ev.preventDefault();
      save();
    });
  });
  window.addEventListener('resize', () => { cols = applyCols(want); redraw(); });
}
