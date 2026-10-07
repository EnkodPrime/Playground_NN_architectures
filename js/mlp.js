/* mlp.js — a multilayer perceptron: every unit sees the whole window at once.
 *
 * The first layer has one weight per input sample. There is no sliding kernel
 * and no weight sharing, so a unit is a fixed template over the 128 positions:
 * the same disturbance shifted by a few samples meets different weights. That
 * is the contrast with the convolutional playground.
 */

class MLPNet {
  /**
   * @param {{layers:{units:number}[], activation:string, nClasses:number, inputLen:number}} cfg
   */
  constructor(cfg) {
    this.kind = 'mlp';
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.build();
    this.t = 0;
  }

  build() {
    const cfg = this.cfg;
    this.seq = [];
    this.stages = [];
    this.params = [];
    let nin = cfg.inputLen;
    cfg.layers.forEach((ls, i) => {
      const dense = new Dense(nin, ls.units);
      const act = new Activation(cfg.activation);
      this.seq.push(dense, act);
      this.params.push(dense);
      this.stages.push({
        index: i, dense, C: ls.units, L: 1, pooled: false, snapshot: null, pre: null,
        units: ls.units, mlp: true,
      });
      nin = ls.units;
    });
    this.dense = new Dense(nin, cfg.nClasses);
    this.seq.push(this.dense);
    this.params.push(this.dense);
    this.finalC = nin;
    this.finalL = 1;
    this.headKind = 'none';
  }

  forward(x, keepActs) {
    let a = x, si = 0;
    for (const layer of this.seq) {
      if (layer === this.dense) {
        this.embedding = a;
        a = layer.forward(a);
      } else if (layer.type === 'dense') {
        a = layer.forward(a);
        if (keepActs) this.stages[si].pre = a.slice();
      } else {
        a = layer.forward(a, 1);
        if (keepActs) this.stages[si].snapshot = a.slice();
        si++;
      }
    }
    this.logits = a;
    return this.softmax(a);
  }

  /**
   * Everything the arithmetic panel needs about unit j of hidden layer li.
   * x0 is the window the stored snapshots were computed from (the inspector example).
   */
  unitDetail(li, j, x0) {
    const st = this.stages[li], d = st.dense;
    const x = li === 0 ? x0 : this.stages[li - 1].snapshot;
    const w = d.W.subarray(j * d.nin, (j + 1) * d.nin);
    let z = d.b[j];
    for (let i = 0; i < d.nin; i++) z += w[i] * x[i];
    return { x, w, b: d.b[j], z, a: st.snapshot ? st.snapshot[j] : 0, nin: d.nin };
  }

  softmax(z) { return ConvNet1D.prototype.softmax.call(this, z); }
  backward(probs, target) { return ConvNet1D.prototype.backward.call(this, probs, target); }
  zeroGrads() { ConvNet1D.prototype.zeroGrads.call(this); }
  step(lr, scale, l2) { ConvNet1D.prototype.step.call(this, lr, scale, l2); }
  trainBatch(xs, ys, idx, lr, l2) { return ConvNet1D.prototype.trainBatch.call(this, xs, ys, idx, lr, l2); }
  evaluate(ds, nClasses, limit, rejectFn) {
    return ConvNet1D.prototype.evaluate.call(this, ds, nClasses, limit, rejectFn);
  }
  paramCount() { return ConvNet1D.prototype.paramCount.call(this); }
}
