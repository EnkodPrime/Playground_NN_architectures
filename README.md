# 1D CNN Playground

An interactive playground for **1D convolutional networks**, in the spirit of the
[TensorFlow Playground](https://playground.tensorflow.org/) but for signals instead of
2D points. The task is recognising power-quality disturbances in 50 Hz mains voltage.

Everything runs in the browser. No dependencies, no build step, no server.

**Live:** https://enkodprime.github.io/Playground_NN_architectures/

## Running locally

Open `index.html` directly, or serve the folder:

```bash
python -m http.server 8765
```

## The task

Each example is a window of **128 samples at 3200 Hz** (40 ms ≈ 2 cycles of 50 Hz).
The network classifies which disturbance the window contains:

| Class | Signal model |
|---|---|
| Clean | fundamental only, plus background noise |
| Ripple (SMPS) | 420–900 Hz tone at 6–15% amplitude |
| Harmonics | 3rd and 5th harmonic distortion |
| Impulse / transient | 1–3 damped oscillations at 700–1400 Hz |
| Voltage sag | 22–50% amplitude drop over part of the window |
| EMI burst | short broadband burst with a smooth envelope |

Background noise, disturbance strength and dataset size are adjustable. The test set is
generated separately at 40% of the training size.

## The network

1–4 convolutional layers (`same` padding, stride 1), 1–10 filters each, kernel
K ∈ {3,5,7,9,11}, optional max-pooling ×2; ReLU / Tanh / Leaky ReLU / Abs activation;
a Global Average Pool, Global Max Pool or Flatten head followed by a linear layer and
softmax. Trained with **Adam** and cross-entropy, with optional L2.

Forward and backward passes are written from scratch in `js/nn.js` — convolution,
max-pooling, global pooling, dense layer and softmax, all over flat `Float32Array`s
indexed as `[channel * length + t]`.

## Panels

**Network diagram.** Every filter is a box showing its output map for the current example,
with the learned kernel in the corner. Link thickness and colour encode weight magnitude and
sign. A *Time / Spectrum* switch turns every map into its magnitude spectrum, which is where
band-pass and high-pass filters become obvious.

**Inspector.** Feeds single examples through the trained network without touching the weights.
The class menu includes classes the network was *not* trained on, a quick test runs 50 fresh
examples and reports the distribution of answers, and a CSV box accepts your own samples.

**Arithmetic.** Clicking a node expands the exact computation with live numbers: every `w · x`
product of the convolution, the bias, the pre-activation, the activation, the pooling
comparison, and the contribution of that filter to each class logit. Clicking the output
expands the softmax and the cross-entropy loss term by term.

**Live stream.** A continuously generated signal flows through a ring buffer; the most recent
128 samples are classified on every frame. Disturbances are toggled on the fly, a scope shows
the signal with the analysis window highlighted, and a ribbon shows the decision over time.

**Unknown state.** Softmax is normalised, so it can never say "I don't know". This panel adds a
novelty score over the logits or features — energy, entropy, max-softmax or k-NN — with a
threshold calibrated against the classes you left out, reported as AUC, true-positive and
false-alarm rates.

**Watermark.** A black-box ownership watermark: a key-derived set of physically impossible
trigger signals with pseudo-random labels, embedded during training and verified with an exact
binomial test. Includes pruning and fine-tuning attacks to see how much of it survives.

## Files

| File | Contents |
|---|---|
| `js/signal.js` | signal and dataset generation |
| `js/fft.js` | radix-2 FFT, spectra and kernel frequency response |
| `js/nn.js` | layers, forward/backprop, Adam, evaluation |
| `js/viz.js` | layout and canvas drawing |
| `js/stream.js` | live generator, scope and decision ribbon |
| `js/ood.js` | novelty scores, calibration, AUC, histograms |
| `js/watermark.js` | trigger watermark, binomial test, attacks |
| `js/main.js` | state, UI, training loop, arithmetic panel |

## Contributing

Issues and pull requests are welcome. The project is deliberately dependency-free — clone it,
open `index.html`, and everything is editable in place. Each feature lives in its own file;
comments explain the maths rather than the syntax.

## Licence

MIT — see [LICENSE](LICENSE).
