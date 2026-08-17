# NN Architecture Playground

An interactive playground for neural network architectures on **signals**, in the spirit of the
[TensorFlow Playground](https://playground.tensorflow.org/) but for time series instead of 2D
points. The task is recognising power-quality disturbances in 50 Hz mains voltage.

Three architectures share the same data, metrics and tooling, switchable at the top of the page:

* **1D CNN** — convolutional filters over the window
* **RNN** — recurrent cells over the same window: simple tanh RNN, GRU or LSTM
* **S4 / Mamba** — state space models: S4D (diagonal, time-invariant) and a Mamba-style
  selective SSM

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

## The networks

**Convolutional.** 1–4 layers (`same` padding, stride 1), 1–10 filters each, kernel
K ∈ {3,5,7,9,11}, optional max-pooling ×2; ReLU / Tanh / Leaky ReLU / Abs activation;
a Global Average Pool, Global Max Pool or Flatten head.

**Recurrent.** 1–2 layers, 1–10 units each, optionally bidirectional; simple tanh RNN, GRU or
LSTM cell; readout by last state, mean over time or max over time. Trained by backpropagation
through all 128 steps, with global gradient-norm clipping — without it the loss oscillates
instead of converging.

**State space.** 1–2 layers, 1–10 channels each, state dimension N ∈ {4,8,16}.

*S4D* keeps a complex diagonal `A` with the S4D-Lin initialisation and discretises by zero-order
hold: `Ā = exp(ΔA)`, `B̄ = (Ā−1)/A`. Nothing depends on the input, so the layer is exactly one FIR
kernel of the full window length — the inspector computes and draws that kernel and its frequency
response from the state modes.

*Mamba* keeps a real diagonal `A` but produces Δ, B and C from the input at every step, so the
model is no longer time-invariant and has no fixed kernel; the inspector shows Δ(t) instead. The
block also carries the short causal depthwise convolution and the SiLU gate branch of the original,
without which a real-diagonal state is only a running average and cannot resolve a frequency.

All three feed a linear layer and softmax, and are trained with **Adam** and cross-entropy plus
optional L2. Forward and backward passes are written from scratch in `js/nn.js`, `js/rnn.js` and
`js/ssm.js` — convolution, pooling, three recurrent cells, both state space variants, dense layer
and softmax, all over flat `Float32Array`s indexed as `[channel * length + t]`. Every gradient,
including the complex chain rule through `Ā` and `B̄`, agrees with numeric finite differences to
within 1% at the full sequence length.

Sequence models need care that the convolutional one does not: Adam gets global gradient-norm
clipping, and both SSM blocks need their nonlinearity — a state space recurrence is linear, so
without it the whole stack collapses into a single linear map and never leaves chance level.

## Panels

**Network diagram.** Every box is a filter (CNN) or a hidden unit (RNN), showing its output map
or its state h(t) for the current example. Link thickness and colour encode weight magnitude and
sign. A *Time / Spectrum* switch turns every map into its magnitude spectrum, which is where
band-pass and high-pass filters become obvious. Selecting a node highlights what it actually sees:
a fixed receptive field for a convolution, everything up to t for a recurrent unit.

**Inspector.** Feeds single examples through the trained network without touching the weights.
The class menu includes classes the network was *not* trained on, a quick test runs 50 fresh
examples and reports the distribution of answers, and a CSV box accepts your own samples.

**Arithmetic.** Clicking a node expands the exact computation with live numbers. For a filter:
every `w · x` product of the convolution, the bias, the pre-activation, the activation, the
pooling comparison. For a recurrent unit: what each gate computes at step t, the weights behind
those sums, and the state update — `c = f·c + i·g` for LSTM, `h = (1−z)·n + z·h` for GRU. Both end
with the contribution to each class logit; clicking the output expands softmax and the loss.

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
| `js/nn.js` | convolutional layers, forward/backprop, Adam, evaluation |
| `js/rnn.js` | RNN / GRU / LSTM cells, BPTT, gradient clipping, readouts |
| `js/ssm.js` | S4D and selective (Mamba) state space layers, kernel extraction |
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
