# NN Architecture Playground

An interactive playground for neural network architectures on **signals**, in the spirit of the
[TensorFlow Playground](https://playground.tensorflow.org/) but for time series instead of 2D
points. The task is recognising power-quality disturbances in 50 Hz mains voltage.

Nine architectures share the same data, metrics and tooling, switchable at the top of the page:

* **MLP** — fully connected layers over the raw window, the baseline with no structure at all
* **1D CNN** — convolutional filters over the window, optionally with residual skips
* **ResNet-1D** — residual blocks of three convolutions with normalisation (Wang et al. 2017)
* **InceptionTime** — parallel convolutions of three lengths plus a pooling branch (Ismail Fawaz et al. 2020)
* **RNN** — recurrent cells over the same window: simple tanh RNN, GRU or LSTM
* **S4 / Mamba** — state space models: S4D (diagonal, time-invariant) and a Mamba-style
  selective SSM
* **GNN** — message passing over the visibility graph the signal builds for itself
* **Transformer** — self-attention over 16 patches of the window, as an encoder or, causal, a decoder
* **KAN** — a Kolmogorov–Arnold network: a learned function on every edge instead of a weight

Everything runs in the browser. No dependencies, no build step, no server.

A second page, **[Autoencoder](https://enkodprime.github.io/Playground_NN_architectures/autoencoder.html)**,
trains networks that learn what normal mains looks like without any labels — see
[Autoencoder playground](#autoencoder-playground) below.

**Live:** https://enkodprime.github.io/Playground_NN_architectures/

## Running locally

Open `index.html` directly, or serve the folder:

```bash
python -m http.server 8765
```

## Training runs

**Run for** next to the transport buttons sets how many epochs one press of ▶ trains for. Pick a
preset from the list (50, 100, 200, 500, 1000) or type any number; leave it empty and ▶ runs until
you press pause, as before. While a fixed run is going the epoch counter shows the epoch it is
heading for, and training stops there on its own so two configurations can be compared over
exactly the same number of epochs. Pausing keeps the target — ▶ resumes the same run rather than
starting another one — and ⟳ clears it along with the weights.

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

**MLP.** 1–3 dense layers of 1–10 units over all 128 samples, with the same activations as the
convolutional net. A first-layer unit has one weight per sample: a template fixed to positions in
the window, which the diagram draws inside its box. Nothing is shared across positions, and that
is the point of the comparison — on *Clean / Ripple* a 118-parameter CNN reaches about 99% test
accuracy in 60 epochs, while an 8-unit MLP with 1,050 parameters stays near chance and, trained
longer, memorises the training windows (above 90% training accuracy, 60–70% test).

**Convolutional.** 1–8 layers (`same` padding, stride 1), 1–10 filters each, kernel
K ∈ {3,5,7,9,11}, optional max-pooling ×2; ReLU / Tanh / Leaky ReLU / Abs activation;
a Global Average Pool, Global Max Pool or Flatten head. With **residual (ResNet)** on, every
layer after the first becomes a residual block (y = act(conv(x)) + x, so the identity path is untouched); when the channel count changes, the
skip is a learned 1×1 convolution. The diagram draws the skips as arcs under the columns. At
the depths offered here (up to 8 layers) the plain stack still trains with Adam and the gradient
at layer 1 is about as large without skips as with them, so the toggle shows the structure more
than an accuracy gain; with ReLU the non-negative branch makes values grow with depth, which is
what batch normalisation fixes in real ResNets.

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

**ResNet-1D.** 1–3 residual blocks of 2–10 channels, the standard deep baseline for time series
classification, scaled down: in every block conv → norm → ReLU for kernels 8, 5 and 3 (or 7-5-3,
5-3, 3-3-3), the last ReLU after the skip is added; the skip is the identity, or a 1×1 convolution
with its own norm when the width changes. Then Global Average Pooling and a linear layer. The
original uses batch norm, which needs a whole mini-batch and couples the examples in its gradient;
this engine trains one example at a time, so the blocks use layer norm (GroupNorm with one group,
exact gradient). A batch norm on running averages was tried first and made training worse. Measured
with 3 blocks on four classes, 20 epochs: layer norm and skips 99–100%, layer norm alone 96–97%, no
normalisation 69–91%.

**InceptionTime.** 1–3 modules: a 1×1 bottleneck (when there is more than one input channel),
convolutions of length 5, 11 and 23 side by side (10/20/40 in the original, for longer series), and
a max-pool branch with its own 1×1 convolution, stacked, normalised and passed through ReLU; 1–3
filters per branch. With three modules one shortcut runs around all of them, as in the original
(one every three modules). Each box is tagged with its branch, and hovering one shows how strongly
each branch answers the current example — after training on all eight classes the K=23 branch is
the most active for harmonics. It is the slowest of the networks to get going: the loss can sit at
equal odds for several epochs, and about one run in five needs a reset.

**Transformer.** The window is cut into 16 patches of 8 samples (2.5 ms; one 50 Hz cycle is 8 patches),
each a token of d ∈ {4, 8, 12} numbers plus a learned position vector. Two tokenizers: by default the
convolutional one of the Compact Convolutional Transformer (Hassani et al. 2021) — d filters of length 7,
ReLU, and the largest response of each in every patch — or, for comparison, the linear patch embedding of
ViT, which maps the 8 raw samples straight to d numbers. 1–3 pre-LN encoder layers follow — x +
Attention(LN(x)), then x + FFN(LN(x)) — with 1, 2 or 4 heads (never fewer than two numbers per head) and
an FFN twice as wide, then a final layer norm, the mean over the tokens and a linear layer. With *causal*
on, a token attends only to itself and the past: the decoder form a Transformer uses on a stream. The
attention map of every encoder layer is drawn above its column (mean over the heads; each head in the
hover view), and clicking follows one token through the layer. The tokenizer decides how well it learns
from 480 windows: on four classes after 30 epochs 68–79% with conv tokens, 55–63% with linear patches
(58–70% for the 2×4 CNN); on all eight classes 59% against 38% for the CNN. With linear patches the
attention of layer 1 picks out the patch with an impulse; with conv tokens it does not need to.

**KAN.** 1–3 Kolmogorov–Arnold layers of 1–10 nodes, then a linear layer to the classes. Every
edge carries φ(x) = w_b·silu(x) + Σ c_m·B_m(x), with cubic B-splines on 5 grid intervals over
−2 … 2 — nine numbers per edge — and the node only sums its edges. Deeper layers read their
inputs standardised with running statistics, a simple form of the paper's grid update: a node
summing 128 edges easily reaches ±40, far outside the grid where every spline is zero. The
diagram draws each learned edge function with the current input marked on it. On raw samples
the KAN shares the MLP's weakness and overfits harder (4,608 first-layer parameters).

**Graph.** A single window comes with no graph, so one is derived from it: in a horizontal
visibility graph two samples are connected when everything between them is lower than both
(Luque et al. 2009). Shape becomes topology — a clean sine yields an almost regular graph with
max degree around 8, while a lone impulse becomes a hub of degree 16. Node features are only the
value, its step difference and its magnitude; 1–3 message passing layers then aggregate over
neighbours by mean, max or sum. The inspector draws the graph as an arc diagram above the signal.

All of them feed a linear layer and softmax, and are trained with **Adam** and cross-entropy plus
optional L2. Forward and backward passes are written from scratch in `js/nn.js`, `js/rnn.js`,
`js/ssm.js`, `js/mlp.js`, `js/kan.js`, `js/blocks.js` and `js/transformer.js` — convolution, residual
and Inception blocks, group/layer normalisation, multi-head self-attention, pooling, three recurrent
cells, both state space variants,
B-spline edges, dense layer and softmax, all over flat
`Float32Array`s indexed as `[channel * length + t]`. Every gradient,
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

**Data-flow diagram.** Clicking a neuron draws its cell right under the network, inside the
Architecture frame, as a block diagram — gate boxes, multiply and add nodes, wires with arrows — with the value that flows along
every wire at the current step: the LSTM cell with its memory line and four gates, the GRU with
its reset and update gates, the plain tanh RNN, the convolution window feeding Σ, bias,
activation and pooling, the S4D / Mamba recurrence with its feedback through Ā and the D skip
(and, for Mamba, Δ(t) steering Ā and B̄ and the gate branch), message passing in the GNN, and
the output head from the last maps to the class probabilities. Hovering a block shows its
formula with the current numbers filled in, so every value can be traced back to where it came
from. A slider next to the diagram moves the step t; the full tables stay in the arithmetic panel
further down.

**Layout.** The dividers between Data, Architecture and Results can be dragged (or moved with
the arrow keys) to give the network more room; a double click puts the default back, and the
widths are remembered in the browser.

**Live stream.** A continuously generated signal flows through a ring buffer; the most recent
128 samples are classified on every frame. Disturbances are toggled on the fly, a scope shows
the signal with the analysis window highlighted, and a ribbon shows the decision over time.

**Unknown state.** Softmax is normalised, so it can never say "I don't know". This panel adds a
novelty score over the logits or features — energy, entropy, max-softmax or k-NN — with a
threshold calibrated against the classes you left out, reported as AUC, true-positive and
false-alarm rates.

**Quantisation.** Post-training weight quantisation as a live experiment: training stays in
float32 while a quantised copy is evaluated beside it on every metrics update, so the cost of each
bit width is visible as the network learns. Bit width 2–16, per-tensor or per-channel scales,
symmetric or asymmetric, with a bit sweep, the signal-to-quantisation-noise ratio of the worst
tensor and the resulting weight memory. A weight histogram is drawn with the quantisation levels
over it. Switching on *keep weights quantised* leaves the rounded values in place, so the kernels
in the diagram and every number in the arithmetic panel become what a fixed-point device would
actually use.

**Watermark.** A black-box ownership watermark: a key-derived set of physically impossible
trigger signals with pseudo-random labels, embedded during training and verified with an exact
binomial test. Includes pruning and fine-tuning attacks to see how much of it survives.

## Autoencoder playground

`autoencoder.html` squeezes the 128-sample window through a bottleneck of k numbers and rebuilds
it from them. Trained on clean mains only, the network rebuilds normal windows well and
everything else badly, so the reconstruction error is an anomaly score that needed no labels.

* **MLP AE** — dense 128 → hidden (tanh) → k → hidden (tanh) → 128
* **Conv AE** — three convolution + ReLU + max-pool steps (128 → 64 → 32 → 16), a dense layer to
  the code, and back with upsampling + convolution
* **LSTM AE** — sequence to sequence: an LSTM (or GRU) reads the window sample by sample, its last
  state becomes the code, and a second one unrolls the window from the code repeated at every step
* **Transformer MAE** — a masked autoencoder: 16 patches of 8 samples, some replaced by a learned
  [MASK] vector, and self-attention fills them in from the rest. No bottleneck; to score a window
  it runs four times, each time hiding every fourth patch, so every patch is predicted unseen

Options: code size k (1–16), hidden units, filters, cell and units, encoder layers and the share
of patches masked in training, a **VAE** switch with its KL weight β, *Train on* clean mains only or every
checked class, a **denoising** mode (noisier input, noise-free target) and the same learning rate,
batch and *Run for* controls as the classifier. The diagram draws the encoder, the code and the
decoder; clicking a unit, a code number or an output sample opens its arithmetic and data-flow
diagram, and the output box shows the reconstruction over the input with the squared error
underneath.

Results panel: training and clean-test error, an **AUC per class** (how often a window of the
class scores a larger error than a clean one), how often the largest error falls inside the
disturbance, the denoising SNR against a moving average, and the **latent space** (the codes
themselves for k = 2, a PCA projection otherwise). An experiment below the arithmetic panel
retrains for k = 1, 2, 3, 4, 8, 16 and plots error and AUC against code size.

Measured on all 8 classes, 40 epochs, trained on clean mains only:

| | mean AUC (k = 2) | hardest classes | largest error inside the disturbance |
|---|---|---|---|
| MLP AE | 0.976–0.993 | over / under 0.91–0.99 | 97–100% |
| Conv AE | 0.954–0.987 | over / under 0.87–0.88 in the weaker run | 89–90% |
| LSTM AE | 0.81–0.95 (0.96 after 100 epochs) | burst, ripple, drift | 59–92% |
| Transformer MAE | 0.82–0.97 (0.94–0.97 after 100) | over / under | 94–100% |

The MAE has no code size; its row is for 50% of the patches hidden in training.

* At k = 2 the clean windows form a **ring** in the latent space — amplitude and phase. With k = 1
  the clean error stays about 40 times larger and the mean AUC is about 0.6 (MLP, 20 epochs:
  k = 1 → 0.60, 2 → 0.91, 4 → 0.99, 16 → 0.99).
* Training on every class instead of clean only lowered the mean AUC just a little (MLP: 0.970 at
  k = 2, 0.975 at k = 16); the frequency drifts lose most.
* The sequence models are slower (LSTM about 0.6 s per epoch, MAE 0.17 s, MLP 0.02 s) and vary
  more from run to run; a run whose clean error stays high also stays low in AUC.
* **Generate** decodes codes drawn from N(0, 1). At k = 8 a plain autoencoder turns them into
  windows of which only 38% (MLP) or 41% (conv) of the energy is a 50 Hz sine; a VAE (β = 0.01)
  gives 98% and 93%. The cost is detection: mean AUC 0.99 → 0.91 (MLP) and 0.98 → 0.89 (conv).
* Denoising, k = 8, added noise 0.1: the MLP raised the SNR from 16.5 to 26.6 dB against 23.0 dB
  for a 5-point moving average; the convolutional autoencoder reached 22.2 dB, no better than the
  filter.

## Files

| File | Contents |
|---|---|
| `js/signal.js` | signal and dataset generation |
| `js/fft.js` | radix-2 FFT, spectra and kernel frequency response |
| `js/nn.js` | convolutional layers, residual blocks, forward/backprop, Adam, evaluation |
| `js/rnn.js` | RNN / GRU / LSTM cells, BPTT, gradient clipping, readouts |
| `js/ssm.js` | S4D and selective (Mamba) state space layers, kernel extraction |
| `js/gnn.js` | visibility graph construction, message passing, arc diagram |
| `js/mlp.js` | multilayer perceptron over the raw window |
| `js/kan.js` | Kolmogorov–Arnold layers: B-spline edges, running grid statistics |
| `js/blocks.js` | ResNet-1D and InceptionTime: residual blocks, Inception modules, group normalisation |
| `js/transformer.js` | patch embedding, multi-head self-attention (optionally causal), pre-LN encoder layers |
| `js/viz.js` | layout and canvas drawing |
| `js/stream.js` | live generator, scope and decision ribbon |
| `js/ood.js` | novelty scores, calibration, AUC, histograms |
| `js/watermark.js` | trigger watermark, binomial test, attacks |
| `js/quant.js` | post-training weight quantisation, bit sweep, histograms |
| `js/main.js` | state, UI, training loop, arithmetic panel |
| `js/flow.js` | data-flow diagrams of the selected node, drawn as SVG |
| `js/ui.js` | layout shared by both pages: resizable columns |
| `js/ae-models.js` | autoencoders: MLP, convolutional, LSTM / GRU sequence to sequence, masked Transformer; upsampling, bottleneck (plain or variational) |
| `js/ae-viz.js` | autoencoder diagram: units, code, maps, state and token heatmaps, attention, the reconstruction box |
| `js/ae-main.js` | autoencoder page: training, AUC, localisation, latent space, generation, code-size sweep |

## Contributing

Issues and pull requests are welcome. The project is deliberately dependency-free — clone it,
open `index.html`, and everything is editable in place. Each feature lives in its own file;
comments explain the maths rather than the syntax.

## Licence

MIT — see [LICENSE](LICENSE).
