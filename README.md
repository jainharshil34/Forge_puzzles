# DataForge — Learn It or Remember It

> **Laboratory instrument comparing two fundamental machine adaptation paradigms: in-context memory vs iterative gradient optimization.**

Inspired by research tools like *TensorFlow Playground* and *Neuronpedia*, **DataForge** is a precise benchmarking suite and interactive evaluation instrument designed to demonstrate how artificial models solve abstract reasoning tasks (such as the Abstraction and Reasoning Corpus, ARC).

---

## Concept: "Learn It or Remember It"

A machine adapts to new tasks through two distinct mechanisms:
1. **The Context / Memory Route (Teal `#5EEAD4`)**: Ingests demonstration pairs in-context, dynamically strengthening associative pathways in a synaptic state graph. Holds memory instantly with zero weight updates or gradient descent overhead.
2. **The Optimization / Gradient Route (Coral `#F2967D`)**: Adapts by backpropagating loss gradients sequentially across training epochs. Computes iterative weight adjustments, incurring cumulative latency and risk of catastrophic forgetting under distribution shifts.

---

## Design System & Register

- **Palette**:
  - Base Background: `#0F1115`
  - Surface Background: `#171A21`
  - Primary Text: `#E8E6DF`
  - Secondary Text: `#8B909C`
  - Accent A (Context / Memory): `#5EEAD4` (Teal)
  - Accent B (Optimization / Gradient): `#F2967D` (Coral)
  - Hairline Borders: `#2A2E38` (1px, 4px border-radius, zero heavy shadows)
  - *Strict Boundary*: Accent A and Accent B are never mixed within the same visual element.
- **Typography**:
  - Headings / UI Labels: **Space Grotesk** (strictly sentence case)
  - Body Text: **Inter**
  - Metrics / Telemetry / Coordinates / Tooltips: **IBM Plex Mono**
- **Motion Philosophy**:
  - Fluid spring physics with ~15ms topological stagger propagation for memory graph activations.
  - Stepped, mechanical 150ms ease-out draws for gradient loss trajectories.
  - Strict adherence to `prefers-reduced-motion` across all components.

---

## Key Features

- **Asymmetric Two-Instrument Comparison View**: Differentiated research instrument panels for Context vs Optimization paths with live telemetry readouts.
- **`<GridDisplay>` & `<DiffGridDisplay>`**: High-precision $N \times N$ matrix rendering with instant 11px monospace cell inspection tooltips and functional 200ms cross-fade diff-spotting on hover.
- **`<StateVectorView>`**: Associative node-and-edge "synaptic memory" graph with spring physics (`stiffness: 300, damping: 15`) and staggered propagation when demonstration pairs are ingested.
- **`<LossCurveView>`**: Minimalist line chart with hairline baseline axes animating step-by-step gradient descent updates mechanically.
- **Interactive Control Suite**:
  - **Demo-Count Slider (1–5)**: Custom thin track with neutral handle, 1.15x hover scale, and floating drag indicator.
  - **Novelty Switch (Familiar / Novel)**: Two-state pill switch with subtle background preview and a 200ms difficulty vignette shift.
  - **"Break It" Action Button**: Triggers adversarial evaluation, causing the optimization model to fail with 1px coral outlines around specifically erroneous cells while the context model remains robust.
  - **`LIVE` / `PRECOMPUTED` Badges**: Dynamic status indicators reflecting backend checkpoint provenance with plain-language tooltips.
- **Full Keyboard Accessibility**: High-contrast `2px solid #E8E6DF` focus rings on all interactive elements.
- **Responsive Mobile Layout**: Gracefully stacks multi-column instruments vertically on narrow viewports without layout distortion.

---

## Live vs. Illustrative Elements

To maintain complete research honesty and transparency:

- **100% Live Neural Inference**:
  - All grid predictions, output logits, confidences, latencies, gradient descent loss histories (`loss_curve`), and cell-level error diffs are executed live against real PyTorch checkpoints in `hubdk17/Forge_puzzle` (`context_model/checkpoint.pt` and `optimization_model/checkpoint.pt`).
  - Dynamic cell failure outlines in `<DiffGridDisplay>` are computed cell-by-cell in real time by comparing `predicted_output[r][c] !== groundTruth[r][c]`.

- **Dimensionality-Reduction Visualization**:
  - The recurrent context model outputs a high-dimensional 128-d latent state vector `h` at each demonstration step.
  - `<StateVectorView>` utilizes a deterministic dimensionality-reduction projection (`src/lib/stateMapper.ts`): the 128-d vector is partitioned into 10 contiguous feature segments, taking the mean absolute value of each segment to drive the 10 topological node activations ($n_0 \dots n_9$) and connecting edge weights. This makes synaptic associative memory updates visually intuitive without attempting to render raw 128-coordinate vectors directly.

---


## API & Architecture

The application is built on **Next.js 16 (Turbopack, App Router, TypeScript)** with API endpoints providing normalized contracts:

- `GET /api/puzzles`: Returns the benchmark puzzle suite.
- `GET /api/puzzles/:id`: Returns puzzle metadata, demonstration pairs, test input, and ground truth.
- `GET /api/predict`: Returns dynamic or precomputed predictions with `state_snapshot`, `loss_curve`, and `incorrect_cells` payloads.
- `GET /api/sweep`: Returns cross-model latency and accuracy trends.

Client integration is exposed via `src/lib/mockApi.js` preserving exact function signatures:
- `getPuzzle(id)`
- `getPrediction(puzzleId, modelType, demoCount, novelty, isAdversarial)`
- `getSweep()`

---

## Getting Started

### Prerequisites
- Node.js 18.18+ or Node.js 20+
- npm / yarn / pnpm

### Installation

```bash
# Clone repository
git clone https://github.com/jainharshil34/Forge_puzzles.git
cd Forge_puzzles

# Install dependencies
npm install
```

### Running Locally

```bash
# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production

```bash
# Build optimized production bundle with TypeScript checks
npm run build

# Start production server
npm run start
```

---

## License

MIT
